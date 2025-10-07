import os
import torch
import torch.nn as nn
import timm
from Helpers import load_hf_model_or_local, download_swin
from safetensors.torch import load_file as load_safetensor
from pathlib import Path

try:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
except NameError:
    BASE_DIR = Path.cwd().parent.parent

MODEL_PLACE = BASE_DIR / 'models'
os.environ["TRANSFORMERS_CACHE"] = str(MODEL_PLACE)
os.environ["TORCH_HOME"] = str(MODEL_PLACE)

class PreFusionEnhancer(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1, max_len=512):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=0.1)
        self.norm1 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x):  # x: (B, L, D)
        B, L, D = x.shape
        x = x + self.pos_embed[:, :L]  # Add trainable positional embedding
        x2, _ = self.self_attn(x, x, x) # Self attention for each token
        x = self.norm1(self.alpha * x + self.dropout(x2))
        return x

class Backbones(nn.Module):
    """
    Image and text backbones: Swin Transformer and ClinicalBERT.
    Outputs global image features and pooled CLS text embeddings.
    """
    def __init__(
            self, 
            img_backbone='swin',
            swin_model_name='swin_base_patch4_window7_224', 
            cnn_model_name='resnet50',
            bert_model_name='emilyalsentzer/Bio_ClinicalBERT',
            swin_checkpoint_path=None,
            bert_local_dir=None,
            pretrained=True,
            img_dim=None,
            txt_dim=None
            ):
        """
        Constructor for Backbones.

        Args:
            img_backbone (str): The image backbone to use. 
                One of 'swin' or 'cnn'. Defaults to 'swin'.
            swin_model_name (str): The Swin Transformer model name.
                Defaults to 'swin_base_patch4_window7_224'.
            cnn_model_name (str): The CNN model name.
                Defaults to 'resnet50'.
            bert_model_name (str): The ClinicalBERT model name.
                Defaults to 'emilyalsentzer/Bio_ClinicalBERT'.
            swin_checkpoint_path (str): The path to the Swin model safetensor checkpoint.
                If None, downloads from HuggingFace.
            bert_local_dir (str): The path to the local ClinicalBERT model directory.
                If None, downloads from HuggingFace.
            pretrained (bool): Whether to load the pre-trained weights.
                Defaults to True.
            img_dim (int): The dimension of the image embedding.
                Defaults to None.
            txt_dim (int): The dimension of the text embedding.
                Defaults to None.
        """
        super().__init__()
        self.img_backbone = img_backbone
        self.swin_channel = 3

        if img_backbone == "swin":
            self.vision = timm.create_model(swin_model_name, pretrained=False, in_chans=self.swin_channel)
            if pretrained and swin_checkpoint_path:
                try:
                    state = load_safetensor(str(swin_checkpoint_path), device="cpu")
                    # collapse patch-embed weights
                    if "patch_embed.proj.weight" in state:
                        w3 = state["patch_embed.proj.weight"]
                        w1 = w3.mean(dim=1, keepdim=True)
                        state["patch_embed.proj.weight"] = w1
                    filtered = {k: v for k, v in state.items() if k in self.vision.state_dict()}
                    self.vision.load_state_dict(filtered, strict=False)
                except Exception as e:
                    print("[WARN] Failed to load Swin checkpoint:", e)
                    print("[INFO] Attempting to download pretrained Swin weights...")
                    download_swin(swin_name=swin_model_name, swin_ckpt_path=swin_checkpoint_path, swin_channels=self.swin_channel)
                    state = load_safetensor(str(swin_checkpoint_path), device="cpu")
                    filtered = {k: v for k, v in state.items() if k in self.vision.state_dict()}
                    self.vision.load_state_dict(filtered, strict=False)
            elif pretrained:
                self.vision = timm.create_model(swin_model_name, pretrained=True, in_chans=self.swin_channel)
            self.img_dim = self.vision.num_features if img_dim is None else img_dim

        elif img_backbone == "cnn":
            from torchvision import models
            if cnn_model_name == "resnet50":
                base = models.resnet50(pretrained=pretrained)
                self.vision = nn.Sequential(*list(base.children())[:-1])
                self.img_dim = base.fc.in_features
            elif cnn_model_name == "efficientnet_b0":
                base = models.efficientnet_b0(pretrained=pretrained)
                self.vision = nn.Sequential(*list(base.children())[:-1])
                self.img_dim = base.classifier[1].in_features
            else:
                raise ValueError(f"Unknown cnn_model_name {cnn_model_name}")

        else:
            raise ValueError(f"Unknown img_backbone {img_backbone}")
        self.swin = self.vision
        if hasattr(self.swin, "norm") and isinstance(getattr(self.swin, "norm"), nn.Module):
            self.swin_norm = self.swin.norm
        else:
            # self.img_dim should have been set already
            print("[Backbones] [WARN] Swin norm not found. Using LayerNorm.")
            self.swin_norm = nn.LayerNorm(self.img_dim)
        
        # ---- Text backbone ----
        self.bert = load_hf_model_or_local(bert_model_name, local_dir=bert_local_dir)
        self.txt_dim = self.bert.config.hidden_size if txt_dim is None else txt_dim

    def swin_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract patch-level Swin features and return (B, H, W, C).
        Robust to timm variants that may return different axis orders or a list of stage outputs.
        Prefers a stage whose channel dim matches self.vision.num_features when possible.
        """
        model = self.swin

        # get raw feats (may be tensor or list/tuple)
        if hasattr(model, "forward_features"):
            feats = model.forward_features(x)
        else:
            feats = model(x)

        # If list/tuple of stage outputs, try to pick stage with channel == vision.num_features
        if isinstance(feats, (list, tuple)):
            chosen = None
            for stage in reversed(feats):  # try deeper stages first
                if isinstance(stage, torch.Tensor) and stage.dim() == 4:
                    # try common layout (B, C, H, W) or other orders - check for a channel match
                    if stage.shape[1] == getattr(self.vision, "num_features", None):
                        chosen = stage
                        break
                    # also check other positions
                    if stage.shape[2] == getattr(self.vision, "num_features", None) or stage.shape[3] == getattr(self.vision, "num_features", None):
                        chosen = stage
                        break
            if chosen is None:
                chosen = feats[-1]
            feats = chosen

        # Now feats is a single tensor
        if not isinstance(feats, torch.Tensor):
            raise ValueError(f"Unexpected feats type: {type(feats)}")

        if feats.dim() != 4:
            # handle flattened (B, N, C)
            if feats.dim() == 3:
                B, N, C = feats.shape
                G = int(N ** 0.5)
                if G * G == N:
                    return feats.view(B, G, G, C).contiguous()
                pooled = feats.mean(dim=1).view(B, 1, 1, C)
                return pooled.contiguous()
            raise ValueError(f"Unexpected Swin feature shape: {tuple(feats.shape)}")

        # feats is 4D but channels might be on any axis: find the axis equal to expected channels
        B, d1, d2, d3 = feats.shape
        expected_C = getattr(self.vision, "num_features", None)

        # find the axis (1,2 or 3) that matches expected_C
        ch_axis = None
        for idx, s in enumerate((d1, d2, d3), start=1):
            if expected_C is not None and s == expected_C:
                ch_axis = idx
                break

        if ch_axis is None:
            # fallback: assume (B, C, H, W) as most timm models do and convert to (B, H, W, C)
            return feats.permute(0, 2, 3, 1).contiguous()

        # build permutation that moves batch first, then the two spatial dims (in original order), then the channel axis last.
        perm = [0] + [i for i in (1, 2, 3) if i != ch_axis] + [ch_axis]
        feats = feats.permute(*perm).contiguous()  # now (B, H, W, C)
        return feats

    def forward(self, image, input_ids=None, attention_mask=None):
        img_global, img_patches = None, None

        if image is not None:
            if self.img_backbone == "swin":
                # get patch-level features (B,H,W,C) without running the vision model twice
                patch_feats = self.swin_features(image)            # (B, H, W, C)
                B, H, W, C = patch_feats.shape
                patch_feats = patch_feats.view(B, H * W, C)       # (B, N_patches, C)
                img_patches = self.swin_norm(patch_feats)         # (B, N_patches, C)
                img_global = patch_feats.mean(dim=1)              # (B, C)

            elif self.img_backbone == "cnn":
                feats = self.vision(image)
                if isinstance(feats, (list, tuple)):
                    feats = feats[-1]

                # For torchvision CNNs, after chopping off fc, we usually get (B, C, H, W)
                if feats.dim() == 4:
                    B, C, H, W = feats.shape
                    img_global = feats.mean(dim=[2, 3])                   # (B, C)
                    img_patches = feats.flatten(2).transpose(1, 2)        # (B, H*W, C)
                elif feats.dim() == 2:
                    # Already pooled to (B, D)
                    img_global = feats
                    img_patches = feats.unsqueeze(1)                      # (B, 1, D)
                else:
                    raise ValueError(f"Unexpected CNN output shape: {feats.shape}")

        txt_feats = None
        if input_ids is not None:
            max_len = getattr(self.bert.config, "max_position_embeddings", 512)
            if input_ids.size(1) > max_len:
                print(f"[WARN] Truncating seq_len {input_ids.size(1)} -> {max_len}")
                input_ids = input_ids[:, :max_len]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, :max_len]

            txt_feats = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).last_hidden_state

        return (img_global, img_patches), txt_feats
    
    def extract_global(self, image: torch.Tensor) -> torch.Tensor:
        """Return only the global embedding (B, D)."""
        (img_global, _), _ = self.forward(image)
        return img_global

class CrossModalFusion(nn.Module):
    """
    Cross-modal fusion module.
    """
    def __init__(self, img_dim, txt_dim, joint_dim=256, num_heads=4, use_cls_only=False):
        """
        Constructor for CrossModalFusion.

        Args:
            img_dim (int): Dimensionality of the image features.
            txt_dim (int): Dimensionality of the text features.
            joint_dim (int, optional): Dimensionality of the joint embedding. Defaults to 256.
            num_heads (int, optional): Number of attention heads. Defaults to 4.
            use_cls_only (bool, optional): Whether to use only the CLS token. Defaults to False.
        """
        super().__init__()
        # —— Self-attention for text and image features ——
        self.txt_self_attn = PreFusionEnhancer(txt_dim, num_heads)
        self.img_patch_self_attn = PreFusionEnhancer(img_dim, num_heads)
        self.img_global_self_attn = PreFusionEnhancer(img_dim, num_heads)

        # —— Text features attend to image patches ——
        self.ln_img = nn.LayerNorm(joint_dim)
        self.ln_txt = nn.LayerNorm(joint_dim)

        # —— Text queries attend to image patches ——
        self.query_txt    = nn.Linear(txt_dim, joint_dim)
        self.key_img      = nn.Linear(img_dim, joint_dim)
        self.value_img    = nn.Linear(img_dim, joint_dim)
        self.attn_txt2img = nn.MultiheadAttention(joint_dim, num_heads, batch_first=True)

        # —— Image patches attend to text ——
        self.query_img     = nn.Linear(img_dim, joint_dim)
        self.key_txt       = nn.Linear(txt_dim, joint_dim)
        self.value_txt     = nn.Linear(txt_dim, joint_dim)
        self.attn_img2txt  = nn.MultiheadAttention(joint_dim, num_heads, batch_first=True)

        # Making sure text and image features have the same dimension
        self.txt_proj = nn.Linear(txt_dim, joint_dim)
        self.img_patch_proj = nn.Linear(img_dim, joint_dim)
        self.img_global_proj = nn.Linear(img_dim, joint_dim)

        # a learnable default pooled text token used when no text is provided
        self.default_txt_token = nn.Parameter(torch.zeros(1, 1, txt_dim))
        nn.init.trunc_normal_(self.default_txt_token, std=0.02)

        self.use_cls_only = use_cls_only
        self.comb_mlp = nn.Sequential(
            nn.Linear(joint_dim * 3, joint_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(joint_dim, joint_dim)
        )

        self.embed_dim = joint_dim

    def forward(self, img_global, img_patch, txt_feats, return_attention=False):
        """
        Compute the cross-modal fusion of the input image and text features.

        Parameters:
            img_global (torch.Tensor): Global image features of shape (B, joint_dim).
            img_patch (torch.Tensor): Image patches of shape (B, N, img_dim).
            txt_feats (torch.Tensor): Text features of shape (B, txt_dim).
            return_attention (bool): Whether to return the attention weights.

        Returns:
            torch.Tensor: The fused features of shape (B, joint_dim).
            dict: A dictionary containing the attention weights if return_attention is True.
        """
        if txt_feats is None:
            # repeat the learnable default token for the batch
            B = img_global.shape[0] if img_global is not None else img_patch.shape[0]
            txt_feats = self.default_txt_token.expand(B, -1, -1).to(img_patch.device if img_patch is not None else (img_global.device if img_global is not None else 'cpu'))
        
        # img_global: (B, joint_dim)
        # img_patch:  (B, N_patches, img_dim)
        # txt_feats:  (B, txt_dim)
        txt_feats = self.txt_self_attn(txt_feats)
        img_global = self.img_global_self_attn(img_global.unsqueeze(1)).squeeze(1) # (B,D)
        img_patch = self.img_patch_self_attn(img_patch)           # (B, N, D)
        
        # Handle CLS token usage
        if self.use_cls_only:
            txt_feats_pooled = txt_feats[:, 0].unsqueeze(1)  # (B, 1, D)
        else:
            txt_feats_pooled = txt_feats                     # (B, L, D)
        B, Np, D = img_patch.shape

        # Text attends to image patches
        Q_txt = self.query_txt(txt_feats_pooled)                   # (B, L or 1, D)
        K_img = self.key_img(img_patch)                            # (B, N, D)
        V_img = self.value_img(img_patch)                          # (B, N, D)
        att_txt2img, attn_weights_txt2img = self.attn_txt2img(Q_txt, K_img, V_img)

        # Image patches attend to text
        Q_img = self.query_img(img_patch)                   # (B, N, D)
        K_txt = self.key_txt(txt_feats_pooled)              # (B, L or 1, D)
        V_txt = self.value_txt(txt_feats_pooled)            # (B, L or 1, D)
        att_img2txt, attn_weights_img2txt = self.attn_img2txt(Q_img, K_txt, V_txt)

        # Fuse image patches with text
        img_patch_proj = self.img_patch_proj(img_patch)     # (B, N, joint_dim)
        patches_fused = img_patch_proj + att_img2txt        # (B, N, joint_dim)

        # Fuse global image features with text
        img_global_proj = self.img_global_proj(img_global)                              # (B, joint_dim)
        att_txt2img_pooled = att_txt2img.mean(dim=1)
        img_global_updated = img_global_proj + att_txt2img_pooled
        x1 = self.ln_img(img_global_updated)                    
        
        # Pool text features
        txt_p = self.txt_proj(txt_feats)
        txt_cls = txt_p[:, 0]
        att_img2txt_pooled = att_img2txt.mean(dim=1)
        x2 = self.ln_txt(txt_cls + att_img2txt_pooled)

        patch_toks = patches_fused                                       # (B, N, joint_dim)
        cls_tok = x1.unsqueeze(1)                                        # (B, 1, joint_dim)
        txt_tok = x2.unsqueeze(1)                                        # (B, 1, joint_dim)

        attn_dict = {'txt2img': attn_weights_txt2img, 'img2txt': attn_weights_img2txt}
        
        if self.use_cls_only:
            # compute fused vector via MLP combiner
            patch_avg = patches_fused.mean(dim=1)  # (B, D)
            cat = torch.cat([x1, patch_avg, x2], dim=1)  # (B, 3*D)
            fused_vec = self.comb_mlp(cat)  # (B, D)

            attn_dict['patch_avg'] = patch_avg.detach() 
            if return_attention:
                return fused_vec, attn_dict
            return fused_vec, None

        seq = torch.cat([cls_tok, patch_toks, txt_tok], dim=1)  # (B, 1+Np+1, joint_dim)
        if return_attention:
            return seq, attn_dict
        return seq, None