import os
import torch
import torch.nn as nn
import timm
from model_utils import load_hf_model_or_local
from safetensors.torch import load_file as load_safetensor
from pathlib import Path

try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd().parent

MODEL_PLACE = BASE_DIR / 'models'
os.environ['TRANSFORMERS_CACHE'] = str(MODEL_PLACE)

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
        x2, _ = self.self_attn(x, x, x)
        x = self.norm1(self.alpha * x + self.dropout(x2))
        return x

class Backbones(nn.Module):
    """
    Image and text backbones: Swin Transformer and ClinicalBERT.
    Outputs global image features and pooled CLS text embeddings.
    """
    def __init__(
            self, swin_model_name='swin_base_patch4_window7_224', bert_model_name='emilyalsentzer/Bio_ClinicalBERT',
            swin_checkpoint_path=None,
            bert_local_dir=None,
            pretrained=True,
            joint_dim=1024
            ):
        """
        Constructor for Backbones.

        Args:
            swin_model_name (str): The Swin Transformer model name.
                Defaults to 'swin_base_patch4_window7_224'.
            bert_model_name (str): The ClinicalBERT model name.
                Defaults to 'emilyalsentzer/Bio_ClinicalBERT'.
            swin_checkpoint_path (str): The path to the Swin model safetensor checkpoint.
                If None, downloads from HuggingFace.
            bert_local_dir (str): The path to the local ClinicalBERT model directory.
                If None, downloads from HuggingFace.
            pretrained (bool): Whether to load the pre-trained weights.
                Defaults to True.
            joint_dim (int): The dimension of the joint embedding. 
            Defaults to 1024.
        """
        super().__init__()
        # instantiate with random weights
        self.swin = timm.create_model(swin_model_name, pretrained=False, in_chans=1)
        if pretrained and swin_checkpoint_path:
            # load the raw safetensors into a state dict
            try:
                state = load_safetensor(str(swin_checkpoint_path), device='cpu')
            except ImportError:
                # fallback to torch.load for .pth
                state = torch.load(str(swin_checkpoint_path), map_location='cpu')

            # collapse the 3‑channel patch‑embed weights to 1 channel by averaging
            if 'patch_embed.proj.weight' in state:
                w3 = state['patch_embed.proj.weight']  # shape [128, 3, 4, 4]
                w1 = w3.mean(dim=1, keepdim=True)      # shape [128, 1, 4, 4]
                state['patch_embed.proj.weight'] = w1
            
            # filter out the relative_position_index & attn_mask keys
            filtered = {
                k: v for k, v in state.items()
                if k in self.swin.state_dict()  # only keep matching names
            }

            # load with strict=False to ignore missing / unexpected
            missing, unexpected = self.swin.load_state_dict(filtered, strict=False)
            print(f"[Info] [fusion.py] Swin loaded from {swin_checkpoint_path}, missing keys: {missing}, unexpected keys: {unexpected}")
            
        elif pretrained:
            # if no local checkpoint, let timm download / cache normally
            self.swin = timm.create_model(
                swin_model_name,
                pretrained=True,
                in_chans=1,
                checkpoint_path=str(MODEL_PLACE)
            )

        self.ln_txt2img = nn.LayerNorm(joint_dim)
        self.ln_img2txt = nn.LayerNorm(joint_dim)
        self.swin_features = nn.Sequential(*list(self.swin.children())[:-1])
        self.img_dim       = self.swin.num_features
        # load ClinicalBERT & record txt_dim
        self.bert   = load_hf_model_or_local(bert_model_name, local_dir=bert_local_dir)
        self.txt_dim = self.bert.config.hidden_size

    def forward(self, image, input_ids, attention_mask):
        # image
        patch_feats = self.swin_features(image)            # (B, H, W, C)
        B, H, W, C = patch_feats.shape
        patch_feats = patch_feats.reshape(B, H * W, C)     # (B, N_patches, C)
        patch_feats = self.swin.norm(patch_feats)          # (B, N_patches, C)
        global_feats = patch_feats.mean(dim=1)             # (B, C)

        # text
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        txt_feats = outputs.last_hidden_state       

        return (global_feats, patch_feats), txt_feats

class CrossModalFusion(nn.Module):
    """
    Cross-modal fusion module.
    """
    def __init__(self, img_dim, txt_dim, joint_dim=256, num_heads=4):
        """
        Constructor for CrossModalFusion.

        Args:
            img_dim (int): Dimensionality of the image features.
            txt_dim (int): Dimensionality of the text features.
            joint_dim (int, optional): Dimensionality of the joint embedding. Defaults to 256.
            num_heads (int, optional): Number of attention heads. Defaults to 4.
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

        # final projection after concatenating attended img + text
        self.output = nn.Sequential(
            nn.Linear(joint_dim * 2, joint_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.adapter = nn.Sequential(
            nn.Linear(joint_dim, joint_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(joint_dim // 2, joint_dim)
        )

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
        # img_global: (B, joint_dim)
        # img_patch:  (B, N_patches, img_dim)
        # txt_feats:  (B, txt_dim)
        txt_feats = self.txt_self_attn(txt_feats)
        txt_feats_pooled = txt_feats[:, 0]
        img_global = self.img_global_self_attn(img_global.unsqueeze(1)).squeeze(1)
        img_patch = self.img_patch_self_attn(img_patch)

        B, N, D = img_patch.shape

        # Text attends to image patches
        Q_txt = self.query_txt(txt_feats_pooled).unsqueeze(1)      # (B, 1, joint_dim)
        K_img = self.key_img(img_patch)                     # (B, N, joint_dim)
        V_img = self.value_img(img_patch)                   # (B, N, joint_dim)
        att_txt2img, attn_weights_txt2img = self.attn_txt2img(Q_txt, K_img, V_img)
        att_txt2img = att_txt2img.squeeze(1)                # (B, joint_dim)

        # Image patches attend to text
        Q_img = self.query_img(img_patch)                   # (B, N, joint_dim)
        K_txt = self.key_txt(txt_feats_pooled).unsqueeze(1)        # (B, 1, joint_dim)
        V_txt = self.value_txt(txt_feats_pooled).unsqueeze(1)      # (B, 1, joint_dim)
        att_img2txt, attn_weights_img2txt = self.attn_img2txt(Q_img, K_txt, V_txt)

        img_patch_proj = self.img_patch_proj(img_patch)     # (B, N, joint_dim)
        patches_fused = img_patch_proj + att_img2txt      

        # Pool fused patches to get updated global image vector
        img_global_proj = self.img_global_proj(img_global)                              # (B, joint_dim)
        img_global_updated = patches_fused.mean(dim=1) + img_global_proj                    
        txt_p = self.txt_proj(txt_feats)

        # Concatenate and final projection
        att_img2txt_pooled = att_img2txt.mean(dim=1)
        x1 = self.ln_img(img_global_updated + att_txt2img)
        txt_cls = txt_p[:, 0]
        x2 = self.ln_txt(txt_cls + att_img2txt_pooled)
        x = torch.cat([x1, x2], dim=1)                                                  # (B, joint_dim + txt_dim)
        fused = self.output(x)
        fused = fused + self.adapter(fused)                                             # (B, joint_dim)

        if return_attention:
            return fused, {'txt2img': attn_weights_txt2img, 'img2txt': attn_weights_img2txt}
        return fused, None


