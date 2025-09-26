import torch
import torch.nn as nn
from .fusion import CrossModalFusion, Backbones
from Retrieval import  RetrievalEngine, make_retrieval_engine
from pathlib import Path
from .explain import ExplanationEngine
import numpy as np
import json

BASE_DIR = Path(__file__).resolve().parent.parent.parent
EMBEDDINGS_DIR = BASE_DIR / "embeddings"

class MultiHeadMLP(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.linear1 = nn.Linear(dim, dim * 2)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(dim * 2, dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x

class StochasticDepth(nn.Module):
    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x, residual):
        if not self.training or self.drop_prob == 0.:
            return x + residual
        keep_prob = 1 - self.drop_prob
        shape = [x.shape[0]] + [1] * (x.ndim - 1)  # (B, 1, 1...)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_mask = torch.floor(random_tensor)
        return x + residual * binary_mask / keep_prob

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=512):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.pe = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.normal_(self.pe, std=0.02)

    def forward(self, x):
        L = x.size(1)
        if L > self.max_len:
            # extend PE on the fly
            extra = torch.zeros(1, L - self.max_len, self.dim, device=x.device)
            nn.init.normal_(extra, std=0.02)
            pe = torch.cat([self.pe, extra], dim=1)
            return x + pe[:, :L]
        return x + self.pe[:, :L]

class MultiModalRetrievalModel(nn.Module):
    """
    Wraps Backbones + Fusion + Classification head
    into one model that produces:
      - joint_emb (B, joint_dim)
      - logits    (B, num_classes)
    """
    def __init__(
        self,
        joint_dim:    int = 256,
        num_heads:    int = 4,
        num_classes:  int = 22,
        num_fusion_layers: int = 3,
        fusion_type:  str = "cross",
        img_backbone: str = "swin",
        swin_name:    str = "swin_base_patch4_window7_224",
        cnn_name: str = "resnet50",
        bert_name:    str = "emilyalsentzer/Bio_ClinicalBERT",
        swin_ckpt_path:    str = None,
        bert_local_dir:   str = None,
        pretrained:   bool = True,
        checkpoint_path:str = None,
        device:         torch.device = torch.device("cpu"),
        training:      bool = False,
        use_shared_ffn: bool = True,
        use_cls_only: bool = False,
        model_type: str = "multimodal",
        retriever: RetrievalEngine = None
    ):
        """
        :param joint_dim: dimensionality of the joint embedding
        :param num_heads: number of attention heads for CrossModalFusion
        :param num_classes: number of output classes
        :param num_fusion_layers: number of fusion layers to use
        :param use_shared_ffn: whether to use a shared FFN across fusion layers
        :param fusion_type: type of fusion module to use; one of "cross", "simple", "gated"
        :param img_backbone: type of image backbone to use; one of "swin", "cnn"
        :param swin_name: name of the Swin transformer model to use
        :param cnn_name: name of the CNN model to use
        :param bert_name: name of the ClinicalBERT model to use
        :param swin_ckpt_path: path to a Swin transformer checkpoint to load
        :param bert_local_dir: directory containing a ClinicalBERT model to load
        :param pretrained: whether to load pre-trained weights for the Swin and ClinicalBERT models
        :param checkpoint_path: path to a model checkpoint to load
        :param device: device to run the model on
        :param training: whether the model is being trained or used for inference
        :param use_cls_only: whether to use only the CLS token for BERT
        :param model_type: type of model to use; one of "multimodal", "image", "text"
        :param retriever: optional retrieval engine for case-based retrieval

        Args:
            joint_dim (int, optional): dimensionality of the joint embedding. Defaults to 256.
            num_heads (int, optional): number of attention heads for CrossModalFusion. Defaults to 4.
            num_classes (int, optional): number of output classes. Defaults to 22.
            use_shared_ffn (bool, optional): whether to use a shared FFN across fusion layers. Defaults to True.
            num_fusion_layers (int, optional): number of fusion layers to use. Defaults to 3.
            fusion_type (str, optional): type of fusion module to use; one of "cross", "simple", "gated". Defaults to "cross".
            img_backbone (str, optional): type of image backbone to use; one of "swin", "cnn". Defaults to "swin".
            swin_name (str, optional): name of the Swin transformer model to use. Defaults to "swin_base_patch4_window7_224".
            cnn_name (str, optional): name of the CNN model to use. Defaults to "resnet50".
            bert_name (str, optional): name of the ClinicalBERT model to use. Defaults to "emilyalsentzer/Bio_ClinicalBERT".
            swin_ckpt_path (str, optional): path to a Swin transformer checkpoint to load. Defaults to None.
            bert_local_dir (str, optional): directory containing a ClinicalBERT model to load. Defaults to None.
            pretrained (bool, optional): whether to load pre-trained weights for the Swin and ClinicalBERT models. Defaults to True.
            checkpoint_path (str, optional): path to a model checkpoint to load. Defaults to None.
            device (torch.device, optional): device to run the model on. Defaults to torch.device("cpu").
            training (bool, optional): whether the model is being trained or used for inference. Defaults to False.
            use_cls_only (bool, optional): whether to use only the CLS token for BERT. Defaults to False.
            model_type (str, optional): type of model to use; one of "multimodal", "image", "text". Defaults to "multimodal".
            retriever (RetrievalEngine, optional): optional retrieval engine for case-based retrieval. Defaults to None.
        """
        super().__init__()
        self.device = device
        self.use_shared_ffn = use_shared_ffn
        # instantiate vision+text backbones
        self.backbones = Backbones(
            img_backbone       = img_backbone,
            swin_model_name    = swin_name,
            cnn_model_name     = cnn_name,
            bert_model_name    = bert_name,
            swin_checkpoint_path = swin_ckpt_path,
            bert_local_dir       = bert_local_dir,
            pretrained           = pretrained
        ).to(device)
        img_dim = self.backbones.img_dim
        txt_dim = self.backbones.txt_dim
        
        # set up model type
        if model_type in ["multimodal", "image", "text"]:
            self.model_type = model_type
        else:
            raise ValueError(f"Unknown model_type {model_type!r}")

        # set up fusion
        if fusion_type == "cross":
            self.fusion_layers = nn.ModuleList([
                CrossModalFusion(img_dim, txt_dim, joint_dim, num_heads, use_cls_only).to(device)
                for _ in range(num_fusion_layers)
            ])
        else:
            raise ValueError(f"Unknown fusion_type {fusion_type!r}")
        
        self.self_attn = nn.MultiheadAttention(embed_dim=joint_dim, num_heads=num_heads, batch_first=True)

        # LayerNorm and FFN for residual connections and normalization
        self.norm1_layers = nn.ModuleList([
            nn.LayerNorm(joint_dim, eps=1e-5) for _ in range(num_fusion_layers)
        ])
        self.norm2_layers = nn.ModuleList([
            nn.LayerNorm(joint_dim, eps=1e-5) for _ in range(num_fusion_layers)
        ])

        self.alpha = nn.Parameter(torch.ones(1)) # learnable scaling factor for residual connections
        self.dropout = nn.Dropout(0.1)
        self.pos_encoder = PositionalEncoding(joint_dim, txt_dim)

        # Feed-forward network for each fusion layer
        if self.use_shared_ffn:
            self.shared_ffn = MultiHeadMLP(joint_dim, num_heads)
            self.ffn = None
        else:
            self.ffn = nn.ModuleList([
                MultiHeadMLP(joint_dim, num_heads)
                for _ in range(num_fusion_layers)
            ])
            self.shared_ffn = None

        # Stochastic depth for residual connections
        self.drop_path_layers = nn.ModuleList([
            StochasticDepth(0.1) for _ in range(num_fusion_layers)
        ])

        # Projection for image and text embeddings only case
        self.img_proj = nn.Linear(img_dim, joint_dim).to(device)
        self.txt_proj = nn.Linear(txt_dim, joint_dim).to(device)

        # adapters for each fusion layer
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(joint_dim, joint_dim // 2),
                nn.GELU(),
                nn.Linear(joint_dim // 2, joint_dim)
            ) for _ in range(num_fusion_layers)
        ])

        # classification head on the joint embedding
        self.classifier = nn.Sequential(
            nn.Linear(joint_dim, joint_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(joint_dim * 4, num_classes),
            nn.Dropout(0.1)
        ).to(device)

        self.use_cls_only = use_cls_only

        #  Retrieval engine (offline index of embeddings)
        if not training:
            if checkpoint_path:
                state = torch.load(checkpoint_path, map_location=device)
                self.load_state_dict(state)
                self.to(device)
                self.eval()
            else:
                raise ValueError("checkpoint_path must be provided for inference")
            
            if retriever:
                self.retriever = retriever
            else:
                features_path = EMBEDDINGS_DIR / "val_joint_embeddings.npy"
                ids_path      = EMBEDDINGS_DIR / "val_ids.json"
                if not features_path.exists() or not ids_path.exists():
                    raise FileNotFoundError(
                        f"Expected embeddings at {features_path} and IDs at {ids_path}"
                    )
                import warnings
                warnings.warn("No retrieval engine provided. Using default DLS with val embeddings.")

                self.retriever = make_retrieval_engine(
                    features_path=str(features_path),
                    ids_path=str(ids_path),
                    method="dls",
                    link_threshold=0.5,
                    max_links=10
                )

            # set up explanation 
            self.explainer   = None
            self.ig_steps    = 100
            self.image_size  = (224,224)

        else:
            # during training, we can use a dummy path
            features_path = EMBEDDINGS_DIR / "dummy_embeddings.npy"
            ids_path      = EMBEDDINGS_DIR / "dummy_ids.json"
            if not features_path.exists() or not ids_path.exists():
                np.save(features_path, np.zeros((1, joint_dim), dtype=np.float32))
                with open(ids_path, "w") as f:
                    json.dump(["dummy_id"], f)
            
            self.retriever = None

    def set_retriever(self, retriever: RetrievalEngine):
        self.retriever = retriever

    def forward(self, image, input_ids, attention_mask, return_attention=False):
        """
        Inputs:
        image (torch.Tensor): (B, C, H, W)
        input_ids (torch.Tensor): (B, L)
        attention_mask (torch.Tensor): (B, L)

        Returns:
        dict with keys:
            - "joint_emb": (B, joint_dim)
            - "img_emb":   (B, joint_dim) or None
            - "txt_emb":   (B, joint_dim) or None
            - "logits":    (B, num_classes)
            - "attn":      dict of attention maps (if return_attention=True), else None
        """
        attn_weights = {}
        joint_emb = None

        # --- Backbones ---
        (img_global, img_patches), txt_feats = self.backbones(
            image.to(self.device) if image is not None else None,
            input_ids.to(self.device) if input_ids is not None else None,
            attention_mask.to(self.device) if attention_mask is not None else None
        )

        img_emb = self.img_proj(img_global) if img_global is not None else None
        if txt_feats is not None:
            if self.use_cls_only:
                txt_emb = txt_feats[:, 0, :]
            else:
                txt_emb = txt_feats.mean(dim=1)
            txt_emb = self.txt_proj(txt_emb)
        else:
            txt_emb = None
        
        if self.model_type == "multimodal":
            for i, fusion in enumerate(self.fusion_layers):
                # fusion may return either pooled (B, D) or sequence (B, L, D)
                fused_out, attn_from_fusion = fusion(
                    img_global,
                    img_patches,
                    txt_feats,
                    return_attention=return_attention
                )

                # Handling fuse output 
                if fused_out is None:
                    raise RuntimeError("Fusion returned None fused_out")
                if fused_out.dim() == 3:
                    seq = fused_out  # (B, L, D)
                elif fused_out.dim() == 2:
                    seq = fused_out.unsqueeze(1)  # (B, 1, D)
                else:
                    raise RuntimeError(f"Unexpected fused_out shape: {fused_out.shape}")

                # dropout + positional encoding (seq: B,L,D)
                seq = self.dropout(seq)
                seq = self.pos_encoder(seq)

                seq_out, comb_attn_weights = self.self_attn(seq, seq, seq)

                # store comb & fusion attn weights (detach -> cpu to avoid keeping graph)
                if return_attention:
                    try:
                        attn_weights[f"layer_{i}_comb"] = comb_attn_weights.detach().cpu()
                    except Exception:
                        # comb_attn_weights may be None or not a tensor (guard)
                        attn_weights[f"layer_{i}_comb"] = comb_attn_weights

                    # fusion produced cross-attention dict (txt2img/img2txt) — store if present
                    if attn_from_fusion:
                        if "txt2img" in attn_from_fusion and attn_from_fusion["txt2img"] is not None:
                            try:
                                attn_weights[f"layer_{i}_txt2img"] = attn_from_fusion["txt2img"].detach().cpu()
                            except Exception:
                                attn_weights[f"layer_{i}_txt2img"] = attn_from_fusion["txt2img"]
                        if "img2txt" in attn_from_fusion and attn_from_fusion["img2txt"] is not None:
                            try:
                                attn_weights[f"layer_{i}_img2txt"] = attn_from_fusion["img2txt"].detach().cpu()
                            except Exception:
                                attn_weights[f"layer_{i}_img2txt"] = attn_from_fusion["img2txt"]
                        # optional extras (patch_avg etc.)
                        if "patch_avg" in attn_from_fusion:
                            try:
                                attn_weights[f"layer_{i}_patch_avg"] = attn_from_fusion["patch_avg"].detach().cpu()
                            except Exception:
                                attn_weights[f"layer_{i}_patch_avg"] = attn_from_fusion["patch_avg"]

                if self.use_cls_only:
                    fused = fused_out[:, 0, :]
                else:
                    fused = seq_out.mean(dim=1)

                # Residual
                if i == 0:
                    x = fused
                else:
                    x = self.norm1_layers[i](joint_emb)
                    x = self.drop_path_layers[i](x, self.alpha * fused)

                # FFN + Adapter
                x_ffn = self.norm2_layers[i](x)
                if self.use_shared_ffn:
                    x = x + self.shared_ffn(x_ffn)
                else:
                    x = x + self.ffn[i](x_ffn)
                x = x + self.adapters[i](x)

                # Update joint_emb
                joint_emb = x

            # After loop, ensure joint_emb is (B, D)
            if joint_emb is None:
                raise RuntimeError("joint_emb is still None after forward")
            if joint_emb.dim() == 3:
                # if singleton seq dim (B,1,D) -> squeeze; otherwise mean-pool tokens
                if joint_emb.size(1) == 1:
                    joint_emb = joint_emb.squeeze(1)
                else:
                    joint_emb = joint_emb.mean(dim=1)

        # --- Image-only ---
        elif self.model_type == "image":
            x = img_global  # (B, D_img)
            x = self.img_proj(x)  # nn.Linear(img_dim, joint_dim)
            x = self.shared_ffn(x) if self.use_shared_ffn else self.ffn[0](x)
            joint_emb = x

        # --- Text-only ---
        elif self.model_type == "text":
            if self.use_cls_only:
                x = txt_feats[:, 0, :]  # CLS
            else:
                x = txt_feats.mean(dim=1)  # mean pool
            x = self.txt_proj(x)  # nn.Linear(txt_dim, joint_dim)
            x = self.shared_ffn(x) if self.use_shared_ffn else self.ffn[0](x)
            joint_emb = x
        
        logits = self.classifier(joint_emb)

        return {
            "joint_emb": joint_emb,
            "img_emb": img_emb,
            "txt_emb": txt_emb,
            "logits": logits,
            "attn": (attn_weights if return_attention else None)
        }

    def predict(self,
                image: torch.Tensor,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                threshold=0.5,
                K: int = 5,
                explain: bool = False
                ) -> dict:
        
        joint_emb, logits, attn_weights = self.forward(
            image, input_ids, attention_mask, return_attention=explain
        )

        # probabilities 
        probs = torch.sigmoid(logits)               # (B, num_classes)
        preds = (probs >= threshold).int()
        topk_vals, topk_idx = probs.topk(K, dim=-1)     # (B, K)

        q_emb = joint_emb.detach().cpu().numpy()          # (B, D)

        # ---- Assemble output ----
        output = {
            'probs':            probs.detach().cpu().numpy(),
            'preds':            preds.detach().cpu().numpy(),
            'topk_idx':         topk_idx.detach().cpu().tolist(),
            'topk_vals':        topk_vals.detach().cpu().tolist(),
        }

        if explain:
            targets = topk_idx[0].tolist()

            if self.model_type == "multimodal":
                expl = self.explain(
                    image, input_ids, attention_mask,
                    joint_emb, attn_weights,
                    targets=targets,
                    K=K
                )
            elif self.model_type == "image":
                # no attention/text branch
                (img_global, img_patches), _ = self.backbones(
                    image.to(self.device),
                    None, None
                )
                expl = self.explain_image_only(img_global, img_patches, targets)
                # add retrieval to keep schema consistent
                retr_ids, retr_dists = self.retriever.retrieve(q_emb, K=K)
                expl.update({
                    "retrieval_ids": retr_ids,
                    "retrieval_dists": retr_dists,
                    "attention_map": None,   # not applicable
                })
            elif self.model_type == "text":
                _, txt_feats = self.backbones(
                    None,
                    input_ids.to(self.device),
                    attention_mask.to(self.device)
                )
                expl = self.explain_text_only(txt_feats, targets)
                retr_ids, retr_dists = self.retriever.retrieve(q_emb, K=K)
                expl.update({
                    "retrieval_ids": retr_ids,
                    "retrieval_dists": retr_dists,
                    "attention_map": None,   # not applicable
                    "gradcam_maps": None,    # not applicable
                })
            else:
                raise ValueError(f"Unsupported model_type: {self.model_type}")

            output.update(expl)

        return output

    def explain(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        joint_emb: torch.Tensor,
        attn_weights: dict,
        targets: list,
        K: int = 5
    ) -> dict:
        # Case-based retrieval again as part of explanation
        q_emb = joint_emb.detach().cpu().numpy()
        retr_ids, retr_dists = self.retriever.retrieve(q_emb, K=K)

        # Lazy-init explainer
        if self.explainer is None:
            self.explainer = ExplanationEngine(
                fusion_model=self.fusion_layers[-1],
                classifier_head=self.classifier,
                image_size=self.image_size,
                ig_steps=self.ig_steps,
                device=self.device
            )

        # Extract features for heatmap methods
        (img_global, img_patches), txt_feats = self.backbones(
            image.to(self.device),
            input_ids.to(self.device),
            attention_mask.to(self.device)
        )

        # Extract attention maps
        last_idx = len(self.fusion_layers) - 1
        attn_weights_expl = {
            "txt2img": attn_weights.get(f"layer_{last_idx}_txt2img"),
            "img2txt": attn_weights.get(f"layer_{last_idx}_img2txt"),
            "comb":    attn_weights.get(f"layer_{last_idx}_comb", None)
        }

        # Get attention maps explained
        maps = self.explainer.explain(
            img_global=img_global,
            img_patches=img_patches,
            txt_feats=txt_feats,
            attn_weights=attn_weights_expl,
            targets=targets
        )

        return {
            'retrieval_ids':   retr_ids,
            'retrieval_dists': retr_dists,
            'attention_map':   maps['attention_map'],
            'ig_maps':         maps['ig_maps'],
            'gradcam_maps':      maps['gradcam_maps']
        }
    
    def get_explain_score(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        targets: list,
    ) -> dict:
        """
        Computes explanation maps for a single input using three methods.

        Args:
            image: torch.Tensor of shape (B, C, H, W)
            input_ids: torch.Tensor of shape (B, T)
            attention_mask: torch.Tensor of shape (B, T)
            targets: single int or list of ints for the class indices to explain

        Returns:
            dict with three keys:
                - 'attention_map': (H, W) attention map over image patches
                - 'ig_maps':       dict of target to (H, W) IG map over image patches
                - 'gradcam_maps':  dict of target to (H, W) Grad-CAM map over image patches
        """
        # Lazy-init explainer
        if self.explainer is None:
            self.explainer = ExplanationEngine(
                fusion_model=self.fusion_layers[-1],
                classifier_head=self.classifier,
                image_size=self.image_size,
                ig_steps=self.ig_steps,
                device=self.device
            )

        _, _, attn_weights = self.forward(
            image, input_ids, attention_mask, return_attention=True
        )

        # Extract features for heatmap methods
        (img_global, img_patches), txt_feats = self.backbones(
            image.to(self.device),
            input_ids.to(self.device),
            attention_mask.to(self.device)
        )

        # Extract attention maps
        last_idx = len(self.fusion_layers) - 1
        attn_weights_expl = {
            "txt2img": attn_weights.get(f"layer_{last_idx}_txt2img"),
            "img2txt": attn_weights.get(f"layer_{last_idx}_img2txt"),
            "comb":    attn_weights.get(f"layer_{last_idx}_comb", None)
        }

        maps = self.explainer.explain(
            img_global=img_global,
            img_patches=img_patches,
            txt_feats=txt_feats,
            attn_weights=attn_weights_expl,
            targets=targets
        )

        return {
            'attention_map':   maps['attention_map'],
            'ig_maps':         maps['ig_maps'],
            'gradcam_maps':      maps['gradcam_maps']
        }
    
    def explain_image_only(self, img_global, img_patches, targets, K=5):
        if self.explainer is None:
            self.explainer = ExplanationEngine(
                fusion_model=None,            # no fusion in image-only
                classifier_head=self.classifier,
                image_size=self.image_size,
                ig_steps=self.ig_steps,
                device=self.device
            )

        # Get explanation maps for image branch only
        maps = self.explainer.explain_image_only(
            img_global=img_global,
            img_patches=img_patches,
            targets=targets
        )

        return {
            "attention_map": None,        # not applicable in single modality
            "ig_maps": maps["ig_maps"],   # from ExplanationEngine
            "gradcam_maps": maps["gradcam_maps"]
        }

    def explain_text_only(self, txt_feats, targets, K=5):
        if self.explainer is None:
            self.explainer = ExplanationEngine(
                fusion_model=None,             # no fusion in text-only
                classifier_head=self.classifier,
                image_size=self.image_size,
                ig_steps=self.ig_steps,
                device=self.device
            )

        # Get explanation maps for text branch only
        maps = self.explainer.explain_text_only(
            txt_feats=txt_feats,
            targets=targets
        )

        return {
            "attention_map": None,        # not applicable
            "ig_maps": maps["ig_maps"],   # token-level attribution
            "gradcam_maps": maps["gradcam_maps"] # token-level Grad-CAM
        }
