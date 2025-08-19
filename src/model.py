import torch
import torch.nn as nn
import torch.nn.functional as F
from fusion import CrossModalFusion, Backbones
from retrieval import RetrievalEngine, make_retrieval_engine
from pathlib import Path
from explain import ExplanationEngine
import os
import numpy as np
import json

BASE_DIR = Path(__file__).resolve().parent.parent
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
        B, D = x.size()
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
    def __init__(self, dim, max_len=500):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.normal_(self.pe, std=0.02)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

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
        swin_name:    str = "swin_base_patch4_window7_224",
        bert_name:    str = "emilyalsentzer/Bio_ClinicalBERT",
        swin_ckpt_path:    str = None,
        bert_local_dir:   str = None,
        pretrained:   bool = True,
        checkpoint_path:str = None,
        device:         torch.device = torch.device("cpu"),
        training:      bool = False,
        use_shared_ffn=True,
        retriever: RetrievalEngine = None
    ):
        """
        :param joint_dim: dimensionality of the joint embedding
        :param num_heads: number of attention heads for CrossModalFusion
        :param num_classes: number of output classes
        :param num_fusion_layers: number of fusion layers to use
        :param use_shared_ffn: whether to use a shared FFN across fusion layers
        :param fusion_type: type of fusion module to use; one of "cross", "simple", "gated"
        :param swin_name: name of the Swin transformer model to use
        :param bert_name: name of the ClinicalBERT model to use
        :param swin_ckpt_path: path to a Swin transformer checkpoint to load
        :param bert_local_dir: directory containing a ClinicalBERT model to load
        :param pretrained: whether to load pre-trained weights for the Swin and ClinicalBERT models
        :param checkpoint_path: path to a model checkpoint to load
        :param device: device to run the model on
        :param training: whether the model is being trained or used for inference
        :param retriever: optional retrieval engine for case-based retrieval

        Args:
            joint_dim (int, optional): dimensionality of the joint embedding. Defaults to 256.
            num_heads (int, optional): number of attention heads for CrossModalFusion. Defaults to 4.
            num_classes (int, optional): number of output classes. Defaults to 22.
            use_shared_ffn (bool, optional): whether to use a shared FFN across fusion layers. Defaults to True.
            num_fusion_layers (int, optional): number of fusion layers to use. Defaults to 3.
            fusion_type (str, optional): type of fusion module to use; one of "cross", "simple", "gated". Defaults to "cross".
            swin_name (str, optional): name of the Swin transformer model to use. Defaults to "swin_base_patch4_window7_224".
            bert_name (str, optional): name of the ClinicalBERT model to use. Defaults to "emilyalsentzer/Bio_ClinicalBERT".
            swin_ckpt_path (str, optional): path to a Swin transformer checkpoint to load. Defaults to None.
            bert_local_dir (str, optional): directory containing a ClinicalBERT model to load. Defaults to None.
            pretrained (bool, optional): whether to load pre-trained weights for the Swin and ClinicalBERT models. Defaults to True.
            checkpoint_path (str, optional): path to a model checkpoint to load. Defaults to None.
            device (torch.device, optional): device to run the model on. Defaults to torch.device("cpu").
            training (bool, optional): whether the model is being trained or used for inference. Defaults to False.
            retriever (RetrievalEngine, optional): optional retrieval engine for case-based retrieval. Defaults to None.
        """
        super().__init__()
        self.device = device
        self.use_shared_ffn = use_shared_ffn
        # instantiate vision+text backbones
        self.backbones = Backbones(
            swin_model_name    = swin_name,
            bert_model_name    = bert_name,
            swin_checkpoint_path = swin_ckpt_path,
            bert_local_dir       = bert_local_dir,
            pretrained           = pretrained
        ).to(device)
        img_dim = self.backbones.img_dim
        txt_dim = self.backbones.txt_dim

        # set up fusion
        if fusion_type == "cross":
            self.fusion_layers = nn.ModuleList([
                CrossModalFusion(img_dim, txt_dim, joint_dim, num_heads).to(device)
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
        self.pos_encoder = PositionalEncoding(joint_dim)

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
          pixel_values  (B, 1, H, W) or (B, 3, H, W)
          input_ids     (B, L)
          attention_mask(B, L)
        Returns:
          joint_emb (B, joint_dim)
          logits    (B, num_classes)
        """
        # extracting separate features
        (img_global, img_patches), txt_feats = self.backbones(
            image.to(self.device),
            input_ids.to(self.device),
            attention_mask.to(self.device)
        )

        attn_weights = {}
        joint_emb = None

        for i, fusion in enumerate(self.fusion_layers):
            fused, attn = fusion(
                img_global,
                img_patches,
                txt_feats,
                return_attention=return_attention
            )
            fused = self.dropout(fused)
            fused = fused.unsqueeze(1)
            fused = self.pos_encoder(fused)  # Add positional encoding
            fused, _ = self.self_attn(fused, fused, fused)
            fused, self_attn_weights = self.self_attn(fused, fused, fused)
            if return_attention:
                attn_weights[f"layer_{i}_comb"] = self_attn_weights

            fused = fused.squeeze(1)

            # First layer does not have residual connection
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

            # Collect per-layer attention maps
            if return_attention and attn is not None:
                attn_weights[f"layer_{i}_txt2img"] = attn["txt2img"]
                attn_weights[f"layer_{i}_img2txt"] = attn["img2txt"]
    
        logits = self.classifier(joint_emb)

        return joint_emb, logits, attn_weights if return_attention else None

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
            expl = self.explain(
                image, input_ids, attention_mask,
                joint_emb, attn_weights,
                targets=topk_idx[0].tolist(),
                K=K
            )
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
        print("IG targets:", targets)

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
            image: torch.Tensor of shape (B, 3, H, W)
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