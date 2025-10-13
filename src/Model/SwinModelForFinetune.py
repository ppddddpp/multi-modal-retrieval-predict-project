import torch
import torch.nn as nn
from pathlib import Path

from Helpers import Config
from .fusion import Backbones
from LabelData import disease_groups, normal_groups, finding_groups, symptom_groups

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "models"
CONFIG_DIR = BASE_DIR / "configs"

class SwinModelForFinetune(nn.Module):
    """
    Label-aware classifier:
        - Uses backbones.swin_features(images) -> patch tokens or feature map
        - Projects to embed_dim if needed
        - Uses learnable label queries + MultiheadAttention (batch_first=True)
        - Produces per-label logits
    """
    def __init__(
        self,
        backbones,
        num_classes: int,
        num_heads: int = None,
        dropout: float = 0.2,
        fusion_hidden: int = None
    ):
        super().__init__()
        cfg = Config.load(CONFIG_DIR / 'config.yaml')
        num_heads = num_heads or cfg.num_heads
        fusion_hidden = fusion_hidden or cfg.joint_dim

        if backbones is None:
            self.backbones = Backbones(
                img_backbone=cfg.image_backbone,
                swin_checkpoint_path=MODEL_DIR / "swin_checkpoint.safetensors",
                pretrained=True
            )
        else:
            self.backbones = backbones

        combined_groups = {**disease_groups, **normal_groups, **finding_groups, **symptom_groups}
        self.num_labels = num_classes if num_classes > 0 else len(combined_groups)

        self.embed_dim = self.backbones.img_dim

        # Learnable label queries (L, E)
        self.label_queries = nn.Parameter(torch.randn(self.num_labels, self.embed_dim) * 0.02)

        # Attention: queries = labels, keys/values = image patches
        self.mha = nn.MultiheadAttention(self.embed_dim, num_heads, batch_first=True, dropout=dropout)

        # fuse attention output with global vector
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.embed_dim * 2, fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, self.embed_dim)
        )

        # final classifier
        self.label_to_logit = nn.Linear(self.embed_dim, 1)

    def forward(self, images, return_attn_weights=False):
        # get backbone outputs
        (img_global, img_patches), _ = self.backbones(images)  # txt_feats not needed

        B = img_patches.size(0)

        # expand label queries for this batch
        queries = self.label_queries.unsqueeze(0).expand(B, -1, -1)  # (B, L, D)

        # run attention: labels attend to patch tokens
        attn_out, attn_weights = self.mha(queries, img_patches, img_patches, need_weights=True)  # (B, L, D)

        # fuse with global vector
        gp_exp = img_global.unsqueeze(1).expand(-1, self.num_labels, -1)  # (B, L, D)
        fused = torch.cat([attn_out, gp_exp], dim=-1)  # (B, L, 2D)
        label_embs = self.fusion_mlp(fused)            # (B, L, D)

        # map label embeddings -> logits
        logits = self.label_to_logit(label_embs).squeeze(-1)  # (B, L)
        
        if return_attn_weights:
            return logits, label_embs, attn_weights
        return logits, label_embs, None