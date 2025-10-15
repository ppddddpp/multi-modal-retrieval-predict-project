import torch
import torch.nn as nn
from pathlib import Path

from Helpers import Config
from .fusion import Backbones
from LabelData import disease_groups, normal_groups, finding_groups, symptom_groups

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "models"
CONFIG_DIR = BASE_DIR / "configs"

class GlobalAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )
    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class MiniSwinBlock(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=7):
        super().__init__()
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )
    def forward(self, x):
        B, L, D = x.shape
        ws = self.window_size
        num_windows = L // ws
        x = x[:, :num_windows * ws, :].reshape(B * num_windows, ws, D)
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        x = x.reshape(B, num_windows * ws, D)
        return x


class SwinModelForFinetune(nn.Module):
    """
    Label-aware classifier:
        - Uses backbones.swin_features(images) -> patch tokens or feature map
        - Projects to embed_dim if needed
        - Uses learnable label queries + MultiheadAttention (batch_first=True)
        - Produces per-label logits
    """
    def __init__(self, backbones, num_classes: int, num_heads: int = None, dropout: float = 0.2, fusion_hidden: int = None):
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

        # --- Image enhancement block ---
        self.image_enhancer = nn.Sequential(
            GlobalAttentionBlock(self.embed_dim, num_heads=num_heads),
            MiniSwinBlock(self.embed_dim, num_heads=num_heads, window_size=7)
        )

        # Label-aware attention
        self.label_queries = nn.Parameter(torch.randn(self.num_labels, self.embed_dim) * 0.02)
        self.mha = nn.MultiheadAttention(self.embed_dim, num_heads, batch_first=True, dropout=dropout)

        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.embed_dim * 2, fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, self.embed_dim)
        )

        # Classifier
        self.label_to_logit = nn.Linear(self.embed_dim, 1)

    def forward(self, images, return_attn_weights=False):
        (img_global, img_patches), _ = self.backbones(images)
        B = img_patches.size(0)

        # --- Image enhancement ---
        seq = torch.cat([img_global.unsqueeze(1), img_patches], dim=1)
        seq = self.image_enhancer(seq)
        img_global = seq[:, 0, :]
        img_patches = seq[:, 1:, :]

        # --- Label-aware attention ---
        queries = self.label_queries.unsqueeze(0).expand(B, -1, -1)
        attn_out, attn_weights = self.mha(queries, img_patches, img_patches, need_weights=True)

        # --- Fuse with global token ---
        gp_exp = img_global.unsqueeze(1).expand(-1, self.num_labels, -1)
        fused = torch.cat([attn_out, gp_exp], dim=-1)
        label_embs = self.fusion_mlp(fused)
        logits = self.label_to_logit(label_embs).squeeze(-1)

        if return_attn_weights:
            return logits, label_embs, attn_weights
        return logits, label_embs, None