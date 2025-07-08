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

class Backbones(nn.Module):
    """
    Image and text backbones: Swin Transformer and ClinicalBERT.
    Outputs global image features and pooled CLS text embeddings.
    """
    def __init__(
            self, swin_model_name='swin_base_patch4_window7_224', bert_model_name='emilyalsentzer/Bio_ClinicalBERT',
            swin_checkpoint_path=None,
            bert_local_dir=None,
            pretrained=True
            ):
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
            print(f"[fusion] Swin loaded – missing keys: {missing}, unexpected keys: {unexpected}")
            
        elif pretrained:
            # if no local checkpoint, let timm download / cache normally
            self.swin = timm.create_model(
                swin_model_name,
                pretrained=True,
                in_chans=1,
                checkpoint_path=str(MODEL_PLACE)
            )

        self.swin_features = nn.Sequential(*list(self.swin.children())[:-1])
        self.img_dim       = self.swin.num_features
        # load ClinicalBERT & record txt_dim
        self.bert   = load_hf_model_or_local(bert_model_name, local_dir=bert_local_dir)
        self.txt_dim = self.bert.config.hidden_size


    def forward(self, image, input_ids, attention_mask):
        # image: (B, C, H, W)
        B = image.size(0)
        img_feats = self.swin_features(image)   # (B, feat, 1, 1)
        img_feats = img_feats.view(B, -1)       # (B, img_dim)

        # text
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        txt_feats = outputs.pooler_output       # (B, txt_dim)

        return img_feats, txt_feats


class SimpleFusionHead(nn.Module):
    """
    Fusion MLP: concatenates image and text features and projects to joint embedding.
    """
    def __init__(self, img_dim, txt_dim, joint_dim=256, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(img_dim + txt_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, joint_dim)
        )

    def forward(self, img_feats, txt_feats):
        x = torch.cat([img_feats, txt_feats], dim=1)
        joint = self.fusion(x)
        return joint


class CrossModalFusion(nn.Module):
    """
    Text queries attend over global image features.
    """
    def __init__(self, img_dim, txt_dim, joint_dim=256, num_heads=4):
        super().__init__()
        # project into a common space for attention
        self.query_proj = nn.Linear(txt_dim, joint_dim)
        self.key_proj   = nn.Linear(img_dim, joint_dim)
        self.value_proj = nn.Linear(img_dim, joint_dim)

        self.attn = nn.MultiheadAttention(joint_dim, num_heads, batch_first=True)

        # final projection after concatenating attended img + text
        self.output = nn.Sequential(
            nn.Linear(joint_dim + txt_dim, joint_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, img_feats, txt_feats):
        # img_feats: (B, img_dim), txt_feats: (B, txt_dim)
        B = img_feats.size(0)

        # prepare Q, K, V as sequences of length 1
        Q = self.query_proj(txt_feats).unsqueeze(1)   # (B, 1, joint_dim)
        K = self.key_proj(img_feats).unsqueeze(1)     # (B, 1, joint_dim)
        V = self.value_proj(img_feats).unsqueeze(1)   # (B, 1, joint_dim)

        # cross-attention: text queries, image keys/values
        attended, _ = self.attn(Q, K, V)              # (B, 1, joint_dim)
        attended = attended.squeeze(1)                # (B, joint_dim)

        # concatenate with original text embedding
        x = torch.cat([attended, txt_feats], dim=1)   # (B, joint_dim + txt_dim)
        return self.output(x)                         # (B, joint_dim)

class GatedFusion(nn.Module):
    def __init__(self, img_dim, txt_dim, joint_dim=256):
        super().__init__()
        # individual projections
        self.proj_img = nn.Linear(img_dim, joint_dim)
        self.proj_txt = nn.Linear(txt_dim, joint_dim)
        # gating
        self.gate = nn.Sequential(
            nn.Linear(img_dim + txt_dim, joint_dim),
            nn.Sigmoid()
        )

    def forward(self, img_feats, txt_feats):
        gi = self.proj_img(img_feats)   # (B, joint_dim)
        gt = self.proj_txt(txt_feats)   # (B, joint_dim)
        # compute gate z ∈ [0,1]^joint_dim
        z = self.gate(torch.cat([img_feats, txt_feats], dim=1))
        # fuse: z * gi + (1 - z) * gt
        fused = z * gi + (1 - z) * gt
        return fused

# compact bilinear pooling (MCB), multi‑modal factorized bilinear (MFB), or Mutan fusion

