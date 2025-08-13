import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from typing import Union, List, Dict
import numpy as np

class ExplanationEngine:
    def __init__(
        self,
        fusion_model: torch.nn.Module,
        classifier_head: torch.nn.Module,
        image_size=(224,224),
        ig_steps: int = 50,
        device: torch.device = torch.device("cpu")
    ):
        """
        fusion_model    : image+text fusion backbone
        classifier_head : final head mapping fused features → logits
        image_size      : H x W of output heatmaps
        ig_steps        : n steps for Integrated Gradients
        device          : device to run the explanation on
        """
        self.fusion_model    = fusion_model
        self.classifier_head = classifier_head
        self.image_size      = image_size
        self.ig_steps        = ig_steps
        self.device = device

    def compute_attention_map(
        self,
        attn_tensor: torch.Tensor,
        grid_size: int
    ) -> np.ndarray:
        # attn_tensor: (B,1,N_patches) or (1,1,N_patches)
        scores = attn_tensor[0].squeeze().cpu()                  # (N,)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        grid   = scores.view(1,1,grid_size,grid_size)           # (1,1,G,G)
        up     = F.interpolate(grid, size=self.image_size, mode='bilinear', align_corners=False)
        return up.squeeze().detach().numpy()                             # (H,W)

    def _forward_patches_batchfirst(self, img_patches, txt_feats, target_idx):
        """
        img_patches : (B, Np, E) or (B, G1, G2, E)
        txt_feats   : (B, T, E)
        Returns logits for the requested target index (shape: (B,1))
        """
        # Flatten if 4D -> (B, G1*G2, E)
        if img_patches.dim() == 4:
            B, G1, G2, E = img_patches.shape
            img_patches = img_patches.view(B, G1 * G2, E)
        elif img_patches.dim() != 3:
            raise ValueError(f"img_patches must be 3D or 4D, got {tuple(img_patches.shape)}")

        # keep batch-first and contiguous
        img_patches = img_patches.contiguous()
        txt_feats = txt_feats.contiguous()

        # create img_global as mean over patches (same as training)
        img_global = img_patches.mean(dim=1)   # (B, E)

        # Call fusion layer — it returns (fused, attn_dict) or (fused, None)
        fusion_out = self.fusion_model(
            img_global = img_global,
            img_patch  = img_patches,
            txt_feats  = txt_feats,
            return_attention = False
        )

        # normalize handling: fusion_out can be tuple (fused, attn) or a single tensor
        if isinstance(fusion_out, tuple) or isinstance(fusion_out, list):
            fused = fusion_out[0]
        else:
            fused = fusion_out

        # fused should be (B, joint_dim); now run classifier_head to get logits (B, num_classes)
        logits = self.classifier_head(fused)

        # return logits for the requested class index
        return logits[:, target_idx].unsqueeze(-1)


    def compute_ig_map_for_target(self, img_global, img_patches, txt_feats, target_idx, steps=50, internal_batch_size=1):
        """
        Returns HxW IG heatmap over image patches for the first item in the batch.
        - img_patches: (B, Np, E)  or (B, G, G, E)
        - txt_feats:   (B, T, E)
        """
        # Ensure inputs are on the right device and have proper grad settings
        img_patches = img_patches.clone().detach().to(self.device).requires_grad_(True)
        txt_feats   = txt_feats.clone().detach().to(self.device)  # text is fixed, no grads

        # Use IntegratedGradients with a forward that accepts only the image patches
        ig = IntegratedGradients(lambda ip: self._forward_patches_batchfirst(ip, txt_feats, target_idx))

        # Baseline = zeros like image patches
        baselines = torch.zeros_like(img_patches, device=self.device)

        # Attribute (use small internal_batch_size to reduce memory / intermediate views)
        attributions = ig.attribute(
            inputs=img_patches,
            baselines=baselines,
            n_steps=steps,
            internal_batch_size=internal_batch_size
        )  # (B, Np, E) or (B, G, G, E)

        # Reduce embed dim to single importance per patch
        if attributions.dim() == 4:
            B, G1, G2, E = attributions.shape
            assert G1 == G2, f"Expected square grid for patches, got {G1}x{G2}"
            att = attributions.norm(p=1, dim=-1).view(B, G1 * G2)  # L1 over E
            G = G1
            att = att.view(B, G, G)
        else:
            B, Np, E = attributions.shape
            G = int(Np ** 0.5)
            assert G * G == Np, f"Number of patches {Np} is not a perfect square"
            att = attributions.norm(p=1, dim=-1)  # (B, Np)
            att = att.view(B, G, G)

        # ensure float
        att = att.to(dtype=torch.float32)

        # robust min/max over last two dims (works across PyTorch versions)
        try:
            min_vals = att.amin(dim=(-2, -1), keepdim=True)   # (B,1,1)
            max_vals = att.amax(dim=(-2, -1), keepdim=True)   # (B,1,1)
        except Exception:
            flat = att.view(B, -1)                             # (B, G*G)
            min_vals = flat.min(dim=1, keepdim=True)[0].view(B, 1, 1)
            max_vals = flat.max(dim=1, keepdim=True)[0].view(B, 1, 1)

        # Normalize to [0,1] safely
        range_vals = (max_vals - min_vals)
        att = (att - min_vals) / (range_vals + 1e-8)
        att = att.clamp(0.0, 1.0)

        # guard against any NaNs
        att = torch.nan_to_num(att, nan=0.0, posinf=1.0, neginf=0.0)

        return att[0].detach().cpu().numpy()

    def explain(
        self,
        img_global: torch.Tensor,
        img_patches: torch.Tensor,
        txt_feats: torch.Tensor,
        attn_weights: Dict[str,torch.Tensor],
        targets: Union[int, List[int]]
    ) -> Dict[str, Union[np.ndarray, Dict[int,np.ndarray]]]:
        """
        Returns:
          - 'attention_map': np.ndarray (H×W)
          - 'ig_maps':       dict of target → np.ndarray (H×W)
        """
        # Attention
        N = attn_weights['txt2img'].shape[-1]
        G = int(N**0.5)
        attention_map = self.compute_attention_map(attn_weights['txt2img'], grid_size=G)

        # IG maps (handle single or list of targets)
        if isinstance(targets, int):
            targets = [targets]

        ig_maps = {}
        for t in targets:
            ig_maps[t] = self.compute_ig_map_for_target(
                img_global[0:1].to(self.device),
                img_patches[0:1].to(self.device),
                txt_feats[0:1].to(self.device),
                t
            )

        return {
            'attention_map': attention_map,
            'ig_maps':       ig_maps
        }
