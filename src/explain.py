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
        # Here we support batch dimension by taking first element
        scores = attn_tensor[0].squeeze().cpu()                  # (N,)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        grid   = scores.view(1,1,grid_size,grid_size)           # (1,1,G,G)
        up     = F.interpolate(grid, size=self.image_size, mode='bilinear', align_corners=False)
        return up.squeeze().detach().numpy()                             # (H,W)

    def compute_ig_map_for_target(
        self,
        img_global: torch.Tensor,
        img_patches: torch.Tensor,
        txt_feats: torch.Tensor,
        target: int
    ) -> np.ndarray:
        """
        Run IG for one target, using both image and text features.
        img_global: (1, D_img); img_patches: (1, N, D_patch); txt_feats: (1, D_txt)
        """
        img_global  = img_global.to(self.device)
        img_patches = img_patches.to(self.device)
        txt_feats   = txt_feats.to(self.device)

        # fuse with gradient tracking
        output = self.fusion_model(
            img_global.requires_grad_(True),
            img_patches.requires_grad_(True),
            txt_feats.requires_grad_(True),
            return_attention=False
        )

        if isinstance(output, tuple):
            fused = output[0]
        else:
            fused = output

        # wrapper returns the logit for `target`
        def _wrapper(x):
            return self.classifier_head.to(self.device)(x)[:, target]

        # move fused to CPU to avoid GPU/Captum issues
        fused_for_ig = fused.detach().to(self.device)

        ig = IntegratedGradients(_wrapper)
        baseline = torch.zeros_like(fused_for_ig)
        attr     = ig.attribute(fused_for_ig, baselines=baseline, n_steps=self.ig_steps)  # (1, D)
        
        heat     = attr.squeeze(0)
        heat     = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        G        = int(heat.shape[0] ** 0.5)
        grid     = heat.view(1,1,G,G)
        up       = F.interpolate(grid, size=self.image_size, mode='bilinear', align_corners=False)
        return up.squeeze().cpu().numpy()  # (H,W)

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
