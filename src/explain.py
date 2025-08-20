import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from typing import Union, List, Dict
import numpy as np
import math

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
        classifier_head : final head mapping fused features to logits
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
        return self.upscale_heatmap(up.squeeze().detach().numpy(), target_size=self.image_size)   # (H,W)

    def _forward_patches_batchfirst(self, img_patches, txt_feats, target_idx=None):
        """
        img_patches : (B, Np, E) or (B, G1, G2, E)
        txt_feats   : (B, T, E)
        If target_idx is None -> return logits (B, num_classes)
        If target_idx is int -> return logits[:, target_idx].unsqueeze(-1) (B,1)
        """
        # Flatten if 4D -> (B, G1*G2, E)
        if img_patches.dim() == 4:
            B, G1, G2, E = img_patches.shape
            img_patches = img_patches.view(B, G1 * G2, E)
        elif img_patches.dim() != 3:
            raise ValueError(f"img_patches must be 3D or 4D, got {tuple(img_patches.shape)}")

        img_patches = img_patches.contiguous()
        txt_feats = txt_feats.contiguous()

        # create img_global as mean over patches (same as training)
        img_global = img_patches.mean(dim=1)   # (B, E)

        # Call fusion layer — it returns (fused, attn_dict) or fused tensor
        fusion_out = self.fusion_model(
            img_global = img_global,
            img_patch  = img_patches,
            txt_feats  = txt_feats,
            return_attention = False
        )

        if isinstance(fusion_out, (tuple, list)):
            fused = fusion_out[0]
        else:
            fused = fusion_out

        logits = self.classifier_head(fused)  # (B, num_classes)

        if target_idx is None:
            return logits
        else:
            return logits[:, int(target_idx)].unsqueeze(-1)

    def compute_gradcam_map_for_target(self,
                                    img_global: torch.Tensor,
                                    img_patches: torch.Tensor,
                                    txt_feats: torch.Tensor,
                                    target_idx: int) -> np.ndarray:
        """
        Embedding-based Grad-CAM for patch embeddings.
        - img_patches: (B, Np, E) or (B, G, G, E)
        - Returns: HxW numpy heatmap for the first item in batch normalized to [0,1]
        """
        # Move to device and ensure patches require grad
        device = getattr(self, "device", img_patches.device)
        img_patches = img_patches.clone().detach().to(device).requires_grad_(True)  # (B,Np,E)
        txt_feats   = txt_feats.clone().detach().to(device)
        img_global  = img_global.clone().detach().to(device)

        fused_out = self.fusion_model(
            img_global = img_patches.mean(dim=1),
            img_patch  = img_patches,
            txt_feats  = txt_feats,
            return_attention = False
        )
        if isinstance(fused_out, (tuple, list)):
            fused_out = fused_out[0]
        logits = self.classifier_head(fused_out)  # (B, num_classes)

        # Pick target score scalar: sum across batch to get a single scalar
        if logits.dim() == 2 and logits.size(1) == 1:
            target_score = logits.squeeze().sum()
        else:
            target_score = logits[:, int(target_idx)].sum()

        # compute gradients w.r.t. patches ONLY (no param grads)
        grads = torch.autograd.grad(
            outputs=target_score,
            inputs=img_patches,
            retain_graph=True,
            create_graph=False,
            allow_unused=True
        )[0]  # (B, Np, E) or None

        if grads is None:
            raise RuntimeError("Gradients wrt img_patches are None. Ensure forward path depends on img_patches.")

        # Channel-weighted sum analogous to Grad-CAM: elementwise product then sum over embedding dim
        cam = (grads * img_patches).sum(dim=-1)  # (B, Np)

        # Keep positive contributions only (ReLU)
        cam = torch.relu(cam)  # (B, Np)

        # Reshape into grid
        B, Np = cam.shape
        G = int(np.sqrt(Np))
        if G * G == Np:
            cam_grid = cam.view(B, G, G)  # (B, G, G)
        else:
            # fallback: treat as 1 x Np and later upsample (keeps spatial ordering if non-square)
            cam_grid = cam.view(B, 1, Np)

        # Upsample to image size (self.image_size should be (H,W) tuple)
        target_size = getattr(self, "image_size", (224, 224))
        cam_grid = cam_grid.unsqueeze(1)  # (B,1,G,G) or (B,1,1,Np)
        cam_up = F.interpolate(cam_grid, size=target_size, mode="bilinear", align_corners=False)  # (B,1,H,W)
        cam_up = cam_up.squeeze(1)  # (B,H,W)

        # Normalize per-sample to [0,1]
        cam_np = cam_up.detach().cpu().numpy()
        out_map = cam_np[0]  # take first sample
        eps = 1e-8
        out_map = (out_map - out_map.min()) / (out_map.max() - out_map.min() + eps)
        out_map = np.nan_to_num(out_map, nan=0.0, posinf=1.0, neginf=0.0)
        return out_map

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

        ig_up = self.upscale_heatmap(att[0].detach().cpu().numpy(), target_size=self.image_size)
        return ig_up

    def _attn_to_patch_tensor(self, att, img_patches, method="mean"):
        """
        Convert attention weights of arbitrary (B, *, *) shape into a torch tensor shaped (B,1,Np)
        where Np = number of image patches (img_patches.shape[1]).
        Returns None if it cannot reliably infer a per-patch map.
        """
        if att is None:
            return None

        # convert to torch tensor on same device as img_patches for shape checking
        if not torch.is_tensor(att):
            try:
                att = torch.as_tensor(att, device=img_patches.device)
            except Exception:
                att = torch.tensor(att, device=img_patches.device)

        # If heads dimension present (B, heads, Lq, Lk) -> average heads
        if att.dim() == 4:
            # assume shape (B, heads, Lq, Lk)
            att = att.mean(dim=1)  # -> (B, Lq, Lk)

        # now expect (B, Lq, Lk) or (Lq, Lk) or (B, L)
        if att.dim() == 3:
            B, Lq, Lk = att.shape
            Np = img_patches.shape[1]

            # Case: txt2img typical -> (B, 1, Np)
            if Lk == Np and Lq == 1:
                return att[:, 0:1, :].detach()

            # Case: img2txt when keys pooled -> (B, Np, 1)
            if Lq == Np and Lk == 1:
                return att[:, :, 0:1].transpose(1, 2).detach()  # (B,1,Np)

            # Case: img2txt when keys are tokens -> (B, Np, L_tokens)
            if Lq == Np and Lk > 1:
                if method == "mean":
                    per_patch = att.mean(dim=-1)  # (B, Np)
                elif method == "max":
                    per_patch = att.max(dim=-1)[0]
                else:
                    per_patch = att.mean(dim=-1)
                return per_patch.unsqueeze(1).detach()  # (B,1,Np)

            # Case: tokens->patches -> (B, L_tokens, Np)
            if Lk == Np and Lq > 1:
                per_patch = att.mean(dim=1)  # average queries -> (B, Np)
                return per_patch.unsqueeze(1).detach()

            # ambiguous: try aggregating over keys to yield (B, Lq) and see if Lq == Np
            agg_key = att.mean(dim=-1)  # (B, Lq)
            if agg_key.shape[1] == Np:
                return agg_key.unsqueeze(1).detach()

            # try averaging over queries -> (B, Lk)
            agg_q = att.mean(dim=1)
            if agg_q.shape[1] == Np:
                return agg_q.unsqueeze(1).detach()

            # give up -> return None
            return None

        # att.dim() == 2 -> maybe (B, Np) or (Np,) etc.
        if att.dim() == 2:
            if att.shape[1] == img_patches.shape[1]:
                return att.unsqueeze(1).detach()
            # maybe shape (Np, ) or (1, Np)
            if att.shape[0] == img_patches.shape[1]:
                return att.unsqueeze(0).unsqueeze(1).detach()
            return None

        # other dims -> cannot handle
        return None

    def upscale_heatmap(self, heatmap, target_size=None):
        if target_size is None:
            target_size = self.image_size
        t = torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        t_up = F.interpolate(t, size=target_size, mode="bilinear", align_corners=False)
        return t_up.squeeze().numpy()

    def explain(
        self,
        img_global: torch.Tensor,
        img_patches: torch.Tensor,
        txt_feats: torch.Tensor,
        attn_weights: Dict[str,torch.Tensor],
        targets: Union[int, List[int]]
    ) -> Dict[str, Union[np.ndarray, Dict[int,np.ndarray]]]:
        # Attention
        """
        Computes explanation maps for a single input using three methods.

        Args:
            img_global: (B, E) global image features
            img_patches: (B, Np, E) or (B, G, G, E) image patches
            txt_feats:   (B, T, E) text features
            attn_weights: attention weights from fusion layer, expected to have key 'txt2img'
            targets: single int or list of ints for the class indices to explain

        Returns:
            dict with three keys:
                - 'attention_map': (H, W) attention map over image patches
                - 'ig_maps':       dict of target to (H, W) IG map over image patches
                - 'gradcam_maps':  dict of target to (H, W) Grad-CAM map over image patches
        """
        attention_maps = {}

        for key in ["txt2img", "img2txt", "comb"]:
            att = attn_weights.get(key, None)
            if att is None:
                continue

            # Convert arbitrary att tensor into (B,1,Np) if possible
            att_patch_tensor = self._attn_to_patch_tensor(att, img_patches, method="mean")
            if att_patch_tensor is None:
                # debug: show shape and skip
                try:
                    shape_info = tuple(att.shape)
                except Exception:
                    shape_info = "unknown"
                print(f"[WARN] skipping {key}: could not convert attention to per-patch vector (att shape {shape_info})")
                continue

            # get number of patches
            N = att_patch_tensor.shape[-1]
            G = int(math.sqrt(N))

            if G * G != N:
                # still not square -> skip
                print(f"[WARN] skipping {key}: attention length {N} not square (n_patches)")
                continue

            # att_patch_tensor is (B,1,Np) -> pass to compute_attention_map
            attention_maps[key] = self.compute_attention_map(att_patch_tensor, grid_size=G)

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

        # Grad-CAM maps
        gradcam_maps = {}
        for t in targets:
            gradcam_maps[t] = self.compute_gradcam_map_for_target(
                img_global[0:1],
                img_patches[0:1],
                txt_feats[0:1],
                t
            )

        return {
            "attention_map": attention_maps,
            "ig_maps": ig_maps,
            "gradcam_maps": gradcam_maps
        }