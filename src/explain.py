import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from typing import Union, List, Dict
import numpy as np
import math
import warnings

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

    def avg_heads(self, att):
        """
        Given a tensor of arbitrary shape, if it has a `heads` dimension (i.e., it has shape (B, H, Lq, Lk)),
        average over the `heads` dimension to return a tensor of shape (B, Lq, Lk). Otherwise, return the
        input tensor as-is.
        """
        
        if att is None:
            return None
        att = torch.as_tensor(att) if not torch.is_tensor(att) else att
        if att.dim() == 4:  # (B, H, Lq, Lk)
            return att.mean(dim=1)  # (B, Lq, Lk)
        return att

    def compute_attention_map(self, attn_tensor: torch.Tensor, grid_size: int) -> np.ndarray:
        # accept torch/numpy and various shapes, unify to (B,1,Np)
        """
        Compute attention map from attention tensor.

        Parameters
        ----------
        attn_tensor: torch.Tensor
            The attention tensor.
        grid_size: int
            The size of the output attention map.

        Returns
        -------
        np.ndarray
            The attention map as a numpy array.
        """
        if attn_tensor is None:
            return None
        if not torch.is_tensor(attn_tensor):
            attn_tensor = torch.as_tensor(attn_tensor)
        
        # unify shapes
        if attn_tensor.dim() == 1:
            attn_tensor = attn_tensor.unsqueeze(0).unsqueeze(0)
        elif attn_tensor.dim() == 2:
            # assume (B, Np) or (1, Np)
            attn_tensor = attn_tensor.unsqueeze(1)
        elif attn_tensor.dim() == 3:
            # assume (B,1,Np) or (B,Np,1) -> try to coerce
            if attn_tensor.shape[1] != 1 and attn_tensor.shape[2] == 1:
                attn_tensor = attn_tensor.transpose(1, 2)
        else:
            # unexpected dims -> try to reduce
            attn_tensor = attn_tensor.reshape(attn_tensor.shape[0], 1, -1)

        scores = attn_tensor[0].squeeze().cpu()
        # Normalize
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

        # safety check: expected length == grid_size*grid_size
        expected = grid_size * grid_size
        if scores.numel() != expected:
            # If mismatch, try to resize via interpolation instead of reshaping (fallback)
            grid = scores.view(1, 1, 1, -1) # (1,1,1,N)
            up_t = F.interpolate(grid, size=self.image_size, mode='bilinear', align_corners=False)
            arr = up_t.squeeze().numpy()
            return self.upscale_heatmap(arr, target_size=self.image_size)

        grid = scores.view(1, 1, grid_size, grid_size)
        up = F.interpolate(grid, size=self.image_size, mode='bilinear', align_corners=False)
        return self.upscale_heatmap(up.squeeze().detach().numpy(), target_size=self.image_size)

    def upscale_heatmap(self, heatmap, target_size=None):
        """
        Upsample heatmap to target_size using bilinear interpolation.

        Args:
            heatmap: 2D numpy array of shape (H,W)
            target_size: tuple of ints (H2,W2) for desired output size

        Returns:
            2D numpy array of shape (H2,W2) upsampled from input heatmap
        """
        if target_size is None:
            target_size = self.image_size
        t = torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        t_up = F.interpolate(t, size=target_size, mode='bilinear', align_corners=False)
        return t_up.squeeze().numpy()

    def _forward_patches_batchfirst(self, img_patches, txt_feats, target_idx=None):
        """
        Call the fusion model and classifier head on a batch of image patches and text features.

        Args:
            img_patches: 3D or 4D tensor of shape (B, Np, E) or (B, G1, G2, E) where B is the batch size,
                Np is the number of patches, G1 and G2 are the grid size, and E is the embedding size.
            txt_feats: 3D tensor of shape (B, T, E) where B is the batch size, T is the sequence length,
                and E is the embedding size.
            target_idx: int or None. If int, the index of the target class to compute the gradient with respect to.
                If None, return the logits for all classes.

        Returns:
            If target_idx is None, returns a 2D tensor of shape (B, num_classes) containing the logits for all classes.
            If target_idx is int, returns a 2D tensor of shape (B, 1) containing the logit for the specified class.
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

        # call fusion model
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
        Compute Grad-CAM explanation map for a single input.

        Args:
            img_global: (B, E) global image features
            img_patches: (B, Np, E) or (B, G1, G2, E) image patches
            txt_feats:   (B, T, E) text features
            target_idx: int, the index of the target class to explain

        Returns:
            A 2D numpy array of shape (H, W) containing the explanation map.

        Notes:
            - Currently only supports batch size 1.
            - The returned map is normalized to [0,1] per sample.
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
        fused_out = fused_out[0] if isinstance(fused_out, (tuple, list)) else fused_out
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
        Computes an Integrated Gradients heatmap over image patches for a single target class.

        Args:
            img_global: (B, E) global image features
            img_patches: (B, Np, E) or (B, G, G, E) image patches
            txt_feats:   (B, T, E) text features
            target_idx:  int, index of the target class to explain
            steps:       int, number of IG steps
            internal_batch_size: int, batch size to use for internal computation

        Returns:
            (H, W) numpy heatmap for the first item in batch normalized to [0,1]
        """
        img_patches = img_patches.clone().detach().to(self.device).requires_grad_(True)
        txt_feats   = txt_feats.clone().detach().to(self.device)

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

        Parameters:
            att (torch tensor or None): attention weights
            img_patches (torch tensor): image patches (B, Np, E)
            method (str, optional): how to reduce attention weights when Lk > 1. Defaults to "mean"

        Returns:
            torch tensor (B,1,Np) or None
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
    
    def _attn_to_token_tensor(self, att, txt_feats, method="mean"):
        """
        Fallback: reduce attention matrix to token-level importance scores.

        Args:
            att (torch.Tensor): raw attention [B, H, T, T] or [B, T, T].
            txt_feats (torch.Tensor): text features [B, T, D].
            method (str): "mean" (default) or "max".

        Returns:
            torch.Tensor: reduced token-level vector [B, 1, T], normalized to [0,1].
        """
        if att is None:
            return None

        # ensure [B, H, T, T]
        if att.dim() == 3:
            att = att.unsqueeze(1)

        B, H, T, _ = att.shape

        # reduce across heads
        if method == "mean":
            v = att.mean(dim=1)   # [B, T, T]
        elif method == "max":
            v = att.max(dim=1)[0] # [B, T, T]
        else:
            raise ValueError(f"Unknown reduction method: {method}")

        # collapse context dimension → get a single score per token
        v = v.mean(dim=-1, keepdim=True).transpose(1, 2)  # [B, 1, T]

        # normalize to [0,1]
        v = (v - v.min(dim=-1, keepdim=True)[0]) / (v.max(dim=-1, keepdim=True)[0] - v.min(dim=-1, keepdim=True)[0] + 1e-8)

        return v

    def comb_attention_to_patch_vector(self, att, img_patches, min_mass_ratio=0.12):
        return self._comb_helper(att, img_patches, target_len=img_patches.shape[1], min_mass_ratio=min_mass_ratio, swap=False)

    def comb_attention_to_token_vector(self, att, txt_feats, min_mass_ratio=0.12):
        return self._comb_helper(att, txt_feats, target_len=txt_feats.shape[1], min_mass_ratio=min_mass_ratio, swap=True)

    def txt2img_to_patch_vector(self, att_txt2img, img_patches, reduction="mean"):
        """
        Extract a per-patch attention vector from the txt2img attention map.

        Args:
            att_txt2img (torch.Tensor): raw attention [B, H, T, Np] or [B, T, Np]
            img_patches (torch.Tensor): image patches [B, Np, E]
            reduction (str): reduction method, either "mean" (default) or "max"

        Returns:
            torch.Tensor: per-patch vector [B, 1, Np], normalized to [0,1]
        """
        att = self.avg_heads(att_txt2img)
        if att is None or att.dim() != 3:
            return None
        per_patch = att.mean(dim=1)
        return per_patch.unsqueeze(1).detach()

    def img2txt_to_token_vector(self, att_img2txt, txt_feats, reduction="mean"):
        """
        Extract a per-token attention vector from the img2txt attention map.

        Args:
            att_img2txt (torch.Tensor): raw attention [B, H, T, Nt] or [B, T, Nt]
            txt_feats (torch.Tensor): text features [B, Nt, D]
            reduction (str): reduction method, either "mean" (default) or "max"

        Returns:
            torch.Tensor: per-token vector [B, 1, Nt], normalized to [0,1]
        """
        att = self.avg_heads(att_img2txt)
        if att is None or att.dim() != 3:
            return None
        per_token = att.mean(dim=1)
        return per_token.unsqueeze(1).detach()
    
    def _comb_helper(self, att, other, target_len, min_mass_ratio=0.12, swap=False):
        """
        Generic sliding-window helper. If swap==False: slide over keys for patches.
        If swap==True: extract token block by swapping Lq/Lk roles.
        """
        if att is None:
            return None
        att = self.avg_heads(att)
        if att is None or att.dim() != 3:
            return None
        
        if other is not None and hasattr(other, "device"):
            try:
                att = att.to(other.device)
            except Exception:
                warnings.warn(f"Cannot transfer attention matrix to {other.device}; continuing on att's device")

        B, Lq, Lk = att.shape
        N = int(target_len)
        # quick exact matches
        if Lk == N:
            return att.mean(dim=1).unsqueeze(1).detach()   # (B,1,N)
        if Lq == N:
            return att.mean(dim=-1).unsqueeze(1).detach()
        # choose primary axis to slide
        if not swap:
            primary_len = Lk
            sums = att.sum(dim=1)  # (B, Lk)
        else:
            primary_len = Lq
            sums = att.sum(dim=-1)  # (B, Lq)

        if primary_len < N:
            return None

        cumsum = torch.cumsum(sums, dim=-1)
        per_sample = []
        for b in range(B):
            row = cumsum[b]
            if primary_len == N:
                wins = row[-1].unsqueeze(0)
            else:
                end = row[N-1:]
                start = torch.cat((torch.tensor([0.0], device=row.device), row[:-N]))
                wins = end - start
            max_val, max_idx = torch.max(wins, dim=0)
            total = (sums[b].sum() + 1e-12)
            if (max_val / total) < min_mass_ratio:
                per_sample.append(torch.zeros(N, device=att.device))
                continue
            off = int(max_idx.item())
            if not swap:
                slice_block = att[b:b+1, :, off:off+N]   # (1, Lq, N)
                per_vec = slice_block.mean(dim=1).squeeze(0)  # (N,)
            else:
                slice_block = att[b:b+1, off:off+N, :]   # (1, N, Lk)
                per_vec = slice_block.mean(dim=-1).squeeze(0)
            per_sample.append(per_vec)
        out = torch.stack(per_sample, dim=0)  # (B, N)
        return out.unsqueeze(1).detach()      # (B,1,N)

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

        # txt2img -> per-patch
        att_txt2img = attn_weights.get("txt2img", None)
        p_from_txt = None
        if att_txt2img is not None:
            p_from_txt = self.txt2img_to_patch_vector(att_txt2img, img_patches)
            if p_from_txt is not None:
                N = p_from_txt.shape[-1]
                G = int(math.sqrt(N)) if int(math.sqrt(N)) ** 2 == N else None
                if G is None:
                    print(f"[WARN] txt2img skip: {N} not square")
                else:
                    attention_maps["txt2img"] = self.compute_attention_map(p_from_txt, grid_size=G)

        # img2txt -> per-token
        att_img2txt = attn_weights.get("img2txt", None)
        t_from_img = None
        if att_img2txt is not None:
            t_from_img = self.img2txt_to_token_vector(att_img2txt, txt_feats)
            if t_from_img is not None:
                attention_maps["img2txt"] = t_from_img[0, 0].cpu().numpy()

        # comb -> try both patch and token extraction (heuristic)
        att_comb = attn_weights.get("comb", None)
        p_from_comb = None
        t_from_comb = None

        if att_comb is not None:
            print("[DEBUG] comb att shape:", att_comb.shape,
                "min:", att_comb.min().item(), "max:", att_comb.max().item(),
                "mean:", att_comb.mean().item())

        if att_comb is not None:
            # patch-level comb
            p_from_comb = self.comb_attention_to_patch_vector(att_comb, img_patches, min_mass_ratio=0.06)
            if p_from_comb is None:
                p_from_comb = self._attn_to_patch_tensor(att_comb, img_patches, method="mean")
                print("[INFO] comb_img fallback: used mean patch tensor")

            if p_from_comb is not None:
                N = p_from_comb.shape[-1]
                G = int(math.sqrt(N)) if int(math.sqrt(N)) ** 2 == N else None
                if G is not None:
                    # if p_from_comb is constant zeros, consider fallback
                    if torch.allclose(p_from_comb, torch.zeros_like(p_from_comb)):
                        print("[INFO] comb_img was constant -> leaving out (fallback will be used)")
                    else:
                        attention_maps["comb_img"] = self.compute_attention_map(p_from_comb, grid_size=G)

            # token-level comb
            t_from_comb = self.comb_attention_to_token_vector(att_comb, txt_feats, min_mass_ratio=0.0)
            if t_from_comb is None:
                t_from_comb = self._attn_to_token_tensor(att_comb, txt_feats, method="mean")
                print("[INFO] comb_txt fallback: used mean token tensor")

            if t_from_comb is not None:
                # skip near-constant vectors
                if torch.allclose(t_from_comb, torch.zeros_like(t_from_comb)):
                    print("[INFO] comb_txt was constant -> leaving out (fallback will be used)")
                else:
                    attention_maps["comb_txt"] = t_from_comb[0, 0].detach().cpu().numpy()

        # Combine to final maps (weighted)
        final_patch = None
        if p_from_txt is not None and p_from_comb is not None:
            device = p_from_txt.device
            p_from_comb = p_from_comb.to(device)

            # ensure same patch length before arithmetic (trim to shorter)
            Lp_txt = p_from_txt.size(-1)
            Lp_comb = p_from_comb.size(-1)
            if Lp_txt != Lp_comb:
                minL = min(Lp_txt, Lp_comb)
                p_from_txt = p_from_txt[..., :minL]
                p_from_comb = p_from_comb[..., :minL]
                print(f"[INFO] patch-length mismatch: trimmed to {minL} (was {Lp_txt},{Lp_comb})")
            final_patch = 0.6 * p_from_txt + 0.4 * p_from_comb
        elif p_from_txt is not None:
            final_patch = p_from_txt
        elif p_from_comb is not None:
            final_patch = p_from_comb

        # Combine token maps (already had code for tokens) — keep your trimming approach
        final_token = None
        if t_from_img is not None and t_from_comb is not None:
            device = t_from_img.device
            t_from_comb = t_from_comb.to(device)

            # ensure same token length before arithmetic (trim to shorter)
            L_img = t_from_img.size(-1)
            L_comb = t_from_comb.size(-1)
            if L_img != L_comb:
                minL = min(L_img, L_comb)
                t_from_img = t_from_img[..., :minL]
                t_from_comb = t_from_comb[..., :minL]
                print(f"[INFO] token-length mismatch: trimmed to {minL} (was {L_img},{L_comb})")
            final_token = 0.6 * t_from_img + 0.4 * t_from_comb
        elif t_from_img is not None:
            final_token = t_from_img
        elif t_from_comb is not None:
            final_token = t_from_comb

        def _norm_and_to_numpy(x):
            if x is None:
                return None
            y = x.clone().detach().cpu()
            mi = y.view(y.size(0), -1).min(dim=1)[0].view(-1, 1, 1)
            ma = y.view(y.size(0), -1).max(dim=1)[0].view(-1, 1, 1)
            y = (y - mi) / (ma - mi + 1e-8)
            return y

        final_patch = _norm_and_to_numpy(final_patch)
        final_token = _norm_and_to_numpy(final_token)

        if final_patch is not None:
            N = final_patch.shape[-1]
            G = int(math.sqrt(N)) if int(math.sqrt(N)) ** 2 == N else None
            if G is not None:
                attention_maps["final_patch_map"] = self.compute_attention_map(final_patch, grid_size=G)
            else:
                attention_maps["final_patch_map"] = None
        else:
            attention_maps["final_patch_map"] = None

        attention_maps["final_token_map"] = final_token[0, 0].cpu().numpy() if final_token is not None else None

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