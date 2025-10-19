from pathlib import Path
import sys
try:
    base = Path(__file__).resolve().parent.parent
except NameError:
    base = Path.cwd().parent
sys.path.append(str(base))

import wandb
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
from safetensors.torch import save_model
import random

from DataHandler import parse_openi_xml, build_dataloader
from Model import Backbones
from Model import SwinModelForFinetune
from Helpers import Config
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import WeightedRandomSampler
from LabelData import disease_groups, normal_groups, finding_groups, symptom_groups
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    average_precision_score, roc_auc_score,
    precision_recall_curve, classification_report
)
from safetensors.torch import load_file, save_file

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "models"
CONFIG_DIR = BASE_DIR / "configs"
SPLIT_DIR = BASE_DIR / "splited_data"
OUTPUT_DIR = BASE_DIR / "finetune_swin_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class FocalBCEWithLogits(nn.Module):
    """
    Multi-label focal binary cross-entropy on logits.
    alpha: None | tensor(shape=[L]) or scalar. If provided, it's weight for positives.
    gamma: focusing parameter.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        if alpha is not None and not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha, dtype=torch.float32)
        self.alpha = alpha
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: (B, L), targets: same shape (0/1 floats)
        prob = torch.sigmoid(logits)
        ce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")  # (B,L)
        p_t = prob * targets + (1 - prob) * (1 - targets)  # (B,L)
        mod = (1 - p_t) ** self.gamma
        loss = mod * ce
        if self.alpha is not None:
            alpha_pos = self.alpha.to(logits.device)
            alpha_t = alpha_pos * targets + (1 - alpha_pos) * (1 - targets)
            loss = alpha_t * loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
    
class HybridLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, bce_weight=0.5, focal_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=alpha)
        self.focal = FocalBCEWithLogits(alpha=alpha, gamma=gamma, reduction="mean")
        self.bce_w = bce_weight
        self.focal_w = focal_weight

    def forward(self, logits, targets):
        return self.bce_w * self.bce(logits, targets) + self.focal_w * self.focal(logits, targets)

class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for Multi-Label Classification
    Ref: https://arxiv.org/abs/2009.14119

    Added optional `pos_weight` (per-class multiplier applied to the positive part of the loss).
    """
    def __init__(
        self,
        gamma_pos=0,
        gamma_neg=4,
        clip=0.05,
        eps=1e-8,
        disable_torch_grad_focal_loss=True,
        pos_weight=None,
    ):
        super(AsymmetricLoss, self).__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

        # store pos_weight (convert to tensor if needed)
        if pos_weight is not None and not torch.is_tensor(pos_weight):
            # allow list/ndarray/tensor/torch.Tensor input
            try:
                pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
            except Exception:
                # fallback: leave as-is (will raise later if incompatible)
                pass
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        """
        logits: (B, L)
        targets: (B, L) in {0,1}
        """
        probas = torch.sigmoid(logits)
        targets = targets.type_as(logits)

        # Asymmetric clipping (for negatives)
        if self.clip is not None and self.clip > 0:
            pt = (probas + self.clip).clamp(max=1)
        else:
            pt = probas

        # Loss calculation
        loss_pos = targets * torch.log(probas.clamp(min=self.eps))
        loss_neg = (1 - targets) * torch.log((1 - pt).clamp(min=self.eps))

        # apply per-class pos_weight if provided (broadcastable)
        if self.pos_weight is not None:
            pw = self.pos_weight.to(logits.device)
            # ensure shape is (1, C) for broadcasting across batch dimension
            if pw.dim() == 1:
                pw = pw.view(1, -1)
            loss_pos = loss_pos * pw

        loss = loss_pos + loss_neg

        # Asymmetric focusing
        if self.gamma_pos > 0 or self.gamma_neg > 0:
            pt_pos = probas * targets
            pt_neg = (1 - probas) * (1 - targets)
            pt_comb = pt_pos + pt_neg

            if self.disable_torch_grad_focal_loss:
                pt_comb = pt_comb.detach()

            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            loss *= (1 - pt_comb) ** one_sided_gamma

        return -loss.mean()
    
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensures deterministic behavior for some operations
    torch.backends.cudnn.deterministic = True

def make_multilabel_sampler(records):
    """
    records: list of dicts, each record has 'labels' as list/array of 0/1 for L classes.
    returns: WeightedRandomSampler instance
    """
    labels = np.array([r['labels'] for r in records], dtype=np.float32)  # shape (N, L)
    class_counts = labels.sum(axis=0)  # shape (L,)
    N, L = labels.shape
    eps = 1e-6

    # avoid division by zero
    class_counts_safe = np.where(class_counts > 0, class_counts, N)
    class_weights = (N / (class_counts_safe + eps)).astype(np.float32)  # shape (L,)

    # sample weights
    sample_weights = labels.dot(class_weights)  # shape (N,)
    sample_weights = np.where(sample_weights > 0, sample_weights, np.min(sample_weights[sample_weights>0]) if np.any(sample_weights>0) else 1.0)
    
    # create sampler
    return WeightedRandomSampler(weights=torch.from_numpy(sample_weights),
                                    num_samples=len(sample_weights),
                                    replacement=True)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def smooth_labels(labels, smoothing=0.01):
    # labels in {0,1}, output in [smoothing, 1-smoothing]
    return labels * (1.0 - smoothing) + 0.5 * smoothing

def make_state_dict_no_shared_storage(state_dict: dict):
    """
    Return a copy of state_dict where any tensors that share underlying storage
    are cloned (moved to CPU) so they no longer share memory. This makes the
    dict safe to save via safetensors (which rejects shared-storage tensors).
    """
    # map storage_id -> list of keys that share that storage
    storage_map = {}
    for k, v in state_dict.items():
        if not torch.is_tensor(v):
            continue
        # ensure on cpu for stable storage pointer; .storage().data_ptr() is int
        try:
            sid = (v.storage().data_ptr(), v.storage().size())
        except Exception:
            # Fallback: use id of storage object
            sid = id(v.storage())
        storage_map.setdefault(sid, []).append(k)

    new_state = {}
    # For keys whose tensors share storage we will clone for all keys except the first
    handled = set()
    for sid, keys in storage_map.items():
        if len(keys) == 1:
            k = keys[0]
            t = state_dict[k]
            # move to cpu (safetensors expects CPU tensors) and detach/copy reference
            new_state[k] = t.detach().cpu()
            handled.add(k)
        else:
            # multiple keys share the same storage: keep first as cpu tensor,
            # clone CPU copy for the rest to avoid shared memory
            first = True
            for k in keys:
                t = state_dict[k]
                if first:
                    new_state[k] = t.detach().cpu()
                    first = False
                else:
                    # break shared storage by cloning a CPU copy
                    new_state[k] = t.detach().cpu().clone()
                handled.add(k)

    # copy over non-tensor items (if any)
    for k, v in state_dict.items():
        if k in handled:
            continue
        # for safety: put tensors on cpu
        if torch.is_tensor(v):
            new_state[k] = v.detach().cpu()
        else:
            new_state[k] = v

    return new_state

def build_finetune_subset(xml_dir, dicom_root, combined_groups, split_dir,
                            finetune_ratio=0.4, train_ratio=0.75, seed=42, max_retry=20):
    """
    Build a balanced finetune subset:
        - Start from TRAIN SPLIT ONLY (no leakage from val/test).
        - Take finetune_ratio portion of train (e.g. 0.4 = 40%).
        - Split that subset into train_ratio for train and rest for validation.
        - Retry random splits if validation set has any label with all zeros.
    """
    import pandas as pd
    rng = np.random.default_rng(seed)

    # Load all parsed records and label CSV
    records = parse_openi_xml(xml_dir, dicom_root, combined_groups=combined_groups)
    lab_path = BASE_DIR / "outputs" / "openi_labels_final.csv"
    if not lab_path.exists():
        raise FileNotFoundError(f"Labels CSV not found: {lab_path}")
    labels_df = pd.read_csv(lab_path).set_index("id")

    label_cols = sorted(combined_groups.keys())
    filtered = []
    for r in records:
        if r["id"] in labels_df.index:
            r2 = r.copy()
            r2["labels"] = labels_df.loc[r["id"], label_cols].tolist()
            filtered.append(r2)

    # Load original train split IDs
    with open(split_dir / "train_split_ids.json") as f:
        train_ids = set(json.load(f))
    train_records_full = [r for r in filtered if r["id"] in train_ids]

    # Retry to get a balanced validation subset
    for attempt in range(max_retry):
        idx = rng.permutation(len(train_records_full))
        subset_size = int(len(idx) * finetune_ratio)
        subset_idx = idx[:subset_size]
        subset_records = [train_records_full[i] for i in subset_idx]

        split = int(len(subset_records) * train_ratio)
        ft_train_records = subset_records[:split]
        ft_val_records = subset_records[split:]

        val_labels = np.array([r["labels"] for r in ft_val_records])
        all_zero = (val_labels.sum(axis=0) == 0).sum()
        all_one = (val_labels.sum(axis=0) == len(val_labels)).sum()

        if all_zero == 0 and all_one == 0:
            print(f"[INFO] Found balanced split on attempt {attempt + 1}")
            break
        else:
            print(f"[WARN] Unbalanced val split on attempt {attempt + 1} (all_zero={all_zero}, all_one={all_one})")
    else:
        print(f"[WARN] Could not perfectly balance after {max_retry} attempts; continuing with last split.")

    print(f"[INFO] Finetune subset built: total_train={len(train_records_full)}, "
            f"subset={len(subset_records)} (train={len(ft_train_records)}, val={len(ft_val_records)})")

    return ft_train_records, ft_val_records, label_cols

def freeze_backbone(backbones: Backbones):
    for p in backbones.parameters():
        p.requires_grad = False


def unfreeze_backbone(backbones: Backbones):
    for p in backbones.parameters():
        p.requires_grad = True

def unfreeze_last_stage(backbones: Backbones):
    """
    Unfreeze the last Swin stage (stage4/layer4) and all LayerNorms.
    Keep earlier stages frozen for stability.
    """
    for name, p in backbones.swin.named_parameters():
        if (
            "layers.3" in name
            or "stage4" in name
            or "layer4" in name
            or "norm" in name  # unfreeze all normalization layers
        ):
            p.requires_grad = True
        else:
            p.requires_grad = False

    # Optional debug print
    trainable = sum(p.numel() for p in backbones.swin.parameters() if p.requires_grad)
    total = sum(p.numel() for p in backbones.swin.parameters())
    print(f"[DEBUG] Partial unfreeze: {trainable:,}/{total:,} params trainable ({trainable/total:.2%})")

def lr_finder(optimizer, model, criterion, train_loader, device,
              init_lr=1e-7, final_lr=1e-1, beta=0.98):
    lrs, losses = [], []
    best_loss = float('inf')

    model_state = copy.deepcopy(model.state_dict())

    num_batches = len(train_loader) - 1
    mult = (final_lr / init_lr) ** (1/num_batches)
    lr = init_lr
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    avg_loss, batch_num = 0.0, 0
    model.train()
    for batch in train_loader:
        batch_num += 1
        imgs = batch["image"].to(device)
        labels = batch["labels"].to(device).float()

        optimizer.zero_grad()
        logits, _, _ = model(imgs)
        loss = criterion(logits, labels)

        avg_loss = beta * avg_loss + (1-beta) * loss.item()
        smooth_loss = avg_loss / (1 - beta**batch_num)

        lrs.append(lr)
        losses.append(smooth_loss)

        if batch_num > 1 and smooth_loss > 4 * best_loss:
            break
        if smooth_loss < best_loss:
            best_loss = smooth_loss

        loss.backward()
        optimizer.step()

        lr *= mult
        for pg in optimizer.param_groups:
            pg['lr'] = lr

    model.load_state_dict(model_state)
    return lrs, losses

def quick_eval(model, train_loader, val_loader, criterion, device, steps=200):
    """
    Run a short warmup training to evaluate candidate ASL settings.
    Returns composite score (macro F1 + AUROC).
    """
    model_copy = copy.deepcopy(model).to(device)
    opt = optim.AdamW(model_copy.parameters(), lr=1e-4)
    model_copy.train()

    it = iter(train_loader)
    for i in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)
        imgs, labels = batch["image"].to(device), batch["labels"].float().to(device)
        opt.zero_grad()
        logits, _, _ = model_copy(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        opt.step()

    # Quick validation
    model_copy.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            imgs, labels = batch["image"].to(device), batch["labels"].float().to(device)
            logits, _, _ = model_copy(imgs)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_logits, all_labels = np.vstack(all_logits), np.vstack(all_labels)
    probs = torch.sigmoid(torch.from_numpy(all_logits)).numpy()

    y_bin = (probs > 0.5).astype(int)
    macro_f1 = f1_score(all_labels, y_bin, average="macro", zero_division=0)
    try:
        macro_auc = roc_auc_score(all_labels, probs, average="macro")
    except:
        macro_auc = 0.5
    return 0.5 * macro_f1 + 0.5 * macro_auc

def thresholds_min_precision(y_true, probs, p_target=0.20, min_thr=0.02):
    """Return per-class thresholds that achieve at least p_target precision if possible, else maximize F1.
       p_target: desired minimum precision (try 0.20 or 0.25).
       min_thr: minimal allowed threshold so very tiny probs can be accepted.
    """
    C = y_true.shape[1]
    thr = np.zeros(C, dtype=float)
    for c in range(C):
        p, r, t = precision_recall_curve(y_true[:, c], probs[:, c])
        # find thresholds where precision >= target
        idxs = np.where(p >= p_target)[0]
        if idxs.size > 0:
            # pick the threshold with highest recall among those that satisfy precision >= p_target
            best_idx = idxs[np.argmax(r[idxs])]
            thr[c] = t[best_idx] if best_idx < len(t) else 0.5
        else:
            # fallback to F1-optimal
            f1s = 2 * p * r / (p + r + 1e-12)
            if f1s.size > 0:
                best_idx = np.nanargmax(f1s)
                thr[c] = t[best_idx] if best_idx < len(t) else 0.5
            else:
                thr[c] = 0.5
        thr[c] = max(thr[c], min_thr)
    return thr

def train(
    cfg=None,
    finetune_mode="partial",     # "frozen" | "partial" | "full"
    swin_init_ckpt=None,
    out_path=OUTPUT_DIR,
    epochs=None,
    batch_size=None,
    lr=None,
    weight_decay=1e-2,
    device=None,
    augment=True,
    loss = "hybrid",
    see_debug=False,
    seed=None
):
    """
    Finetune Swin model with given config and records.

    Args:
        cfg (Config): Config object containing hyperparameters.
        finetune_mode (str): One of "frozen", "partial", or "full".
        swin_init_ckpt (str): Path to Swin model checkpoint.
        out_path (str): Path to output checkpoint.
        epochs (int): Number of epochs to train.
        batch_size (int): Batch size for training.
        lr (float): Learning rate.
        weight_decay (float): Weight decay for Adam optimizer.
        device (str): Device to use (e.g. "cuda", "cpu").
        augment (bool): Whether to use data augmentation for training.
        loss (str): Whether to use focal loss. Options: "hybrid", "bce", "asl", "asl-auto"
        seed (int): Random seed for reproducibility.
    Returns:
        None

    Notes:
        - Swin model is instantiated with given config.
        - Records are split into train and validation sets.
        - Backbones are frozen or partially unfrozen depending on finetune_mode.
        - Finetune mode is set to "frozen" by default.
        - Model is trained with Adam optimizer and cosine annealing learning rate scheduler.
        - Validation is performed at the end of each epoch.
        - Best checkpoint is saved to out_path.
    """
    device = device or get_device()
    print(f"[INFO] device: {device}")

    if seed is not None:
        set_seed(seed)

    combined_groups = {**disease_groups, **normal_groups, **finding_groups, **symptom_groups}
    cfg = Config.load(CONFIG_DIR / 'config.yaml') if cfg is None else cfg

    epochs = cfg.epochs if epochs is None else epochs
    batch_size = cfg.batch_size if batch_size is None else batch_size
    lr = 1e-4 if lr is None else lr

    # Prepare records
    xml_dir = BASE_DIR / 'data' / 'openi' / 'xml' / 'NLMCXR_reports' / 'ecgen-radiology'
    dicom_root = BASE_DIR / 'data' / 'openi' / 'dicom'
    
    train_records, val_records, label_cols = build_finetune_subset(
        xml_dir, dicom_root, combined_groups, SPLIT_DIR,
        finetune_ratio=0.4,
        train_ratio=0.75,
        seed=seed if seed is not None else 42
    )
    print(f"[INFO] train records: {len(train_records)}, val records: {len(val_records)}, labels: {len(label_cols)}")

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    # create weighted sampler for the training records
    sampler = make_multilabel_sampler(train_records)

    # build dataloaders
    train_loader = build_dataloader(
        train_records,
        batch_size=batch_size,
        mean=imagenet_mean,
        std=imagenet_std,
        augment=augment,
        max_length=cfg.text_dim,
        sampler=sampler,
    )

    # build validation dataloader
    val_loader = build_dataloader(
        val_records,
        batch_size=batch_size,
        mean=imagenet_mean,
        std=imagenet_std,
        augment=False,
        max_length=cfg.text_dim
    )

    # Compute pos_weight
    label_counts = np.array([r['labels'] for r in train_records]).sum(axis=0)
    num_samples = len(train_records)
    label_counts = torch.tensor(label_counts, dtype=torch.float32)
    pos_weight = ((num_samples - label_counts) / (label_counts + 1e-9))
    support = label_counts.clone()
    cap = torch.where(support < 20,
                    torch.tensor(10.0, device=pos_weight.device),
                    torch.tensor(5.0, device=pos_weight.device))
    pos_weight = torch.min(pos_weight, cap).clamp(min=1.0).to(device)

    # ensure minimum 1.0 still
    pos_weight = pos_weight.clamp(min=1.0)
    pos_weight = pos_weight.to(device)

    if see_debug:
        print("[DEBUG] pos_weight stats:", pos_weight.min().item(), pos_weight.max().item())

    label_freq = (label_counts / float(num_samples)).clamp(min=1e-6)  # tensor len L, on CPU or device?
    # invert and normalize into [0,1]
    inv = 1.0 / label_freq
    alpha_pos = 0.7 + 0.3 * (inv / inv.max())   # maps roughly to [0.7, 1.0]
    alpha_pos = alpha_pos.clamp(0.01, 0.99).to(device)
    alpha_pos = alpha_pos.to(device)  # move to device for loss later
    gamma = 2.0  # experiment with 1.0..3.0

    if see_debug:
        print(f"[DEBUG] focal alpha_pos sample (first 10): {alpha_pos[:10].tolist()}, gamma={gamma}")

    # Instantiate backbone
    backbones = Backbones(
        img_backbone=cfg.image_backbone,
        swin_checkpoint_path=Path(swin_init_ckpt) if swin_init_ckpt else (MODEL_DIR / "swin_checkpoint.safetensors"),
        bert_local_dir=None,
        pretrained=True
    ).to(device)

    # Confirm swin checkpoint
    swin_path = Path(swin_init_ckpt) if swin_init_ckpt else (MODEL_DIR / "swin_checkpoint.safetensors")
    try:
        exists = swin_path.exists()
    except Exception:
        exists = False
    print(f"[DIAG] Swin checkpoint argument (swin_init_ckpt): {swin_init_ckpt}")
    print(f"[DIAG] Resolved swin_path: {swin_path} (exists={exists})")

    # Print backbone summary: total params and trainable params
    total_params = sum(p.numel() for p in backbones.parameters())
    trainable_params = sum(p.numel() for p in backbones.parameters() if p.requires_grad)
    print(f"[DIAG] Backbone params: {trainable_params:,}/{total_params:,} trainable ({trainable_params/total_params:.2%})")

    # Finetune config
    if finetune_mode == "frozen":
        print("[INFO] Finetune mode: frozen (only train classifier head)")
        freeze_backbone(backbones)
    elif finetune_mode == "partial":
        print("[INFO] Finetune mode: partial (unfreeze last stage)")
        unfreeze_last_stage(backbones)
    elif finetune_mode == "full":
        print("[INFO] Finetune mode: full (unfreeze entire backbone)")
        unfreeze_backbone(backbones)
    else:
        raise ValueError("finetune_mode must be one of: frozen, partial, full")

    model = SwinModelForFinetune(backbones=backbones, num_classes=len(label_cols)).to(device)

    # Report actual trainable params after applying freeze/unfreeze
    total_params = sum(p.numel() for p in backbones.parameters())
    trainable_params = sum(p.numel() for p in backbones.parameters() if p.requires_grad)
    print(f"[DIAG] After freeze/unfreeze: Backbone params: {trainable_params:,}/{total_params:,} trainable ({trainable_params/total_params:.2%})")

    if see_debug:
        # Print loader batch info (you already do some of these)
        batch = next(iter(train_loader))
        imgs = batch['image']
        labels = batch['labels']
        print("imgs.shape", imgs.shape)              # should be [B, 3, H, W]
        print("imgs.dtype", imgs.dtype)
        print("imgs.min/max", float(imgs.min()), float(imgs.max()))
        print("imgs.mean/std", float(imgs.mean()), float(imgs.std()))
        print("labels.shape", labels.shape)
        print("labels.min/max", labels.min(), labels.max())
        print("labels.dtype", labels.dtype)

        # Confirm label values are float 0/1
        labels = labels.float()
        print("labels unique counts:", torch.unique(labels, return_counts=True))

        # Check model output shape on a batch
        model.eval()
        with torch.no_grad():
            logits, _, _ = model(imgs.to(device))
        print("logits.shape", logits.shape)          # should be [B, num_classes]
        print("logits mean/min/max/std (cpu)", logits.detach().cpu().numpy().mean(),
                                                logits.detach().cpu().numpy().min(),
                                                logits.detach().cpu().numpy().max(),
                                                logits.detach().cpu().numpy().std())

        batch = next(iter(train_loader))
        print("sampled batch label sums:", batch['labels'].sum(dim=0)[:30].tolist())

        train_labels = np.array([r['labels'] for r in train_records])
        val_labels = np.array([r['labels'] for r in val_records])
        print("train prevalence (first 20):", train_labels.sum(axis=0) / len(train_labels))
        print("val prevalence (first 20):", val_labels.sum(axis=0) / len(val_labels))

    # Separate params
    backbone_params = [p for _, p in backbones.named_parameters() if p.requires_grad]
    backbone_param_ids = {id(p) for p in backbone_params}
    head_params = [p for _, p in model.named_parameters() if p.requires_grad and id(p) not in backbone_param_ids]
    if not head_params:
        head_params = [p for _, p in model.named_parameters() if p.requires_grad]

    param_groups = [{"params": head_params, "lr": lr}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr * 0.1})

    # Loss function 
    if loss == "hybrid":
        criterion = HybridLoss(alpha=pos_weight, gamma=gamma, bce_weight=0.6, focal_weight=0.4)
    elif loss == "focal":
        criterion = FocalBCEWithLogits(alpha=alpha_pos, gamma=gamma, reduction="mean")
    elif loss == "bce":
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif loss == "asl":
        criterion = AsymmetricLoss(gamma_pos=1, gamma_neg=3, clip=0.05)
    elif loss == "asl-auto":
        candidate_settings = [
            {"gamma_pos": 0, "gamma_neg": 2, "clip": 0},
            {"gamma_pos": 0, "gamma_neg": 4, "clip": 0},
            {"gamma_pos": 1, "gamma_neg": 2, "clip": 0.05},
            {"gamma_pos": 1, "gamma_neg": 4, "clip": 0.05},
        ]
        best_cfg, best_score = None, -1
        for cand in candidate_settings:
            print(f"[ASL-AUTO] Trying config: {cand}")
            crit = AsymmetricLoss(**cand)
            score = quick_eval(model, train_loader, val_loader, crit, device, steps=200)
            print(f"[ASL-AUTO] Candidate score={score:.4f}")
            if score > best_score:
                best_cfg, best_score = cand, score
        print(f"[ASL-AUTO] Selected config: {best_cfg} with score={best_score:.4f}")
        criterion = AsymmetricLoss(**best_cfg)
    else:
        raise ValueError("loss must be one of: focal, bce, hybrid, asl")

    # Build a temporary optimizer for LR finder
    tmp_optimizer = optim.AdamW(model.parameters(), lr=1e-7, weight_decay=weight_decay)

    print("[INFO] Running LR finder...")
    lrs, losses = lr_finder(tmp_optimizer, model, criterion, train_loader, device)

    # Plot curve (optional if running in notebook)
    try:
        plt.plot(lrs, losses)
        plt.xscale("log")
        plt.xlabel("Learning rate")
        plt.ylabel("Smoothed loss")
        plt.title("LR Finder")
        plt.savefig(out_path / "lr_finder_curve.png")
        plt.close()
    except Exception as e:
        print("[WARN] Could not plot LR curve:", e)

    # Pick best LR = 10x lower than LR where loss starts rising
    min_loss_idx = int(np.argmin(losses))
    best_lr = lrs[min_loss_idx]
    chosen_lr = best_lr / 10
    print(f"[INFO] LR Finder suggests: {chosen_lr:.2e}")

    # Rebuild optimizer with discriminative LR
    param_groups = [{"params": head_params, "lr": chosen_lr}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": chosen_lr * 0.01})

    optimizer = optim.AdamW(param_groups, weight_decay=weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    use_label_smoothing = True
    if loss in ("focal", "hybrid"):
        use_label_smoothing = False

    best_score = -float("inf")
    best_epoch = -1
    out_path_file = out_path / "finetuned_swin_labelaware.safetensors"
    patience = cfg.patience
    epochs_no_improve = 0

    if see_debug:
        # run a quick overfit test on 1 batch (temp code)
        model.train()
        batch = next(iter(train_loader))
        imgs = batch["image"].to(device)
        labels = batch["labels"].to(device)
        tmp_opt = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=chosen_lr, weight_decay=weight_decay)
        for i in range(20):
            tmp_opt.zero_grad()
            logits, _, _ = model(imgs)
            loss = criterion(logits, labels)

            logits_np = logits.detach().cpu().numpy()
            print("logits mean/min/max/std:", logits_np.mean(), logits_np.min(), logits_np.max(), logits_np.std())

            loss.backward()
            tmp_opt.step()
            if i % 5 == 0:
                print(f"[OVERFIT] step {i}, loss={loss.item():.6f}")
    
    # Main training loop
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Finetuning Swin Epoch {epoch}/{epochs}")

        if epoch == 1:
            freeze_backbone(backbones)          # only head trains
        elif epoch == 5:
            unfreeze_last_stage(backbones)      # last stage + norms
        elif epoch == 10:
            unfreeze_backbone(backbones)        # all layers train

        # ---- Dynamic reweighting ----
        epoch_factor = min(1.0, epoch / 20.0)
        dyn_pos_weight = (1 - epoch_factor) * torch.ones_like(pos_weight) + epoch_factor * pos_weight

        if loss == "hybrid":
            criterion = HybridLoss(alpha=dyn_pos_weight, gamma=gamma, bce_weight=0.6, focal_weight=0.4)
        elif loss == "focal":
            criterion = FocalBCEWithLogits(alpha=alpha_pos, gamma=gamma, reduction="mean")
        elif loss == "bce":
            criterion = nn.BCEWithLogitsLoss(pos_weight=dyn_pos_weight)
        elif loss == "asl":
            criterion = AsymmetricLoss(pos_weight=dyn_pos_weight)

        for batch in pbar:
            imgs = batch["image"].to(device)
            if use_label_smoothing:
                labels = smooth_labels(batch["labels"].to(device).float(), smoothing=0.01)
            else:
                labels = batch["labels"].to(device).float()

            optimizer.zero_grad()
            logits, _, _ = model(imgs)
            loss = criterion(logits, labels)

            loss.backward()

            # Clip gradient norm
            head_grad_sum = 0.0
            for p in head_params:
                if p.grad is not None:
                    head_grad_sum += float(p.grad.abs().sum().cpu())

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item() * imgs.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        scheduler.step()

        # Validation
        model.eval()
        val_loss, all_logits, all_labels = 0.0, [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validate"):
                imgs = batch["image"].to(device)
                labels = batch["labels"].to(device).float()
                logits, _, _ = model(imgs)
                loss = criterion(logits, labels)
                val_loss += loss.item() * imgs.size(0)
                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        all_logits = np.vstack(all_logits)
        all_labels = np.vstack(all_labels)

        # ----- Temperature scaling -----
        try:
            logits_tensor = torch.from_numpy(all_logits).float().to(device)
            labels_tensor = torch.from_numpy(all_labels).float().to(device)

            log_T = torch.nn.Parameter(torch.tensor(0.0, device=device))
            optimizer_T = optim.LBFGS([log_T], lr=1.0, max_iter=50, line_search_fn="strong_wolfe")

            def _closure():
                optimizer_T.zero_grad()
                T = torch.exp(log_T)
                loss_T = nn.functional.binary_cross_entropy_with_logits(
                    logits_tensor / T, labels_tensor
                )
                loss_T.backward()
                return loss_T

            optimizer_T.step(_closure)
            T_val = float(torch.exp(log_T).detach().cpu().item())
            T_val = max(T_val, 1.0)   # enforce at least 1.0; try 1.2 if negatives stay extreme
            print(f"[INFO] Calibrated temperature (clipped min=1.0): {T_val:.4f}")
        except Exception as e:
            print(f"[WARN] Temperature scaling failed: {e}")
            T_val = 1.0

        # ----- Apply temperature & compute calibrated probabilities -----
        calibrated_probs = torch.sigmoid(
            torch.from_numpy(all_logits).float().to(device) / max(T_val, 1e-6)
        ).cpu().numpy()

        np.save(out_path/ "last_val_temperature.npy", np.array([T_val], dtype=float))

        # ----- Metrics -----
        probs = calibrated_probs  # use calibrated version here
        # Compute best per-class thresholds for this epoch
        best_thr = thresholds_min_precision(all_labels, probs, p_target=0.20, min_thr=0.02)
        best_thr = np.clip(best_thr, 0.02, 0.9)

        # Apply thresholds to get binary predictions
        best_thr = np.clip(best_thr, 0.05, 0.9)
        y_bin = (probs > best_thr[None, :]).astype(int)

        macro_f1 = f1_score(all_labels, y_bin, average='macro', zero_division=0)
        micro_f1 = f1_score(all_labels, y_bin, average='micro', zero_division=0)
        macro_prec = precision_score(all_labels, y_bin, average='macro', zero_division=0)
        micro_prec = precision_score(all_labels, y_bin, average='micro', zero_division=0)
        macro_rec = recall_score(all_labels, y_bin, average='macro', zero_division=0)
        micro_rec = recall_score(all_labels, y_bin, average='micro', zero_division=0)

        try:
            macro_ap = average_precision_score(all_labels, probs, average='macro')
            micro_ap = average_precision_score(all_labels, probs, average='micro')
        except Exception:
            macro_ap, micro_ap = np.nan, np.nan

        try:
            macro_auc = roc_auc_score(all_labels, probs, average='macro')
        except Exception:
            macro_auc = np.nan

        composite = 0.5 * macro_f1 + 0.5 * macro_auc

        print(
            f"[E{epoch}] train_loss={epoch_train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"F1(macro/micro)={macro_f1:.3f}/{micro_f1:.3f} | "
            f"Prec(macro/micro)={macro_prec:.3f}/{micro_prec:.3f} | "
            f"Rec(macro/micro)={macro_rec:.3f}/{micro_rec:.3f} | "
            f"AP(macro/micro)={macro_ap:.3f}/{micro_ap:.3f} | "
            f"AUC(macro)={macro_auc:.3f}"
        )

        wandb.log({
            "swin/train_loss": epoch_train_loss,
            "swin/val_loss": val_loss,
            "swin/f1_macro": macro_f1,
            "swin/f1_micro": micro_f1,
            "swin/precision_macro": macro_prec,
            "swin/precision_micro": micro_prec,
            "swin/recall_macro": macro_rec,
            "swin/recall_micro": micro_rec,
            "swin/auc_macro": macro_auc,
            "swin/ap_micro": micro_ap,
            "swin/ap_macro": macro_ap,
            "swin/composite": composite,
            "swin/epoch": epoch
        })

        # Save best or count non-improving epochs
        if composite > best_score:
            best_score = composite
            best_epoch = epoch
            epochs_no_improve = 0

            # Save best model
            orig_state = model.state_dict()
            safe_state = make_state_dict_no_shared_storage(orig_state)
            save_file(safe_state, str(out_path_file))
            print(f"[INFO] Saved best model -> {out_path_file}")

            # Save thresholds for later test-time inference
            np.save(out_path / "last_val_best_thresholds.npy", best_thr)

            # Debug save for post-mortem
            np.savez(out_path / f"debug_epoch{epoch}_run{wandb.run.id}.npz",
                    probs=probs, labels=all_labels, pos_weight=pos_weight.detach().cpu().numpy(),
                    alpha_pos=alpha_pos.detach().cpu().numpy() if isinstance(alpha_pos, torch.Tensor) else np.array(alpha_pos),
                    best_thr=best_thr, T_val=np.array([T_val]))
            # quick summary of true-positive prob distribution
            tp_mask = (all_labels == 1)
            if tp_mask.any():
                tp_probs = probs[tp_mask]
                print(f"[DIAG] TP probs: mean={tp_probs.mean():.4f}, median={np.median(tp_probs):.4f}, pct>0.5={(tp_probs>0.5).mean():.2f}")
            else:
                print("[DIAG] No positive labels in this validation set.")

            # Save per-class report only for best epoch
            report = classification_report(
                all_labels, y_bin,
                target_names=label_cols,
                zero_division=0,
                output_dict=True
            )
            per_class_path = out_path / f"per_class_report_best.json"
            with open(per_class_path, "w") as f:
                json.dump(report, f, indent=2)
            print(f"[INFO] Saved best per-class report -> {per_class_path}")

            # Log best metrics to W&B summary
            wandb.run.summary["best_epoch"] = epoch
            wandb.run.summary["best_val_loss"] = val_loss
            wandb.run.summary["best_f1_macro"] = macro_f1
            wandb.run.summary["best_f1_micro"] = micro_f1
            wandb.run.summary["best_precision_macro"] = macro_prec
            wandb.run.summary["best_precision_micro"] = micro_prec
            wandb.run.summary["best_recall_macro"] = macro_rec
            wandb.run.summary["best_recall_micro"] = micro_rec
            wandb.run.summary["best_ap_macro"] = macro_ap
            wandb.run.summary["best_ap_micro"] = micro_ap
            wandb.run.summary["best_auc_macro"] = macro_auc
            wandb.run.summary["best_composite"] = composite
            
        else:
            epochs_no_improve += 1
            print(f"[INFO] No improvement for {epochs_no_improve} epoch(s)")

        # Early stopping condition
        if epochs_no_improve >= patience:
            print(f"[EARLY STOPPING] No improvement for {patience} epochs — stopping training early.")
            break
    
    # --- Save best finetune metrics as JSON ---
    best_path = BASE_DIR / "best"
    if not best_path.exists():
        best_path.mkdir(parents=True)
    best_json_path = best_path / "best_swin_finetune_metrics.json"

    best_payload = {
        "best_epoch": best_epoch,
        "best_composite": best_score,
        "best_checkpoint": str(out_path),
        "metrics": {
            "val_loss": wandb.run.summary.get("best_val_loss", None),
            "f1_macro": wandb.run.summary.get("best_f1_macro", None),
            "f1_micro": wandb.run.summary.get("best_f1_micro", None),
            "precision_macro": wandb.run.summary.get("best_precision_macro", None),
            "precision_micro": wandb.run.summary.get("best_precision_micro", None),
            "recall_macro": wandb.run.summary.get("best_recall_macro", None),
            "recall_micro": wandb.run.summary.get("best_recall_micro", None),
            "ap_macro": wandb.run.summary.get("best_ap_macro", None),
            "ap_micro": wandb.run.summary.get("best_ap_micro", None),
            "auc_macro": wandb.run.summary.get("best_auc_macro", None),
        },
        "wandb_run": {
            "name": wandb.run.name if wandb.run else None,
            "id": wandb.run.id if wandb.run else None,
            "project": wandb.run.project if wandb.run else None,
        }
    }

    # Per-class threshold calibration
    print("[INFO] Running final per-class threshold calibration...")
    state_dict = load_file(str(out_path_file))
    model.load_state_dict(state_dict)
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Final threshold calibration"):
            imgs = batch["image"].to(device)
            labels = batch["labels"].to(device).float()
            logits, _, _ = model(imgs)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_logits = np.vstack(all_logits)
    all_labels = np.vstack(all_labels)
    probs = torch.sigmoid(torch.from_numpy(all_logits)).numpy()

    final_thr = np.zeros(probs.shape[1], dtype=float)
    for c in range(probs.shape[1]):
        try:
            p, r, t = precision_recall_curve(all_labels[:, c], probs[:, c])
            f1s = 2 * p * r / (p + r + 1e-12)
            idx = np.nanargmax(f1s)
            final_thr[c] = float(t[idx]) if idx < len(t) else 0.5
        except Exception:
            final_thr[c] = 0.5

    np.save(out_path / "final_per_class_thresholds.npy", final_thr)
    print(f"[INFO] Saved per-class thresholds to {out_path / 'final_per_class_thresholds.npy'}")

    y_bin_final = (probs > final_thr[None, :]).astype(int)

    macro_f1_final = f1_score(all_labels, y_bin_final, average="macro", zero_division=0)
    micro_f1_final = f1_score(all_labels, y_bin_final, average="micro", zero_division=0)
    macro_prec_final = precision_score(all_labels, y_bin_final, average="macro", zero_division=0)
    micro_prec_final = precision_score(all_labels, y_bin_final, average="micro", zero_division=0)
    macro_rec_final = recall_score(all_labels, y_bin_final, average="macro", zero_division=0)
    micro_rec_final = recall_score(all_labels, y_bin_final, average="micro", zero_division=0)

    try:
        macro_ap_final = average_precision_score(all_labels, probs, average="macro")
        micro_ap_final = average_precision_score(all_labels, probs, average="micro")
    except Exception:
        macro_ap_final, micro_ap_final = np.nan, np.nan

    try:
        macro_auc_final = roc_auc_score(all_labels, probs, average="macro")
    except Exception:
        macro_auc_final = np.nan

    composite_final = 0.5 * macro_f1_final + 0.5 * macro_auc_final
    
    best_payload["post_calibration"] = {
        "f1_macro": macro_f1_final,
        "f1_micro": micro_f1_final,
        "precision_macro": macro_prec_final,
        "precision_micro": micro_prec_final,
        "recall_macro": macro_rec_final,
        "recall_micro": micro_rec_final,
        "ap_macro": macro_ap_final,
        "ap_micro": micro_ap_final,
        "auc_macro": macro_auc_final,
        "composite": composite_final,
    }
    
        # ---- Log post-calibration results to W&B ----
    wandb.run.summary["postcal_f1_macro"] = macro_f1_final
    wandb.run.summary["postcal_f1_micro"] = micro_f1_final
    wandb.run.summary["postcal_precision_macro"] = macro_prec_final
    wandb.run.summary["postcal_precision_micro"] = micro_prec_final
    wandb.run.summary["postcal_recall_macro"] = macro_rec_final
    wandb.run.summary["postcal_recall_micro"] = micro_rec_final
    wandb.run.summary["postcal_ap_macro"] = macro_ap_final
    wandb.run.summary["postcal_ap_micro"] = micro_ap_final
    wandb.run.summary["postcal_auc_macro"] = macro_auc_final
    wandb.run.summary["postcal_composite"] = composite_final

    # Optional grouped log so it shows nicely in the run charts
    wandb.log({
        "swin/postcal_f1_macro": macro_f1_final,
        "swin/postcal_f1_micro": micro_f1_final,
        "swin/postcal_precision_macro": macro_prec_final,
        "swin/postcal_precision_micro": micro_prec_final,
        "swin/postcal_recall_macro": macro_rec_final,
        "swin/postcal_recall_micro": micro_rec_final,
        "swin/postcal_ap_macro": macro_ap_final,
        "swin/postcal_ap_micro": micro_ap_final,
        "swin/postcal_auc_macro": macro_auc_final,
        "swin/postcal_composite": composite_final,
    })

    try:
        with open(best_json_path, "w", encoding="utf8") as f:
            json.dump(best_payload, f, indent=2)
        print(f"[INFO] Saved best Swin finetune metrics -> {best_json_path}")
    except Exception as e:
        print(f"[WARN] Could not save best Swin finetune metrics: {e}")
    

    print(f"[DONE] Best composite score: {best_score:.4f} at epoch {best_epoch}")
    print(f"[INFO] Final best checkpoint written to {out_path}")


if __name__ == "__main__":
    mode = "partial"
    loss_use = "asl-auto"
    import datetime

    wandb.init(
        project="finetune-swin",
        name="finetune-swin-labelaware-" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + f"-{mode}-{loss_use}",
        group="finetune-swin",
        job_type="finetune",
    )
    
    train(
        finetune_mode=mode,
        loss=loss_use,
        seed=2709,
        epochs=8
    )

