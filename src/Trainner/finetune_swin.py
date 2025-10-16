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
from tqdm import tqdm
from safetensors.torch import save_model

from DataHandler import parse_openi_xml, build_dataloader
from Model import Backbones
from Model import SwinModelForFinetune
from Helpers import Config
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import WeightedRandomSampler
from LabelData import disease_groups, normal_groups, finding_groups, symptom_groups
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    average_precision_score, roc_auc_score,
    precision_recall_curve
)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "models"
CONFIG_DIR = BASE_DIR / "configs"
SPLIT_DIR = BASE_DIR / "splited_data"
OUTPUT_DIR = MODEL_DIR
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

    return WeightedRandomSampler(weights=torch.from_numpy(sample_weights),
                                    num_samples=len(sample_weights),
                                    replacement=True)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

def train(
    cfg=None,
    finetune_mode="partial",     # "frozen" | "partial" | "full"
    swin_init_ckpt=None,
    out_path=None,
    epochs=None,
    batch_size=None,
    lr=None,
    weight_decay=1e-2,
    device=None,
    augment=True,
    use_focal = True
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
        use_focal (bool): Whether to use focal loss
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
        seed=cfg.seed
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

    # Compute pos_weight
    label_counts = np.array([r['labels'] for r in train_records]).sum(axis=0)
    num_samples = len(train_records)
    label_counts = torch.tensor(label_counts, dtype=torch.float32)
    pos_weight = ((num_samples - label_counts) / (label_counts + 1e-9))
    pos_weight = pos_weight.clamp(min=1.0, max=5.0)  # clamp to avoid extreme values
    pos_weight = pos_weight.to(device)

    print("[DEBUG] pos_weight stats:", pos_weight.min().item(), pos_weight.max().item())

    label_freq = (label_counts / float(num_samples)).clamp(min=1e-6)  # tensor len L, on CPU or device?
    # invert and normalize into [0,1]
    inv = 1.0 / label_freq
    alpha_pos = 0.5 + 0.5 * (inv / inv.max())   # maps to [0.5, 1.0]
    alpha_pos = alpha_pos.clamp(0.01, 0.99).to(device)
    alpha_pos = alpha_pos.to(device)  # move to device for loss later
    gamma = 2.0  # experiment with 1.0..3.0
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

    optimizer = optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    if use_focal:
        # pass alpha_pos as positive-class weighting into focal loss
        criterion = FocalBCEWithLogits(alpha=alpha_pos, gamma=gamma, reduction="mean")
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_score = -float("inf")
    best_epoch = -1
    out_path = Path(out_path or (MODEL_DIR / "finetuned_swin_labelaware.safetensors"))
    patience = cfg.patience
    epochs_no_improve = 0


    # run a quick overfit test on 1 batch (temp code)
    model.train()
    batch = next(iter(train_loader))
    imgs = batch["image"].to(device)
    labels = batch["labels"].to(device)
    tmp_opt = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
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

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Finetuning Swin Epoch {epoch}/{epochs}")
        for batch in pbar:
            imgs = batch["image"].to(device)
            labels = batch["labels"].to(device).float()

            optimizer.zero_grad()
            logits, _, _ = model(imgs)
            loss = criterion(logits, labels)

            loss.backward()

            # after backward and before optimizer.step
            head_grad_sum = 0.0
            for p in head_params:
                if p.grad is not None:
                    head_grad_sum += float(p.grad.abs().sum().cpu())
            print("head_grad_sum:", head_grad_sum)


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
            print(f"[INFO] Calibrated temperature: {T_val:.4f}")
        except Exception as e:
            print(f"[WARN] Temperature scaling failed: {e}")
            T_val = 1.0

        # ----- Apply temperature & compute calibrated probabilities -----
        calibrated_probs = torch.sigmoid(
            torch.from_numpy(all_logits).float().to(device) / max(T_val, 1e-6)
        ).cpu().numpy()

        np.save(MODEL_DIR / "last_val_temperature.npy", np.array([T_val], dtype=float))

        # ----- Metrics -----
        probs = calibrated_probs  # use calibrated version here
        y_bin = (probs > 0.5).astype(int)

        best_thr = np.zeros(probs.shape[1], dtype=float)
        for c in range(probs.shape[1]):
            try:
                p, r, t = precision_recall_curve(all_labels[:, c], probs[:, c])
                f1s = 2 * p * r / (p + r + 1e-12)
                if f1s.size > 0:
                    idx = np.nanargmax(f1s)
                    best_thr[c] = float(t[idx]) if idx < len(t) else 0.5
                else:
                    best_thr[c] = 0.5
            except Exception:
                best_thr[c] = 0.5

        np.save(MODEL_DIR / "last_val_best_thresholds.npy", best_thr)

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

        composite = 0.3 * macro_f1 + 0.7 * macro_auc

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
            try:
                save_model(model, str(out_path))
                print(f"[INFO] Saved best model checkpoint (safe) to {out_path} (epoch {epoch})")
            except Exception as e:
                print(f"[ERROR] safetensors save_model failed: {e}")
                raise
            
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

    try:
        with open(best_json_path, "w", encoding="utf8") as f:
            json.dump(best_payload, f, indent=2)
        print(f"[INFO] Saved best Swin finetune metrics -> {best_json_path}")
    except Exception as e:
        print(f"[WARN] Could not save best Swin finetune metrics: {e}")
    

    print(f"[DONE] Best composite score: {best_score:.4f} at epoch {best_epoch}")
    print(f"[INFO] Final best checkpoint written to {out_path}")


if __name__ == "__main__":
    train(epochs=3, finetune_mode="frozen",  lr=1e-4,
            swin_init_ckpt=str(MODEL_DIR / "finetuned_swin_labelaware.safetensors"))
