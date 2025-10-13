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
from safetensors.torch import save_file as save_safetensors

from DataHandler import parse_openi_xml, build_dataloader
from Model import Backbones
from Model import SwinModelForFinetune
from Helpers import Config
from torch.optim.lr_scheduler import CosineAnnealingLR
from LabelData import disease_groups, normal_groups, finding_groups, symptom_groups
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    average_precision_score, roc_auc_score
)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "models"
CONFIG_DIR = BASE_DIR / "configs"
SPLIT_DIR = BASE_DIR / "splited_data"
OUTPUT_DIR = MODEL_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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
    for name, p in backbones.swin.named_parameters():
        if "layers.3" in name or "layer4" in name or "blocks" in name and ("3" in name or "stage4" in name):
            p.requires_grad = True
        else:
            p.requires_grad = False


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
    augment=True
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

    epochs = cfg.epochs
    batch_size = cfg.batch_size
    lr = cfg.lr

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

    train_loader = build_dataloader(train_records, batch_size=batch_size, mean=0.5, std=0.25,
                                    augment=augment, max_length=cfg.text_dim)
    val_loader = build_dataloader(val_records, batch_size=batch_size, mean=0.5, std=0.25,
                                  augment=False, max_length=cfg.text_dim)

    # Compute pos_weight
    label_counts = np.array([r['labels'] for r in train_records]).sum(axis=0)
    num_samples = len(train_records)
    label_counts = torch.tensor(label_counts, dtype=torch.float32)
    pos_weight = ((num_samples - label_counts).clamp(min=1.0) / label_counts.clamp(min=1.0)).to(device)

    print("[DEBUG] pos_weight stats:", pos_weight.min().item(), pos_weight.max().item())

    # Instantiate backbone
    backbones = Backbones(
        img_backbone=cfg.image_backbone,
        swin_checkpoint_path=Path(swin_init_ckpt) if swin_init_ckpt else (MODEL_DIR / "swin_checkpoint.safetensors"),
        bert_local_dir=None,
        pretrained=True
    ).to(device)

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
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs * len(train_loader)), eta_min=1e-7)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_score = -float("inf")
    best_epoch = -1
    out_path = Path(out_path or (MODEL_DIR / "finetuned_swin_labelaware.safetensors"))

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Finetuning Swin Epoch {epoch}/{epochs}")
        for batch in pbar:
            imgs = batch["image"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits, _, _ = model(imgs)
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item() * imgs.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss, all_probs, all_labels = 0.0, [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validate"):
                imgs = batch["image"].to(device)
                labels = batch["labels"].to(device)
                logits, _, _ = model(imgs)
                loss = criterion(logits, labels)
                val_loss += loss.item() * imgs.size(0)
                all_probs.append(torch.sigmoid(logits).cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        all_probs = np.vstack(all_probs)
        all_labels = np.vstack(all_labels)
        y_bin = (all_probs > 0.5).astype(int)

        # Metrics
        macro_f1 = f1_score(all_labels, y_bin, average='macro', zero_division=0)
        micro_f1 = f1_score(all_labels, y_bin, average='micro', zero_division=0)
        macro_prec = precision_score(all_labels, y_bin, average='macro', zero_division=0)
        micro_prec = precision_score(all_labels, y_bin, average='micro', zero_division=0)
        macro_rec = recall_score(all_labels, y_bin, average='macro', zero_division=0)
        micro_rec = recall_score(all_labels, y_bin, average='micro', zero_division=0)

        try:
            macro_ap = average_precision_score(all_labels, all_probs, average='macro')
            micro_ap = average_precision_score(all_labels, all_probs, average='micro')
        except Exception:
            macro_ap, micro_ap = np.nan, np.nan

        try:
            macro_auc = roc_auc_score(all_labels, all_probs, average='macro')
        except Exception:
            macro_auc = np.nan

        composite = 0.5 * macro_f1 + 0.5 * (macro_auc if not np.isnan(macro_auc) else 0)

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


        # Save best
        if composite > best_score:
            best_score = composite
            best_epoch = epoch
            vision_state = backbones.vision.state_dict()
            vision_state_cpu = {k: v.cpu() for k, v in vision_state.items()}
            try:
                save_safetensors(vision_state_cpu, str(out_path))
                print(f"[INFO] Saved best swin checkpoint to {out_path} (epoch {epoch})")
            except Exception as e:
                torch.save(vision_state_cpu, str(out_path.with_suffix(".pth")))
                print(f"[WARN] safetensors save failed ({e}), saved torch .pth instead")

    print(f"[DONE] Best composite score: {best_score:.4f} at epoch {best_epoch}")
    print(f"[INFO] Final best checkpoint written to {out_path}")
