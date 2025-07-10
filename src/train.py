import os
import json
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score
from transformers import get_scheduler
from dataParser import parse_openi_xml
from model import MultiModalRetrievalModel
from dataLoader import build_dataloader
from dotenv import load_dotenv
load_dotenv()
import wandb

# --- Config ---
EPOCHS = 1
PATIENCE = 5
BATCH_SIZE = 1
LR = 2e-5
USE_FOCAL = False  # Toggle between BCEWithLogits and FocalLoss
FUSION_TYPE = "cross"

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
XML_DIR = BASE_DIR / 'data' / 'openi' / 'xml' / 'NLMCXR_reports' / 'ecgen-radiology'
DICOM_ROOT = BASE_DIR / 'data' / 'openi' / 'dicom'
SPLIT_DIR = BASE_DIR / 'splited_data'
MODEL_DIR = BASE_DIR / 'models'
CHECKPOINT_DIR = BASE_DIR / 'checkpoints'
EMBED_SAVE_PATH = BASE_DIR / 'embeddings'
CHECKPOINT_DIR.mkdir(exist_ok=True)
EMBED_SAVE_PATH.mkdir(exist_ok=True)
os.environ['TRANSFORMERS_CACHE'] = str(MODEL_DIR)

# --- Focal Loss Class ---
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        focal_weight = (1 - probs) ** self.gamma
        loss = focal_weight * bce
        if self.alpha is not None:
            loss = self.alpha * loss
        return loss.mean() if self.reduction == 'mean' else loss.sum()

# --- Evaluation ---
def evaluate(model, loader):
    model.eval()
    all_labels, all_logits, all_ids, all_embs = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            imgs = batch['image'].cuda()
            ids = batch['input_ids'].cuda()
            mask = batch['attn_mask'].cuda()

            labels = batch['labels'].cpu().numpy()
            id_list = batch['id']

            joint_emb, logits, _ = model(imgs, ids, mask, return_attention=False)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_labels.append(labels)
            all_logits.append(probs)
            all_ids.extend(id_list)
            all_embs.append(joint_emb.cpu())

    y_true = np.vstack(all_labels)
    y_pred = np.vstack(all_logits)
    emb_array = torch.cat(all_embs).numpy()

    macro_auc = roc_auc_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true > 0.5, y_pred > 0.5, average='macro')

    class_auc = roc_auc_score(y_true, y_pred, average=None)
    class_f1 = f1_score(y_true > 0.5, y_pred > 0.5, average=None)

    return macro_auc, macro_f1, class_auc, class_f1, emb_array, all_ids

# --- Main training---
if __name__ == '__main__':
    # --- Load records + split ---
    records = parse_openi_xml(XML_DIR, DICOM_ROOT)
    with open(SPLIT_DIR / "train_split_ids.json") as f:
        train_ids = set(json.load(f))
    with open(SPLIT_DIR / "val_split_ids.json") as f:
        val_ids = set(json.load(f))

    train_records = [r for r in records if r['id'] in train_ids]
    val_records   = [r for r in records if r['id'] in val_ids]

    # --- Dataloaders ---
    train_loader = build_dataloader(train_records, batch_size=BATCH_SIZE, mean=0.5, std=0.25)
    val_loader   = build_dataloader(val_records, batch_size=BATCH_SIZE, mean=0.5, std=0.25)

    # --- Class imbalance handling (pos_weight) ---
    label_counts = np.array([r['labels'] for r in train_records]).sum(axis=0)
    pos_weight = 1.0 / torch.tensor(label_counts, dtype=torch.float32).clamp(min=1)
    pos_weight = pos_weight.cuda()

    # --- Model ---
    model = MultiModalRetrievalModel(
        joint_dim=256,
        num_classes=14,
        fusion_type=FUSION_TYPE,
        swin_ckpt_path=MODEL_DIR / "swin_checkpoint.safetensors",
        bert_local_dir= MODEL_DIR / "clinicalbert_local",
    ).cuda()

    # --- Loss + Optimizer ---
    criterion = FocalLoss(gamma=2) if USE_FOCAL else nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = EPOCHS * len(train_loader)
    scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # --- wandb init ---
    wandb.init(project="multimodal-disease-classification", config={
        "epochs": EPOCHS,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "loss": "focal" if USE_FOCAL else "BCEWithLogits",
        "fusion": FUSION_TYPE
    })

    FINDINGS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
                "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule",
                "Pleural_Thickening", "Pneumonia", "Pneumothorax"]

    # --- Hyperparameters ---
    temperature = 0.07
    cls_weight   = 1.0                  # focuses on getting the labels right (1.0 is very focus on classification, 0.0 is very focus on contrastive learning)
    cont_weight  = 0.5                  # focuses on pulling matching (image, text) embeddings closer in the joint space (1.0 is very focus on contrastive learning, 0.0 is very focus on classification)

    # --- Early stopping ---
    best_f1 = 0
    best_auc = 0
    patience_counter = 0

    print("Starting training...")
    # --- Training Loop ---
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            imgs = batch['image'].cuda()
            ids = batch['input_ids'].cuda()
            mask = batch['attn_mask'].cuda()
            labels = batch['labels'].cuda()

            optimizer.zero_grad()

            # joint_emb: (B, joint_dim), logits: (B, num_classes)
            joint_emb, logits, _ = model(imgs, ids, mask, return_attention=False)

            # Classification loss
            cls_loss = criterion(logits, labels)

            # Contrastive loss (InfoNCE on the joint embedding)
            z_norm = torch.nn.functional.normalize(joint_emb, dim=1)       # (B, D)
            sim_matrix = torch.matmul(z_norm, z_norm.T) / temperature      # cosine‑similarity (B, B) 
            targets = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
            cont_loss = torch.nn.functional.cross_entropy(sim_matrix, targets)

            # Combined loss (InfoNCE term + chosen loss)
            loss = cls_weight * cls_loss + cont_weight * cont_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        val_auc, val_f1, class_auc, class_f1, val_embs, val_ids = evaluate(model, val_loader)

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": epoch_loss / len(train_loader),
            "val_auc": val_auc,
            "val_f1": val_f1,
        })

        for i, name in enumerate(FINDINGS):
            wandb.log({f"val_auc_{name}": class_auc[i], f"val_f1_{name}": class_f1[i]})

        print(f"Epoch {epoch+1} | Loss: {epoch_loss / len(train_loader):.4f} | Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f}")

        torch.save(model.state_dict(), CHECKPOINT_DIR / f"model_epoch{epoch+1}.pt")

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), CHECKPOINT_DIR / "model_best.pt")
            np.save(EMBED_SAVE_PATH / "val_joint_embeddings.npy", val_embs)
            with open(EMBED_SAVE_PATH / "val_ids.json", "w") as f:
                json.dump(val_ids, f)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    # Always save last val embeddings (even if not best)
    val_auc, val_f1, class_auc, class_f1, val_embs, val_ids = evaluate(model, val_loader)
    np.save(EMBED_SAVE_PATH / "val_last_embeddings.npy", val_embs)
    with open(EMBED_SAVE_PATH / "val_last_ids.json", "w") as f:
        json.dump(val_ids, f)

    print("Training complete.")
    print("→ Saving train joint embeddings...")
    # Load best model (ensure embeddings align with what val set saw)
    model.load_state_dict(torch.load(CHECKPOINT_DIR / "model_best.pt"))
    model.eval()
    train_auc, train_f1, _, _, train_embs, train_ids = evaluate(model, train_loader)
    np.save(EMBED_SAVE_PATH / "train_joint_embeddings.npy", train_embs)
    with open(EMBED_SAVE_PATH / "train_ids.json", "w") as f:
        json.dump(train_ids, f)

    wandb.log({
    "train_auc": train_auc,
    "train_f1": train_f1
    })
