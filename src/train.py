import os
import json
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, precision_score, recall_score, accuracy_score
from transformers import get_cosine_schedule_with_warmup
from dataParser import parse_openi_xml
from model import MultiModalRetrievalModel
from dataLoader import build_dataloader
from torch.utils.data import WeightedRandomSampler
from labeledData import disease_groups, normal_groups
import wandb
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

# --- Config ---
EPOCHS = 2              # Number of training epochs : 30
PATIENCE = 10 
BATCH_SIZE = 1          # Batch size for training : 4
LR = 2e-5
USE_FOCAL = True  # Toggle between BCEWithLogits and FocalLoss
FUSION_TYPE = "cross"
JOINT_DIM = 256         # Dimensionality of the joint embedding 768                      

# --- Hyperparameters ---
temperature = 0.07
cls_weight   = 1.0                  # focuses on getting the labels right (1.0 is very focus on classification, 0.0 is very focus on contrastive learning)
cont_weight  = 0.5                  # focuses on pulling matching (image, text) embeddings closer in the joint space (1.0 is very focus on contrastive learning, 0.0 is very focus on classification)

# --- Wandb ---
project_name = "multimodal-disease-classification-2107"

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
XML_DIR = BASE_DIR / 'data' / 'openi' / 'xml' / 'NLMCXR_reports' / 'ecgen-radiology'
DICOM_ROOT = BASE_DIR / 'data' / 'openi' / 'dicom'
SPLIT_DIR = BASE_DIR / 'splited_data'
MODEL_DIR = BASE_DIR / 'models'
CHECKPOINT_DIR = BASE_DIR / 'checkpoints'
EMBED_SAVE_PATH = BASE_DIR / 'embeddings'
ATTN_DIR = BASE_DIR / 'attention_maps'
CSV_EVAL_SAVE_PATH = BASE_DIR / 'eval_csvs'
CSV_EVAL_SAVE_PATH.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
EMBED_SAVE_PATH.mkdir(exist_ok=True)
ATTN_DIR.mkdir(exist_ok=True)
os.environ['TRANSFORMERS_CACHE'] = str(MODEL_DIR)
label_cols = list(disease_groups.keys()) + list(normal_groups.keys())

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Setup ---
device = get_device()
if device.type == "cpu":
    raise RuntimeError("No GPU available, run on a CUDA device for better performance")

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
    all_labels, all_logits, all_ids, all_embs,all_attns = [], [], [], [], []

    with torch.no_grad():
        for batch in loader:
            imgs = batch['image'].cuda()
            ids = batch['input_ids'].cuda()
            mask = batch['attn_mask'].cuda()
            labels = batch['labels'].cpu().numpy()
            id_list = batch['id']

            joint_emb, logits, attn_weights = model(imgs, ids, mask, return_attention=True)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_labels.append(labels)
            all_logits.append(probs)
            all_ids.extend(id_list)
            all_embs.append(joint_emb.cpu())
            
            attn_cpu = {k: v.detach().cpu() for k, v in attn_weights.items()}
            all_attns.append(attn_cpu)

    y_true = np.vstack(all_labels)
    y_pred = np.vstack(all_logits)
    embeddings = torch.cat(all_embs).numpy()
    
    return y_true, y_pred, embeddings, all_ids, all_attns

def find_best_thresholds(y_true, y_logits):
    best_ts = []
    for i in range(y_true.shape[1]):
        p, r, t = precision_recall_curve(y_true[:, i], y_logits[:, i])
        f1 = 2 * p * r / (p + r + 1e-8)
        best_ts.append(t[f1.argmax()])
    return np.array(best_ts)

def check_label_consistency(records, labels_df, label_cols):
    mismatches = []

    for rec in records:
        rec_id = rec["id"]
        if rec_id not in labels_df.index:
            mismatches.append((rec_id, "Missing in labels_df"))
            continue
        
        df_vector = labels_df.loc[rec_id, label_cols].tolist()
        record_vector = rec["labels"]

        if list(map(int, df_vector)) != list(map(int, record_vector)):
            mismatches.append((rec_id, df_vector, record_vector))
    
    if mismatches:
        print(f"Found {len(mismatches)} mismatched records!")
        # Print the first few mismatches
        for i, item in enumerate(mismatches[:5]):
            print(f"\nMismatch #{i+1}")
            print("ID:", item[0])
            print("From labels_df:", item[1])
            print("From record:   ", item[2])
    else:
        print(" All label vectors match the CSV.")

    return mismatches

def df_to_records(df):
    return [
        {
            "id": row["id"],
            "report_text": row["report_text"],
            "labels": [row[f] for f in label_cols],
            "dicom_path": row["dicom_path"]
        }
        for _, row in df.iterrows()
    ]

# --- Main ---
if __name__ == '__main__':
    # --- Load records + split ---
    parsed_records = parse_openi_xml(XML_DIR, DICOM_ROOT)
    labels_df = pd.read_csv(BASE_DIR / "outputs" / "openi_labels_final.csv").set_index("id")
    label_cols = list(disease_groups.keys()) + list(normal_groups.keys())

    records = []
    for rec in parsed_records:
        rec_id = rec["id"]
        if rec_id in labels_df.index:
            label_vec = labels_df.loc[rec_id, label_cols].tolist()
            records.append({
                "id": rec["id"],
                "report_text": rec["report_text"],
                "dicom_path": rec["dicom_path"],
                "labels": label_vec
            })

    mismatches = check_label_consistency(records, labels_df, label_cols)
    if mismatches:
        raise Exception("Mismatched records found in records and labels_df")

    with open(SPLIT_DIR / "train_split_ids.json") as f:
        train_ids = set(json.load(f))
    with open(SPLIT_DIR / "val_split_ids.json") as f:
        val_ids = set(json.load(f))

    train_records = [r for r in records if r["id"] in train_ids]
    val_records   = [r for r in records if r["id"] in val_ids]

    print(f"Train size: {len(train_records)}, Val size: {len(val_records)}")

    # --- compute per‐label frequencies ---
    label_sums = np.zeros(len(label_cols), dtype=float)
    for r in train_records:
        label_sums += np.array(r['labels'], dtype=float)
    pos_freq = label_sums / len(train_records)

    # --- per‐label weight = 1 / freq (clipped) ---
    label_weights = 1.0 / np.clip(pos_freq, 1e-3, None)

    # --- per‐sample weights for sampler ---
    sample_weights = []
    for r in train_records:
        lv = np.array(r['labels'], dtype=float)
        sample_weights.append((lv * label_weights).sum())

    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # --- Dataloaders ---
    train_loader = build_dataloader(
        train_records,
        batch_size=BATCH_SIZE,
        mean=0.5, std=0.25,
        sampler=sampler
    )
    val_loader = build_dataloader(
        val_records,
        batch_size=BATCH_SIZE,
        mean=0.5, std=0.25
    )

    # --- Loss weights & criterion ---
    label_counts = np.array([r['labels'] for r in train_records]).sum(axis=0)
    pos_weight = 1.0 / torch.tensor(label_counts, dtype=torch.float32).clamp(min=1)
    pos_weight = pos_weight.cuda()

    if USE_FOCAL:
        # compute alpha for focal
        alpha = torch.tensor(label_weights, dtype=torch.float32)
        alpha = alpha / alpha.sum()
        alpha = alpha.to(device)
        criterion = FocalLoss(gamma=2, alpha=alpha)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --- Model, optimizer, scheduler ---
    model = MultiModalRetrievalModel(
        joint_dim=JOINT_DIM,
        num_classes=len(label_cols),
        fusion_type=FUSION_TYPE,
        swin_ckpt_path=MODEL_DIR / "swin_checkpoint.safetensors",
        bert_local_dir= MODEL_DIR / "clinicalbert_local",
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = EPOCHS * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),   # e.g. 10% warmup
        num_training_steps=total_steps
    )

    # --- wandb init ---
    wandb.init(project=project_name, config={
        "epochs": EPOCHS,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "loss": "focal" if USE_FOCAL else "BCEWithLogits",
        "fusion": FUSION_TYPE
    })
    wandb.define_metric("epoch")
    wandb.define_metric("val_precision", step_metric="epoch")
    wandb.define_metric("val_recall",    step_metric="epoch")
    wandb.define_metric("val_accuracy",  step_metric="epoch")

    for cn in label_cols:
        wandb.define_metric(f"val_auc_{cn}", step_metric="epoch")
        wandb.define_metric(f"val_f1_{cn}",  step_metric="epoch")
        wandb.define_metric(f"val_prec_{cn}",  step_metric="epoch")
        wandb.define_metric(f"val_rec_{cn}",   step_metric="epoch")

    
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
            joint_emb, logits, attn_weights = model(imgs, ids, mask, return_attention=False)

            # Classification loss
            cls_loss = criterion(logits, labels)

            # Contrastive loss (InfoNCE on the joint embedding)
            z_norm = torch.nn.functional.normalize(joint_emb, dim=1)       # (B, D)
            sim_matrix = torch.matmul(z_norm, z_norm.T) / temperature      # cosine‑similarity (B, B) 
            cont_targets = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
            cont_loss = torch.nn.functional.cross_entropy(sim_matrix, cont_targets)

            # Combined loss (InfoNCE term + chosen loss)
            loss = cls_weight * cls_loss + cont_weight * cont_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        # --- Validation & threshold tuning ---
        y_true, y_pred, val_embs, val_ids, val_attns = evaluate(model, val_loader)
        best_ts = find_best_thresholds(y_true, y_pred)
        y_bin = (y_pred > best_ts[None,:]).astype(int)
        
        # macro metrics (averaged over all classes)
        val_metrics = {
            'val_auc':       roc_auc_score(y_true, y_pred, average='macro'),
            'val_f1':        f1_score(y_true, y_bin, average='macro'),
            'val_precision': precision_score(y_true, y_bin, average='macro', zero_division=0),
            'val_recall':    recall_score(y_true, y_bin, average='macro', zero_division=0),
            'val_accuracy':  accuracy_score(y_true, y_bin),
            'epoch':         epoch+1
        }

        # per-class metrics
        class_aucs = roc_auc_score(y_true, y_pred, average=None)
        class_f1s  = f1_score(y_true, y_bin, average=None)
        class_precs= precision_score(y_true, y_bin, average=None, zero_division=0)
        class_recs = recall_score(y_true, y_bin, average=None, zero_division=0)
        for i, cn in enumerate(label_cols):
            val_metrics[f'val_auc_{cn}'] = class_aucs[i]
            val_metrics[f'val_f1_{cn}'] = class_f1s[i]
            val_metrics[f'val_prec_{cn}'] = class_precs[i]
            val_metrics[f'val_rec_{cn}'] = class_recs[i]

        wandb.log(val_metrics)

        # save CSV for EDA
        df_eval = pd.DataFrame({'id': val_ids})
        for i, cn in enumerate(label_cols):
            df_eval[f'true_{cn}'] = y_true[:,i]
            df_eval[f'pred_{cn}'] = y_pred[:,i]
        df_eval.to_csv(CSV_EVAL_SAVE_PATH/f"eval_epoch_{epoch}.csv", index=False)

        # --- Checkpointing & early stop ---
        torch.save(model.state_dict(), CHECKPOINT_DIR / f"model_epoch_{epoch+1}.pt")

        if val_metrics['val_auc'] > best_auc:
            best_auc = val_metrics['val_auc']
            patience_counter = 0
            torch.save(model.state_dict(), CHECKPOINT_DIR / "model_best.pt")
            np.save(EMBED_SAVE_PATH / "val_joint_embeddings.npy", val_embs)
            torch.save(val_attns, ATTN_DIR / "val_attn_weights.npy")
            with open(EMBED_SAVE_PATH / "val_ids.json", "w") as f:
                json.dump(val_ids, f)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    # Always save last val embeddings (even if not best)
    y_true, y_pred, val_embs, val_ids, val_attns = evaluate(model, val_loader)
    best_ts = find_best_thresholds(y_true, y_pred)
    y_bin = (y_pred > best_ts[None,:]).astype(int)

    val_metrics = {
            'val_auc':       roc_auc_score(y_true, y_pred, average='macro'),
            'val_f1':        f1_score(y_true, y_bin, average='macro'),
            'val_precision': precision_score(y_true, y_bin, average='macro', zero_division=0),
            'val_recall':    recall_score(y_true, y_bin, average='macro', zero_division=0),
            'val_accuracy':  accuracy_score(y_true, y_bin),
            'epoch':         epoch+1
        }

        # per-class metrics
    class_aucs = roc_auc_score(y_true, y_pred, average=None)
    class_f1s  = f1_score(y_true, y_bin, average=None)
    class_precs= precision_score(y_true, y_bin, average=None, zero_division=0)
    class_recs = recall_score(y_true, y_bin, average=None, zero_division=0)

    for i, cn in enumerate(label_cols):
        val_metrics[f'val_auc_{cn}'] = class_aucs[i]
        val_metrics[f'val_f1_{cn}'] = class_f1s[i]
        val_metrics[f'val_prec_{cn}'] = class_precs[i]
        val_metrics[f'val_rec_{cn}'] = class_recs[i]

    df_eval = pd.DataFrame({'id': val_ids})
    for i, cn in enumerate(label_cols):
        df_eval[f'true_{cn}'] = y_true[:, i]
        df_eval[f'pred_{cn}'] = y_pred[:, i]
    df_eval.to_csv(CSV_EVAL_SAVE_PATH / "eval_last.csv", index=False)
    np.save(EMBED_SAVE_PATH / "val_last_embeddings.npy", val_embs)
    torch.save(val_attns, ATTN_DIR / "val_last_attn_weights.npy")
    with open(EMBED_SAVE_PATH / "val_last_ids.json", "w") as f:
        json.dump(val_ids, f)

    print("Training complete.")
    print("Saving train joint embeddings...")
    # Load best model (ensure embeddings align with what val set saw)
    model.load_state_dict(torch.load(CHECKPOINT_DIR / "model_best.pt"))
    model.eval()
    y_true, y_pred, train_embs, train_ids, train_attns = evaluate(model, train_loader)
    np.save(EMBED_SAVE_PATH / "train_joint_embeddings.npy", train_embs)
    torch.save(train_attns, ATTN_DIR / "train_attn_weights.npy")

    with open(EMBED_SAVE_PATH / "train_ids.json", "w") as f:
        json.dump(train_ids, f)

    print("Done.")
