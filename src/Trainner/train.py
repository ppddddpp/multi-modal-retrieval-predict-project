from pathlib import Path
import sys
try:
    base = Path(__file__).resolve().parent.parent
except NameError:
    base = Path.cwd().parent
sys.path.append(str(base))
import os
import json
import torch
import numpy as np
import pandas as pd
import random
import datetime
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score, accuracy_score
from transformers import get_cosine_schedule_with_warmup
from DataHandler import parse_openi_xml, build_dataloader
from Model import MultiModalRetrievalModel
from torch.utils.data import WeightedRandomSampler
from LabelData import disease_groups, normal_groups, finding_groups, symptom_groups
from Helpers import kg_alignment_loss, contrastive_loss, Config, safe_roc_auc, safe_avg_precision
from DataHandler.TripletGenerate import PseudoTripletDataset, LabelEmbeddingLookup
from train_label_attention import train_label_attention
from KnowledgeGraph import KGBuilder, KGTrainer
from KnowledgeGraph.kg_label_create import ensure_label_embeddings
import wandb
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
XML_DIR = BASE_DIR / 'data' / 'openi' / 'xml' / 'NLMCXR_reports' / 'ecgen-radiology'
DICOM_ROOT = BASE_DIR / 'data' / 'openi' / 'dicom'
SPLIT_DIR = BASE_DIR / 'splited_data'
MODEL_DIR = BASE_DIR / 'models'
CHECKPOINT_DIR = BASE_DIR / 'checkpoints'
EMBED_SAVE_PATH = BASE_DIR / 'embeddings'
ATTN_DIR = BASE_DIR / 'attention_maps'
CSV_EVAL_SAVE_PATH = BASE_DIR / 'eval_csvs'
CONFIG_DIR = BASE_DIR / 'configs'
CSV_EVAL_SAVE_PATH.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
EMBED_SAVE_PATH.mkdir(exist_ok=True)
ATTN_DIR.mkdir(exist_ok=True)
os.environ['TRANSFORMERS_CACHE'] = str(MODEL_DIR)

# --- Config ---
cfg = Config.load(CONFIG_DIR / 'config.yaml')

torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)
random.seed(cfg.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.seed)

EPOCHS = cfg.epochs
PATIENCE = cfg.patience 
BATCH_SIZE = cfg.batch_size
LR = cfg.lr
USE_FOCAL = cfg.use_focal  # Toggle between BCEWithLogits and FocalLoss
USE_HYBRID = cfg.use_hybrid  # Toggle between BCEWithLogits + FocalLoss
FUSION_TYPE = cfg.fusion_type
JOINT_DIM = cfg.joint_dim
KG_MODE = cfg.kg_mode

# --- Loss parameters ---
gamma_FOCAL = cfg.gamma_focal  # Focal loss gamma parameter
FOCAL_RATIO = cfg.focal_ratio  # Ratio of focal loss in hybrid loss (if USE_HYBRID is True), BCE_RATIO = 1 - FOCAL_RATIO

# --- Hyperparameters ---
temperature = cfg.temperature                # temperature for contrastive loss
cls_weight   = cfg.cls_weight                  # focuses on getting the labels right (1.0 is very focus on classification, 0.0 is very focus on contrastive learning)
cont_weight  = cfg.cont_weight                  # focuses on pulling matching (image, text) embeddings closer in the joint space (1.0 is very focus on contrastive learning, 0.0 is very focus on classification)
weight_img_joint = cfg.weight_img_joint
weight_text_joint = cfg.weight_text_joint

kg_weight = cfg.kg_weight
kg_emb_dim = cfg.kg_emb_dim
kg_epochs = cfg.kg_epochs
if cfg.kg_method in ["cosine", "mse"]:
    kg_method = cfg.kg_method
else:
    raise RuntimeError(f"KG method {cfg.kg_method} not supported")

num_heads = cfg.num_heads                     # number of attention heads in the fusion model
num_fusion_layers= cfg.num_fusion_layers
use_shared_ffn = cfg.use_shared_ffn
text_dim = cfg.text_dim
use_cls_only = cfg.use_cls_only
image_backbone = cfg.image_backbone

# --- Wandb ---
project_name = cfg.project_name
run_name = cfg.run_name

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
        probs = torch.sigmoid(logits).clamp(min=1e-6, max=1-1e-6)
        focal_weight = (1 - probs).clamp(min=1e-6) ** self.gamma
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

            outputs      = model(imgs, ids, mask, return_attention=True)
            joint_emb    = outputs["joint_emb"]
            img_emb      = outputs["img_emb"]
            txt_emb      = outputs["txt_emb"]
            logits       = outputs["logits"]
            attn_weights = outputs["attn"]

            probs = torch.sigmoid(logits).cpu().numpy()

            all_labels.append(labels)
            all_logits.append(probs)
            all_ids.extend(id_list)
            all_embs.append(joint_emb.cpu())
            
            if attn_weights is not None:
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
        # --- wandb init ---
    wandb.init(
        project=project_name,
        name=run_name,
        config={
            # Training
            "epochs": cfg.epochs,
            "patience": cfg.patience,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "seed": cfg.seed,

            # Model
            "num_fusion_layers": cfg.num_fusion_layers,
            "use_focal": cfg.use_focal,
            "use_hybrid": cfg.use_hybrid,
            "image_backbone": cfg.image_backbone,
            "fusion_type": cfg.fusion_type,
            "joint_dim": cfg.joint_dim,
            "num_heads": cfg.num_heads,
            "text_dim": cfg.text_dim,
            "use_shared_ffn": cfg.use_shared_ffn,
            "use_cls_only": cfg.use_cls_only,

            # Knowledge Graph
            "kg_model": cfg.kg_model,
            "kg_method": cfg.kg_method,
            "kg_emb_dim": cfg.kg_emb_dim,
            "kg_epochs": cfg.kg_epochs,
            "kg_mode": cfg.kg_mode,
            "kg_neg_size": cfg.kg_neg_size,
            "kg_adv_temp": cfg.kg_adv_temp,
            "kg_use_amp": cfg.kg_use_amp,
            "kg_lr": cfg.kg_lr,
            "kg_num_layers": cfg.kg_num_layers,
            "kg_dropout": cfg.kg_dropout,
            "kg_opn": cfg.kg_opn,

            # Loss weights
            "cls_weight": cfg.cls_weight,
            "cont_weight": cfg.cont_weight,
            "kg_weight": cfg.kg_weight,
            "weight_img_joint": cfg.weight_img_joint,
            "weight_text_joint": cfg.weight_text_joint,
            "gamma_focal": cfg.gamma_focal,
            "focal_ratio": cfg.focal_ratio,
            "temperature": cfg.temperature,

            # Label-Aware
            "la_hidden_dim": cfg.la_hidden_dim,
            "la_batch_size": cfg.la_batch_size,
            "la_epochs": cfg.la_epochs,
            "la_lr": cfg.la_lr,
            "la_patience": cfg.la_patience,
        }
    )

    kg_wandb_config = {
        "project": "multi-modal-kg", 
        "name": f"kg_{cfg.kg_model}_{cfg.kg_mode}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "config": {
            "kg_model": cfg.kg_model,
            "kg_method": cfg.kg_method,
            "kg_emb_dim": cfg.kg_emb_dim,
            "kg_epochs": cfg.kg_epochs,
            "kg_mode": cfg.kg_mode,
            "kg_neg_size": cfg.kg_neg_size,
            "kg_adv_temp": cfg.kg_adv_temp,
            "kg_use_amp": cfg.kg_use_amp,
            "kg_lr": cfg.kg_lr,
            "kg_num_layers": cfg.kg_num_layers,
            "kg_dropout": cfg.kg_dropout,
            "kg_opn": cfg.kg_opn,
            "seed": cfg.seed,
        },
    }

    # Used groups
    combined_groups = {**disease_groups, **normal_groups, **finding_groups, **symptom_groups}
    
    # --- Train KG ---
    kg_trainer = KGTrainer(
        kg_dir=BASE_DIR/"knowledge_graph",
        emb_dim=cfg.kg_emb_dim,
        joint_dim=cfg.joint_dim,
        model_name=cfg.kg_model,
        model_kwargs=cfg.kg_model_kwargs,
        lr=cfg.kg_lr,
    )
    kg_feats_path = BASE_DIR / "knowledge_graph" / "kg_image_feats.pt"
    if not (BASE_DIR/"knowledge_graph"/"node_embeddings.npy").exists():
        print("Training Knowledge Graph embeddings...")
        kg_builder = KGBuilder(out_dir=BASE_DIR/"knowledge_graph", combined_groups=combined_groups)
        
        KGBuilder.ensure_exists(
            xml_dir=XML_DIR,
            dicom_root=DICOM_ROOT,
            mode=KG_MODE,
            save_feats_path=str(kg_feats_path),
            backbone_type=image_backbone,
            device="cuda"
        )

        kg_trainer.load_triples(features_path=kg_feats_path)   # assumes triples.csv already built
        kg_trainer.train(epochs=cfg.kg_epochs, patience=cfg.patience,
                        wandb_config=kg_wandb_config, log_to_wandb=True, 
                        negative_size=cfg.kg_neg_size, 
                        advance_temp=cfg.kg_adv_temp, use_amp=cfg.kg_use_amp)
        
        kg_trainer.save_embeddings()
    else:
        print("Using cached KG embeddings")
    
    torch.cuda.empty_cache()
    # --- Load best KG embeddings ---
    kg_dir = BASE_DIR / "knowledge_graph"

    # prefer best checkpoint
    best_node_path = kg_dir / "node_embeddings_best.npy"
    best_rel_path  = kg_dir / "rel_embeddings_best.npy"

    if best_node_path.exists():
        node_emb_path = best_node_path
    else:
        # fallback to generic / epoch embeddings
        candidates = sorted(kg_dir.glob("node_embeddings_epoch*.npy"))
        if not candidates:
            candidates = sorted(kg_dir.glob("node_embeddings*.npy"))
        if not candidates:
            raise FileNotFoundError("No node_embeddings found in knowledge_graph/")
        node_emb_path = candidates[-1]

    print(f"[Train] Using KG embeddings from {node_emb_path}")
    kg_embs = np.load(node_emb_path)
    kg_embs = torch.tensor(kg_embs, dtype=torch.float32, device=device)

    # load node2id map
    with open(kg_dir / "node2id.json") as f:
        node2id = json.load(f)

    # --- Load records + split ---
    print("Loading records...")
    parsed_records = parse_openi_xml(XML_DIR, DICOM_ROOT, combined_groups=combined_groups)
    labels_df = pd.read_csv(BASE_DIR / "outputs" / "openi_labels_final.csv").set_index("id")
    label_cols = list(disease_groups.keys()) + list(normal_groups.keys()) + list(finding_groups.keys()) + list(symptom_groups.keys())
    
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
    pos_freq_np = label_sums / len(train_records)
    
    # --- per‐sample weights for sampler ---
    inv_freq = 1.0 / pos_freq_np.clip(min=1e-3)
    sample_weights = []
    for r in train_records:
        label_vec = np.array(r['labels'], dtype=float)
        if label_vec.sum() > 0:
            weight = (inv_freq * label_vec).max()
        else:
            weight = 1.0  # for all-negative samples
        sample_weights.append(weight)

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
        sampler=sampler,
        max_length=text_dim
    )

    val_loader = build_dataloader(
        val_records,
        batch_size=BATCH_SIZE,
        mean=0.5, std=0.25,
        max_length=text_dim
    )

    # --- Loss weights & criterion ---
    label_counts = np.array([r['labels'] for r in train_records]).sum(axis=0)
    num_samples  = len(train_records)   
    pos_weight = ((num_samples - torch.tensor(label_counts)) / torch.tensor(label_counts)).to(device)
    pos_weight = torch.clamp(pos_weight, min=1.0, max=cfg.pos_weight_clamp_max)  # Ensure no zero weights
    pos_weight = pos_weight.cuda()

    # Inverse frequency for Focal Loss
    pos_freq_t = torch.tensor(pos_freq_np, dtype=torch.float32).to(device)
    inv_freq = 1.0 / pos_freq_t.clamp(min=1e-3)  
    alpha    = (inv_freq / inv_freq.sum()).to(device)
    alpha = alpha.clamp(min=1e-4)

    if USE_FOCAL:
        alpha = alpha.to(device)
        criterion = FocalLoss(gamma=gamma_FOCAL, alpha=alpha)
    elif USE_HYBRID:
        # --- Hybrid BCE + Focal ---
        bce   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        alpha = torch.as_tensor(inv_freq, dtype=torch.float32, device=device)
        focal = FocalLoss(gamma=gamma_FOCAL, alpha=alpha)
        criterion = lambda logits, labels: (1 - FOCAL_RATIO) * bce(logits, labels) + FOCAL_RATIO * focal(logits, labels)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --- Model, optimizer, scheduler ---
    model = MultiModalRetrievalModel(
        joint_dim=JOINT_DIM,
        num_classes=len(label_cols),
        num_fusion_layers=num_fusion_layers,
        num_heads=num_heads,
        fusion_type=FUSION_TYPE,
        img_backbone=image_backbone,
        swin_ckpt_path=MODEL_DIR / "swin_checkpoint.safetensors",
        bert_local_dir= MODEL_DIR / "clinicalbert_local",
        device=device,
        use_shared_ffn=cfg.use_shared_ffn,
        use_cls_only=cfg.use_cls_only,
        training=True
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = EPOCHS * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),   # 10% warmup
        num_training_steps=total_steps
    )

    # Addtional metrics for W&B
    wandb.define_metric("epoch")
    wandb.define_metric("val_precision", step_metric="epoch")
    wandb.define_metric("val_recall",    step_metric="epoch")
    wandb.define_metric("val_accuracy",  step_metric="epoch")
    wandb.define_metric("kg/epoch")
    wandb.define_metric("kg/*", step_metric="kg/epoch")
    wandb.define_metric("kg/loss", step_metric="kg/epoch")
    wandb.define_metric("kg/val_mrr", step_metric="kg/epoch")
    wandb.define_metric("kg/val_hits1", step_metric="kg/epoch")
    wandb.define_metric("kg/val_hits10", step_metric="kg/epoch")

    for cn in label_cols:
        wandb.define_metric(f"val_auc_{cn}", step_metric="epoch")
        wandb.define_metric(f"val_f1_{cn}",  step_metric="epoch")
        wandb.define_metric(f"val_prec_{cn}",  step_metric="epoch")
        wandb.define_metric(f"val_rec_{cn}",   step_metric="epoch")

    labels = torch.tensor([r['labels'] for r in train_records], dtype=torch.float32).to(device)
    batch_labels = labels.mean(dim=0).cpu().numpy()
    wandb.log({f"batch_pos_freq/{cn}": float(batch_labels[i]) 
            for i, cn in enumerate(label_cols)}, step=0)

    # --- Early stopping ---
    best_score = -float("inf")
    patience_counter = 0
    
    print("Starting training...")
    # --- Training Loop ---
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            imgs = batch['image'].to(device)
            ids = batch['input_ids'].to(device)
            mask = batch['attn_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            # joint_emb: (B, joint_dim), logits: (B, num_classes)
            outputs      = model(imgs, ids, mask, return_attention=True)
            joint_emb    = outputs["joint_emb"]
            img_emb      = outputs["img_emb"]
            txt_emb      = outputs["txt_emb"]
            logits       = outputs["logits"]
            attn_weights = outputs["attn"]

            # Classification loss
            cls_loss = criterion(logits, labels)

            # Contrastive loss
            loss_img_txt   = contrastive_loss(img_emb, txt_emb, temperature)
            loss_img_joint = contrastive_loss(img_emb, joint_emb, temperature)
            loss_txt_joint = contrastive_loss(txt_emb, joint_emb, temperature)

            cont_loss = loss_img_txt + weight_img_joint * loss_img_joint + weight_text_joint * loss_txt_joint
            
            kg_loss = kg_alignment_loss(
                joint_emb,
                batch['id'],
                kg_embs,
                node2id,
                trainer=kg_trainer,
                labels=batch['labels'],
                label_cols=label_cols,
                loss_type=cfg.kg_method
            )

            # Combined loss (InfoNCE term + chosen loss)
            loss = cls_weight * cls_loss + cont_weight * cont_loss + kg_weight * kg_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            wandb.log({
                "batch_train_loss": loss.item(),
                "cls_loss": cls_loss.item(),
                "cont_loss": cont_loss.item(),
                "kg_loss": kg_loss.item(),
                "epoch": epoch + 1,
                "batch_idx": batch_idx
            })

        # --- Validation & threshold tuning ---
        avg_train_loss = epoch_loss / len(train_loader)
        wandb.log({
            "train_loss": avg_train_loss,
            "epoch": epoch + 1
        })

        y_true, y_pred, val_embs, val_ids, val_attns = evaluate(model, val_loader)
        best_ts = find_best_thresholds(y_true, y_pred)
        y_bin = (y_pred > best_ts[None, :]).astype(int)

        for i, cn in enumerate(label_cols):
            wandb.log({f"thresh_{cn}": float(best_ts[i]), "epoch": epoch+1})
        
        # macro metrics (averaged over all classes)
        # per-class AUROC (nan if invalid)
        class_aucs = safe_roc_auc(y_true, y_pred, label_names=label_cols)   # shape (C,)

        # per-class Average Precision (PR-AUC), returns NaN for invalid classes
        class_aps = safe_avg_precision(y_true, y_pred, label_names=label_cols)  # shape (C,)

        # thresholds already computed earlier as `best_ts` (if not, use 0.5 during training)
        y_bin = (y_pred > best_ts[None, :]).astype(int)

        # per-class F1 / precision / recall (these functions handle zero_division)
        class_f1s   = f1_score(y_true, y_bin, average=None, zero_division=0)
        class_precs = precision_score(y_true, y_bin, average=None, zero_division=0)
        class_recs  = recall_score(y_true, y_bin, average=None, zero_division=0)

        # aggregated metrics
        # macro AUC = mean over valid classes (ignore NaNs)
        macro_auc = float(np.nanmean(class_aucs))
        macro_ap  = float(np.nanmean(class_aps))

        # micro PR-AUC (global): wrap in try/except because it can error when no positives at all
        try:
            micro_ap = float(average_precision_score(y_true, y_pred, average='micro'))
        except Exception:
            micro_ap = float("nan")

        # micro F1 / precision / recall (global counts)
        micro_f1   = float(f1_score(y_true, y_bin, average='micro', zero_division=0))
        micro_prec = float(precision_score(y_true, y_bin, average='micro', zero_division=0))
        micro_rec  = float(recall_score(y_true, y_bin, average='micro', zero_division=0))

        # macro F1/prec/rec (still useful)
        macro_f1   = float(f1_score(y_true, y_bin, average='macro'))
        macro_prec = float(precision_score(y_true, y_bin, average='macro', zero_division=0))
        macro_rec  = float(recall_score(y_true, y_bin, average='macro', zero_division=0))

        print(f"Epoch {epoch} | Val AUC: {macro_auc:.4f} | Val AP: {macro_ap:.4f}" 
                f" | Val F1: {macro_f1:.4f} | Val Prec: {macro_prec:.4f} | Val Rec: {macro_rec:.4f}" 
                f" | Val Micro AP: {micro_ap:.4f} | Val Micro F1: {micro_f1:.4f}" 
                f" | Val Micro Prec: {micro_prec:.4f} | Val Micro Rec: {micro_rec:.4f}" 
                f" | Train Loss: {avg_train_loss:.4f} | Val Accuracy: {float(accuracy_score(y_true, y_bin)):.4f}" 
            )

        val_metrics = {
            "val_auc_macro": macro_auc,
            "val_ap_macro": macro_ap,
            "val_ap_micro": micro_ap,
            "val_f1_macro": macro_f1,
            "val_f1_micro": micro_f1,
            "val_prec_macro": macro_prec,
            "val_prec_micro": micro_prec,
            "val_rec_macro": macro_rec,
            "val_rec_micro": micro_rec,
            "val_accuracy": float(accuracy_score(y_true, y_bin)),
            "epoch": epoch + 1
        }
        wandb.log(val_metrics)

        # save CSV for EDA
        df_eval = pd.DataFrame({'id': val_ids})
        for i, cn in enumerate(label_cols):
            df_eval[f'true_{cn}'] = y_true[:,i]
            df_eval[f'pred_{cn}'] = y_pred[:,i]
        df_eval.to_csv(CSV_EVAL_SAVE_PATH/f"eval_epoch_{epoch}.csv", index=False)

        # --- Checkpointing & early stop ---
        torch.save(model.state_dict(), CHECKPOINT_DIR / f"model_epoch_{epoch+1}.pt")

        # --- Early stopping ---
        composite = 0.5 * macro_f1 + 0.5 * macro_auc
        wandb.log({"val_composite": composite, "epoch": epoch + 1})
        if composite > best_score:
            best_score = composite
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

    all_class_aucs = safe_roc_auc(y_true, y_pred)
    val_metrics = {
            'val_auc':       np.nanmean(all_class_aucs),
            'val_f1':        f1_score(y_true, y_bin, average='macro'),
            'val_precision': precision_score(y_true, y_bin, average='macro', zero_division=0),
            'val_recall':    recall_score(y_true, y_bin, average='macro', zero_division=0),
            'val_accuracy':  accuracy_score(y_true, y_bin),
            'epoch':         epoch+1
        }

        # per-class metrics
    class_aucs = safe_roc_auc(y_true, y_pred)
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

    # Load best model and save train embeddings
    model.load_state_dict(torch.load(CHECKPOINT_DIR / "model_best.pt"))
    model.eval()
    
    y_true, y_pred, train_embs, train_ids, train_attns = evaluate(model, train_loader)
    np.save(EMBED_SAVE_PATH / "train_joint_embeddings.npy", train_embs)
    torch.save(train_attns, ATTN_DIR / "train_attn_weights.npy")

    with open(EMBED_SAVE_PATH / "train_ids.json", "w") as f:
        json.dump(train_ids, f)

    print("Done.")
