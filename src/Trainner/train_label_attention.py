from pathlib import Path
import sys
try:
    base = Path(__file__).resolve().parent.parent
except NameError:
    base = Path.cwd().parent
sys.path.append(str(base))

import os
import wandb
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from KnowledgeGraph.label_attention import LabelAttention
import json
import pandas as pd
from DataHandler import parse_openi_xml
from DataHandler.TripletGenerate import PseudoTripletDataset, LabelEmbeddingLookup
from KnowledgeGraph.kg_label_create import ensure_label_embeddings
from LabelData import disease_groups, normal_groups, finding_groups, symptom_groups
from Helpers import Config
import numpy as np
from sklearn.metrics import average_precision_score
try:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
except NameError:
    BASE_DIR = Path.cwd().parent.parent
XML_DIR = BASE_DIR / 'data' / 'openi' / 'xml' / 'NLMCXR_reports' / 'ecgen-radiology'
DICOM_ROOT = BASE_DIR / 'data' / 'openi' / 'dicom'
SPLIT_DIR = BASE_DIR / 'splited_data'
CONFIG_DIR = BASE_DIR / 'configs'

# Base project dir (3 levels up)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
# Default model dir
MODEL_DIR = BASE_DIR / "label attention model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

class LabelAttentionWithTemp(LabelAttention):
    def __init__(self, d_emb, hidden=64):
        super().__init__(d_emb=d_emb, hidden=hidden)
        self.temperature = nn.Parameter(torch.tensor(0.07))

class PseudoPairDataset(torch.utils.data.Dataset):
    def __init__(self, df, min_overlap=0.5, max_overlap=0.2, mode="positive"):
        self.df = df
        self.samples = []
        self.mode = mode
        ids = list(df.index)
        for i, id1 in enumerate(ids):
            for j, id2 in enumerate(ids):
                if i != j:
                    labels1 = df.loc[id1].values
                    labels2 = df.loc[id2].values
                    overlap = (labels1 & labels2).sum() / (labels1 | labels2).sum()
                    if mode == "positive" and overlap >= min_overlap:
                        self.samples.append((id1, id2, 1))
                    elif mode == "negative" and overlap <= max_overlap:
                        self.samples.append((id1, id2, 0))

    def _generate_pairs(self):
        ids = list(self.df.index)
        for i, id1 in enumerate(ids):
            for j, id2 in enumerate(ids):
                if i != j:
                    # Overlap between label vectors
                    labels1 = self.df.loc[id1].values
                    labels2 = self.df.loc[id2].values
                    overlap = (labels1 & labels2).sum() / (labels1 | labels2).sum()
                    if overlap >= self.min_overlap:
                        self.samples.append((id1, id2))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]   # (qid, pid, label)

def info_nce_loss(q, p, temperature):
    q = F.normalize(q, dim=1)
    p = F.normalize(p, dim=1)
    logits = q @ p.T / temperature
    labels = torch.arange(q.size(0), device=q.device)
    return F.cross_entropy(logits, labels)

def contrastive_bce_loss(q, p, labels, temperature):
    q = F.normalize(q, dim=1)
    p = F.normalize(p, dim=1)
    sims = (q * p).sum(dim=1) / temperature
    return F.binary_cross_entropy_with_logits(sims, labels.float())

@torch.no_grad()
def evaluate_label_attention(model, label_lookup, df, device="cuda", topk=[1,5,10]):
    model.eval()
    ids = list(df.index)

    # precompute embeddings
    emb_dict = {}
    for rid in ids:
        emb = label_lookup.get_label_embs(rid).to(device)
        mask = torch.ones((emb.shape[0],), dtype=torch.bool, device=device)
        rep, _ = model(emb.unsqueeze(0), mask=mask.unsqueeze(0))
        emb_dict[rid] = rep.squeeze(0).cpu().numpy()

    all_embs = np.stack([emb_dict[r] for r in ids])
    norms = np.linalg.norm(all_embs, axis=1, keepdims=True)
    sims = all_embs @ all_embs.T / (norms @ norms.T)

    recall_at_k, ap_scores = {k: [] for k in topk}, []
    for i, qid in enumerate(ids):
        q_labels = df.loc[qid].values
        labels = np.array([(q_labels & df.loc[pid].values).sum() > 0 for pid in ids], dtype=int)
        labels[i] = 0  # exclude self

        sim_row = sims[i]
        idx = np.argsort(-sim_row)
        sorted_labels = labels[idx]

        for k in topk:
            recall_at_k[k].append(sorted_labels[:k].mean())
        ap_scores.append(average_precision_score(labels, sim_row))

    results = {f"recall@{k}": np.mean(v) for k, v in recall_at_k.items()}
    results["mAP"] = np.mean(ap_scores)

    print("\n[Eval] Retrieval performance:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
    return results

def train_label_attention(
    dataset,
    label_lookup,
    d_emb,
    hidden=64,
    batch_size=32,
    epochs=20,
    lr=1e-3,
    patience=3,
    save_path=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    log_to_wandb=False,
    val_dataset=None,
    loss_type="infonce",
):
    """
    Train a LabelAttention model with InfoNCE loss.

    Args:
        dataset (torch.utils.data.Dataset): Yields (qid, pid) pairs.
        label_lookup (LabelEmbeddingLookup): Lookup for label embeddings.
        d_emb (int): Embedding dimensionality.
        hidden (int): Hidden size of LabelAttention.
        batch_size (int): Training batch size.
        epochs (int): Number of epochs.
        lr (float): Learning rate.
        patience (int): Early stopping patience.
        save_path (str): Path to save model.
        device (str): Device ("cuda" or "cpu").
        log_to_wandb (bool): Log to W&B.
        val_dataset (torch.utils.data.Dataset): Validation dataset (same format as dataset).

    Returns:
        Trained LabelAttention model.
    """
    # default save path
    if save_path is None:
        BASE_DIR = Path(__file__).resolve().parent.parent.parent
        MODEL_DIR = BASE_DIR / "label attention model"
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        save_path = MODEL_DIR / "label_attention_model.pt"

    # init model + optimizer
    model = LabelAttentionWithTemp(d_emb=d_emb, hidden=hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_loss = float("inf")
    patience_counter = 0

    # training loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in tqdm(loader, desc=f"LabelAttention Epoch {epoch}/{epochs}"):
            if loss_type == "infonce":
                qids, pids = batch
                labels = None
            elif loss_type == "bce":
                qids, pids, labels = batch
                labels = labels.to(device)

            batch_q, batch_p, mask_q, mask_p = [], [], [], []

            # collect embeddings for batch
            for qid, pid in zip(qids, pids):
                def get_and_mask(rid):
                    emb = label_lookup.get_label_embs(rid)  # [n_labels, d]
                    mask = torch.ones((emb.shape[0],), dtype=torch.bool)
                    return emb, mask

                q_embs, q_mask = get_and_mask(qid)
                p_embs, p_mask = get_and_mask(pid)

                batch_q.append(q_embs); mask_q.append(q_mask)
                batch_p.append(p_embs); mask_p.append(p_mask)

            # pad sequences to max length
            max_len = max(x.shape[0] for x in batch_q + batch_p)

            def pad_stack(tensors, masks):
                emb_out, mask_out = [], []
                for t, m in zip(tensors, masks):
                    pad_len = max_len - t.shape[0]
                    if pad_len > 0:
                        pad_emb = torch.zeros((pad_len, t.shape[1]), device=device)
                        pad_mask = torch.zeros((pad_len,), dtype=torch.bool)
                        t = torch.cat([t, pad_emb], dim=0)
                        m = torch.cat([m, pad_mask], dim=0)
                    emb_out.append(t)
                    mask_out.append(m)
                return torch.stack(emb_out).to(device), torch.stack(mask_out).to(device)

            q_batch, q_mask_batch = pad_stack(batch_q, mask_q)
            p_batch, p_mask_batch = pad_stack(batch_p, mask_p)

            # forward + loss
            optimizer.zero_grad()
            q_rep, _ = model(q_batch, mask=q_mask_batch)
            p_rep, _ = model(p_batch, mask=p_mask_batch)

            if loss_type == "infonce":
                loss = info_nce_loss(q_rep, p_rep, model.temperature)
            elif loss_type == "bce":
                loss = contrastive_bce_loss(q_rep, p_rep, labels, model.temperature)

            # backward
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}/{epochs}  Loss={avg_loss:.4f}")

        # -----------------
        # validation
        # -----------------
        if val_dataset is not None:
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    if loss_type == "infonce":
                        qids, pids = batch
                        labels = None
                    elif loss_type == "bce":
                        qids, pids, labels = batch
                        labels = labels.to(device)

                    batch_q, batch_p, mask_q, mask_p = [], [], [], []
                    for qid, pid in zip(qids, pids):
                        def get_and_mask(rid):
                            emb = label_lookup.get_label_embs(rid)
                            mask = torch.ones((emb.shape[0],), dtype=torch.bool)
                            return emb, mask

                        q_embs, q_mask = get_and_mask(qid)
                        p_embs, p_mask = get_and_mask(pid)
                        batch_q.append(q_embs); mask_q.append(q_mask)
                        batch_p.append(p_embs); mask_p.append(p_mask)

                    max_len = max(x.shape[0] for x in batch_q + batch_p)
                    q_batch, q_mask_batch = pad_stack(batch_q, mask_q)
                    p_batch, p_mask_batch = pad_stack(batch_p, mask_p)

                    q_rep, _ = model(q_batch, mask=q_mask_batch)
                    p_rep, _ = model(p_batch, mask=p_mask_batch)

                    if loss_type == "infonce":
                        loss = info_nce_loss(q_rep, p_rep, model.temperature)
                    elif loss_type == "bce":
                        loss = contrastive_bce_loss(q_rep, p_rep, labels, model.temperature)

                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            print(f"Validation Loss={avg_val_loss:.4f}")
        else:
            avg_val_loss = avg_loss

        # -----------------
        # logging
        # -----------------
        if log_to_wandb:
            wandb.log({
                "la/train_loss": avg_loss,
                "la/val_loss": avg_val_loss if val_dataset else None,
                "epoch": epoch
            })

        # -----------------
        # early stopping
        # -----------------
        monitor_loss = avg_val_loss
        if monitor_loss < best_loss - 1e-3:
            best_loss = monitor_loss
            patience_counter = 0
            torch.save({
                "model_state": model.state_dict(),
                "config": {
                    "d_emb": d_emb,
                    "hidden": hidden,
                    "epochs": epochs,
                    "lr": lr,
                    "batch_size": batch_size
                }
            }, save_path)
            print(f"  [*] New best model (loss={best_loss:.4f}). Saved to {save_path}")
            if log_to_wandb:
                wandb.log({"la/best_loss": best_loss, "la/epoch": epoch})
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("  [!] Early stopping triggered.")
                if log_to_wandb:
                    wandb.log({"la/early_stop_epoch": epoch})
                break

    # load best model
    if os.path.exists(save_path):
        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])

    return model

if __name__ == "__main__":
    combined_groups = {**disease_groups, **normal_groups, **finding_groups, **symptom_groups}
    cfg = Config.load(CONFIG_DIR / 'config.yaml')

    wandb.init(
        project="LabelAttention-Training",
        name=f"la_hidden{cfg.la_hidden_dim}_lr{cfg.la_lr}",
        config={
            "epochs": cfg.la_epochs,
            "batch_size": cfg.la_batch_size,
            "lr": cfg.la_lr,
            "patience": cfg.la_patience,
            "hidden_dim": cfg.la_hidden_dim,
        },
    )

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

    with open(SPLIT_DIR / "train_split_ids.json") as f:
        train_ids = set(json.load(f))

    with open(SPLIT_DIR / "val_split_ids.json") as f:
        val_ids = set(json.load(f))

    train_records = [r for r in records if r["id"] in train_ids]
    val_records   = [r for r in records if r["id"] in val_ids]

    report_ids = [r["id"] for r in train_records]
    MODEL_LA_DIR = BASE_DIR / "label attention model"

    if not (MODEL_LA_DIR / "label_attention_model.pt").exists():
        print("Training LabelAttention pooling...")
        pseudo_dataset_train = PseudoPairDataset(labels_df.loc[list(train_ids)], min_overlap=0.5)
        pseudo_dataset_val   = PseudoPairDataset(labels_df.loc[list(val_ids)],   min_overlap=0.5)

        try:
            print("Loading label embeddings...")
            label_emb_dict = ensure_label_embeddings(BASE_DIR)
            print("Done.")
        except Exception as e:
            print("[ERROR] ensure_label_embeddings failed:", e)
            import traceback; traceback.print_exc()
            sys.exit(1)
        
        label_lookup = LabelEmbeddingLookup(labels_df, label_emb_dict, device="cuda")

        emb_sample = label_lookup.get_label_embs(report_ids[0])
        d_emb_actual = emb_sample.shape[1]

        # Train
        model_attn = train_label_attention(
            pseudo_dataset_train,
            label_lookup,
            d_emb=d_emb_actual,
            hidden=cfg.la_hidden_dim,
            batch_size=cfg.la_batch_size,
            epochs=cfg.la_epochs,
            lr=cfg.la_lr,
            patience=cfg.la_patience,
            val_dataset=pseudo_dataset_val,
            log_to_wandb=True,
        )
    else:
        print("Using cached LabelAttention model")