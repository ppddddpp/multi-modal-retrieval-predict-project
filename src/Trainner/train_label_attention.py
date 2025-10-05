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
from KnowledgeGraph.label_attention import LabelAttention
import json
import pandas as pd
from DataHandler import parse_openi_xml
from DataHandler.TripletGenerate import PseudoTripletDataset, LabelEmbeddingLookup
from KnowledgeGraph.kg_label_create import ensure_label_embeddings
from LabelData import disease_groups, normal_groups, finding_groups, symptom_groups
from Helpers import Config
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
):
    """
    Train a LabelAttention model.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to train on.
        label_lookup (LabelEmbeddingLookup): The label lookup object.
        d_emb (int): The dimensionality of the embeddings.
        hidden (int, optional): The number of hidden units in the model. Defaults to 64.
        batch_size (int, optional): The batch size. Defaults to 32.
        epochs (int, optional): The number of epochs to train for. Defaults to 20.
        lr (float, optional): The learning rate. Defaults to 1e-3.
        patience (int, optional): The patience for early stopping. Defaults to 3.
        save_path (str, optional): The path to save the model to. Defaults to None, which means the model is saved to MODEL_DIR / "label_attention_model.pt".
        device (str, optional): The device to use. Defaults to "cuda" if available, otherwise "cpu".
        log_to_wandb (bool, optional): Whether to log to Weights and Biases. Defaults to False.
        val_dataset (torch.utils.data.Dataset, optional): The validation dataset. Defaults to None.

    Returns:
        LabelAttention: The trained model.
    """
    if save_path is None:
        save_path = MODEL_DIR / "label_attention_model.pt"

    model = LabelAttention(d_emb=d_emb, hidden=hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_loss = float("inf")
    patience_counter = 0

    model.train()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"LabelAttention Epoch {epoch}/{epochs}"):
            qids, pids, nids = batch
            batch_q, batch_p, batch_n = [], [], []
            mask_q, mask_p, mask_n = [], [], []

            for qid, pid, nid in zip(qids, pids, nids):
                def get_and_mask(rid):
                    emb = label_lookup.get_label_embs(rid)  # [n_labels, d]
                    mask = torch.ones((emb.shape[0],), dtype=torch.bool)
                    return emb, mask
                q_embs, q_mask = get_and_mask(qid)
                p_embs, p_mask = get_and_mask(pid)
                n_embs, n_mask = get_and_mask(nid)
                batch_q.append(q_embs); mask_q.append(q_mask)
                batch_p.append(p_embs); mask_p.append(p_mask)
                batch_n.append(n_embs); mask_n.append(n_mask)

            max_len = max(x.shape[0] for x in batch_q + batch_p + batch_n)

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
            n_batch, n_mask_batch = pad_stack(batch_n, mask_n)

            optimizer.zero_grad()
            q_rep, _ = model(q_batch, mask=q_mask_batch)
            p_rep, _ = model(p_batch, mask=p_mask_batch)
            n_rep, _ = model(n_batch, mask=n_mask_batch)

            loss = triplet_loss(q_rep, p_rep, n_rep)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}/{epochs}  Loss={avg_loss:.4f}")

        if val_dataset is not None:
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for qids, pids, nids in val_loader:
                    batch_q, batch_p, batch_n = [], [], []
                    mask_q, mask_p, mask_n = [], [], []

                    for qid, pid, nid in zip(qids, pids, nids):
                        def get_and_mask(rid):
                            emb = label_lookup.get_label_embs(rid)
                            mask = torch.ones((emb.shape[0],), dtype=torch.bool)
                            return emb, mask
                        q_embs, q_mask = get_and_mask(qid)
                        p_embs, p_mask = get_and_mask(pid)
                        n_embs, n_mask = get_and_mask(nid)
                        batch_q.append(q_embs); mask_q.append(q_mask)
                        batch_p.append(p_embs); mask_p.append(p_mask)
                        batch_n.append(n_embs); mask_n.append(n_mask)

                    max_len = max(x.shape[0] for x in batch_q + batch_p + batch_n)

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
                    n_batch, n_mask_batch = pad_stack(batch_n, mask_n)

                    q_rep, _ = model(q_batch, mask=q_mask_batch)
                    p_rep, _ = model(p_batch, mask=p_mask_batch)
                    n_rep, _ = model(n_batch, mask=n_mask_batch)

                    loss = triplet_loss(q_rep, p_rep, n_rep)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            print(f"Validation Loss={avg_val_loss:.4f}")
        
        monitor_loss = avg_val_loss if val_dataset is not None else avg_loss

        if log_to_wandb:
            wandb.log({
                "la/train_loss": avg_loss,
                "la/val_loss": avg_val_loss if val_dataset else None,
                "epoch": epoch
            })

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
        pseudo_dataset_train = PseudoTripletDataset(labels_df.loc[list(train_ids)], min_overlap=0.5)
        pseudo_dataset_val   = PseudoTripletDataset(labels_df.loc[list(val_ids)],   min_overlap=0.5)

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
            val_dataset=pseudo_dataset_val
        )
    else:
        print("Using cached LabelAttention model")