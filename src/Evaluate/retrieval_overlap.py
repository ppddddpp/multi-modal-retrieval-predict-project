from pathlib import Path
import sys
try:
    base = Path(__file__).resolve().parent.parent
except NameError:
    base = Path.cwd().parent
sys.path.append(str(base))

import json
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve
from sklearn.metrics.pairwise import cosine_similarity

from Helpers import Config
from Model import MultiModalRetrievalModel
from DataHandler import parse_openi_xml, build_dataloader
from LabelData import disease_groups, normal_groups, finding_groups, symptom_groups
from Retrieval.reranker import Reranker
from DataHandler.TripletGenerate import LabelEmbeddingLookup
from KnowledgeGraph.kg_label_create import ensure_label_embeddings
from KnowledgeGraph.label_attention import LabelAttention

# --- Paths & constants ---
BASE_DIR    = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = BASE_DIR / "configs" / "config.yaml"
SPLIT_DIR   = BASE_DIR / "splited_data"
XML_DIR     = BASE_DIR / "data" / "openi" / "xml" / "NLMCXR_reports" / "ecgen-radiology"
DICOM_ROOT  = BASE_DIR / "data" / "openi" / "dicom"
MODEL_DIR   = BASE_DIR / "models"
CKPT_PATH   = BASE_DIR / "checkpoints" / "model_best.pt"
LABELS_CSV  = BASE_DIR / "outputs" / "openi_labels_final.csv"
KG_DIR      = BASE_DIR / "knowledge_graph"
LA_MODEL_PATH = BASE_DIR / "label attention model" / "label_attention_model.pt"
SAVE_DIR    = BASE_DIR / "eval_with_kg_la"
SAVE_DIR.mkdir(exist_ok=True)

def _find_best_thresholds(y_true, y_logits):
    best_ts = []
    for i in range(y_true.shape[1]):
        p, r, t = precision_recall_curve(y_true[:, i], y_logits[:, i])
        if len(t) == 0:
            best_ts.append(0.5)
            continue
        f1 = 2 * p * r / (p + r + 1e-8)
        best_ts.append(float(t[f1.argmax()]))
    return np.array(best_ts)

def cos_sim(a, b):
    if a is None or b is None:
        return float("nan")
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    if a.shape[0] != b.shape[0]:
        return float("nan")
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))

def evaluate_collect_embeddings(model, loader, device):
    model.eval()
    ids, logits, labels, embs = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Running inference"):
            img = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attn_mask"].to(device)
            outputs = model(img, input_ids, attn_mask, return_attention=True)

            logits.append(outputs["logits"].cpu().numpy())
            embs.append(outputs["joint_emb"].cpu().numpy())
            labels.append(batch["labels"].cpu().numpy())
            ids.extend([str(x) for x in batch.get("id", [])])

    return (np.vstack(labels),
            np.vstack(logits),
            np.vstack(embs),
            ids)

def compute_ranking_metrics(query_embs, gallery_embs, query_labels, gallery_labels, k=1):
    sim = cosine_similarity(query_embs, gallery_embs)
    reciprocal_ranks = []
    hits = 0
    recalls = []
    for i in range(len(query_embs)):
        idxs = np.argsort(sim[i])[::-1]  # gallery indices sorted by similarity
        q_labels = set(np.where(query_labels[i] == 1)[0])
        # find rank of first relevant item
        rank = None
        for r, j in enumerate(idxs, start=1):
            g_labels = set(np.where(gallery_labels[j] == 1)[0])
            if len(q_labels & g_labels) > 0:
                rank = r
                break
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        if rank and rank <= k:
            hits += 1
        # compute recall@k
        topk = idxs[:k]
        relevant_topk = 0
        total_relevant = 0
        for j in range(len(gallery_embs)):
            g_labels = set(np.where(gallery_labels[j] == 1)[0])
            if len(q_labels & g_labels) > 0:
                total_relevant += 1
                if j in topk:
                    relevant_topk += 1
        recalls.append(relevant_topk / total_relevant if total_relevant > 0 else 0.0)
    return (np.mean(reciprocal_ranks),
            hits / len(query_embs),
            np.mean(recalls))

def main(query_split="val", gallery_split="test"):
    print(f"[INFO] Starting eval_with_kg_la with query={query_split}, gallery={gallery_split}")

    cfg = Config.load(CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # labels
    combined_groups = {**disease_groups, **finding_groups, **symptom_groups, **normal_groups}
    label_cols = sorted(list(combined_groups.keys()))

    # parse & label dataframe
    parsed_records = parse_openi_xml(XML_DIR, DICOM_ROOT, combined_groups=combined_groups)
    labels_df = pd.read_csv(LABELS_CSV).set_index("id")
    labels_df.index = labels_df.index.astype(str)

    def load_records_for_split(split_name):
        split_file = SPLIT_DIR / f"{split_name}_split_ids.json"
        with open(split_file, "r") as f:
            split_ids = set(json.load(f))
        recs = []
        for rec in parsed_records:
            rec_id = str(rec["id"])
            if rec_id in labels_df.index and rec_id in split_ids:
                label_vec = labels_df.loc[rec_id, label_cols].astype(int).tolist()
                recs.append({
                    "id": rec_id,
                    "report_text": rec["report_text"],
                    "dicom_path": rec["dicom_path"],
                    "labels": label_vec
                })
        return recs

    query_records = load_records_for_split(query_split)
    gallery_records = load_records_for_split(gallery_split)

    query_loader = build_dataloader(records=query_records,
                                   batch_size=cfg.batch_size,
                                   mean=0.5, std=0.25, shuffle=False)
    gallery_loader = build_dataloader(records=gallery_records,
                                     batch_size=cfg.batch_size,
                                     mean=0.5, std=0.25, shuffle=False)

    # model
    model = MultiModalRetrievalModel(
        joint_dim=cfg.joint_dim,
        num_heads=cfg.num_heads,
        num_fusion_layers=cfg.num_fusion_layers,
        num_classes=len(label_cols),
        fusion_type=cfg.fusion_type,
        swin_ckpt_path=MODEL_DIR / "swin_checkpoint.safetensors",
        bert_local_dir=MODEL_DIR / "clinicalbert_local",
        checkpoint_path=str(CKPT_PATH),
        use_shared_ffn=cfg.use_shared_ffn,
        use_cls_only=cfg.use_cls_only,
        device=device,
        training=False
    ).to(device)

    if CKPT_PATH.exists():
        ckpt = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt)

    # embeddings
    y_true_q, y_prob_q, emb_q, ids_q = evaluate_collect_embeddings(model, query_loader, device)
    y_true_g, y_prob_g, emb_g, ids_g = evaluate_collect_embeddings(model, gallery_loader, device)

    # metrics for k=1
    mrr1, hit1, recall1 = compute_ranking_metrics(emb_q, emb_g, y_true_q, y_true_g, k=1)
    # metrics for k=5
    mrr5, hit5, recall5 = compute_ranking_metrics(emb_q, emb_g, y_true_q, y_true_g, k=5)

    print(f"[INFO] MRR@1: {mrr1:.4f}, Hit@1: {hit1:.4f}, Recall@1: {recall1:.4f}")
    print(f"[INFO] MRR@5: {mrr5:.4f}, Hit@5: {hit5:.4f}, Recall@5: {recall5:.4f}")

    summary = {
        "query_split": query_split,
        "gallery_split": gallery_split,
        "mrr@1": float(mrr1),
        "hit1": float(hit1),
        "recall1": float(recall1),
        "mrr@5": float(mrr5),
        "hit5": float(hit5),
        "recall5": float(recall5)
    }
    with open(SAVE_DIR / f"summary_{query_split}_to_{gallery_split}.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
