from pathlib import Path
import sys
try:
    base = Path(__file__).resolve().parent.parent
except NameError:
    base = Path.cwd().parent
sys.path.append(str(base))
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import warnings

# --- Show each unique warning only once ---
warnings.filterwarnings("once")

# --- Suppress repetitive / irrelevant library warnings ---
warnings.filterwarnings(
    "ignore",
    message=".*CUDA path could not be detected.*",
    category=UserWarning,
    module="cupy"
)
warnings.filterwarnings(
    "ignore",
    message=".*TRANSFORMERS_CACHE.*",
    category=FutureWarning,
    module="transformers"
)
warnings.filterwarnings(
    "ignore",
    message=".*clean_up_tokenization_spaces.*",
    category=FutureWarning,
    module="transformers"
)
# --- NEW: silence spaCy / weasel deprecation spam ---
warnings.filterwarnings(
    "ignore",
    message=".*Importing 'parser.split_arg_string' is deprecated.*",
    category=DeprecationWarning,
    module="spacy"
)
warnings.filterwarnings(
    "ignore",
    message=".*Importing 'parser.split_arg_string' is deprecated.*",
    category=DeprecationWarning,
    module="weasel"
)

import json
from typing import List, Dict, Any
import numpy as np
import pandas as pd

# ------------------------
# Simple JSON loader
# ------------------------
def load_json(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected JSON root to be a list.")
    return data

# ------------------------
# Flatten Q->R for CSV export
# ------------------------
def flatten_retrievals(data: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for q in data:
        qid = q.get("qid")
        qreport = q.get("query_report")
        # train retrievals
        for rank, r in enumerate(q.get("retrieval_train", []) or [], start=1):
            base = {
                "qid": qid,
                "query_report": qreport,
                "retrieval_db": "train",
                "rid": r.get("rid"),
                "dist": r.get("dist"),
                "rank": rank,
            }
            cm = r.get("compare_metrics") or {}
            for k, v in cm.items():
                try:
                    base[k] = float(v) if v is not None else np.nan
                except Exception:
                    base[k] = np.nan
            rows.append(base)
        # test retrievals
        for rank, r in enumerate(q.get("retrieval_test", []) or [], start=1):
            base = {
                "qid": qid,
                "query_report": qreport,
                "retrieval_db": "test",
                "rid": r.get("rid"),
                "dist": r.get("dist"),
                "rank": rank,
            }
            cm = r.get("compare_metrics") or {}
            for k, v in cm.items():
                try:
                    base[k] = float(v) if v is not None else np.nan
                except Exception:
                    base[k] = np.nan
            rows.append(base)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df

# ------------------------
# Retrieval-level flatten (R->R)
# ------------------------
def flatten_rr(data: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for q in data:
        qid = q.get("qid")
        row = {"qid": qid}
        # copy keys we care about if present
        for key in [
            "retrieval_diversity_train", "retrieval_diversity_test",
            "retrieval_label_diversity_train", "retrieval_label_diversity_test"
        ]:
            if key in q:
                try:
                    row[key] = float(q[key]) if q[key] is not None else np.nan
                except Exception:
                    row[key] = np.nan
        if len(row) > 1:
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)

# ------------------------
# Utility: simple summary
# ------------------------
def summary_all_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    # metrics are any numeric columns besides id-like
    non_metric = {"qid", "rid", "rank", "retrieval_db", "query_report", "dist"}
    metrics = [c for c in df.columns if c not in non_metric]
    rows = []
    total = len(df)
    for m in metrics:
        s = pd.to_numeric(df[m], errors="coerce")
        cnt = int(s.count())
        rows.append({
            "metric": m,
            "mean": float(s.mean()) if cnt > 0 else np.nan,
            "std": float(s.std()) if cnt > 1 else np.nan,
            "median": float(s.median()) if cnt > 0 else np.nan,
            "min": float(s.min()) if cnt > 0 else np.nan,
            "max": float(s.max()) if cnt > 0 else np.nan,
            "count_nonnull": cnt,
            "total_rows": total,
            "pct_missing": 100.0 * (1.0 - cnt / total) if total > 0 else np.nan
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("metric").sort_index()

# ------------------------
# Diversity metrics
# ------------------------
def compute_embedding_diversity(embeddings: np.ndarray) -> float:
    """1 - mean_pairwise_cosine (range approx [0,2] but typically [0,1])"""
    if embeddings is None or len(embeddings) < 2:
        return 0.0
    # normalize rows
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-8)
    normed = embeddings / norms
    sim = np.dot(normed, normed.T)
    n = sim.shape[0]
    triu = np.triu_indices(n, k=1)
    mean_cos = float(np.mean(sim[triu]))
    return float(1.0 - mean_cos)

def compute_label_diversity_from_labels(labels_list: List[List[str]]) -> float:
    """unique labels / avg per-item label count (simple proxy)."""
    if not labels_list:
        return 0.0
    all_labels = set(l for lab in labels_list for l in lab)
    sizes = [len(lab) for lab in labels_list]
    sizes_nonzero = [s for s in sizes if s > 0]
    if not sizes_nonzero:
        return 0.0
    avg_size = float(np.mean(sizes_nonzero))
    return float(len(all_labels) / avg_size)

# ------------------------
# Core retrieval runner (dual-engine)
# ------------------------
def run_retrieval_experiment(k: int = 5) -> List[Dict[str, Any]]:
    import torch
    from tqdm import tqdm
    from Helpers import Config
    from Model import MultiModalRetrievalModel
    from Retrieval import make_retrieval_engine
    from DataHandler import parse_openi_xml, build_dataloader
    from LabelData import disease_groups, normal_groups, finding_groups, symptom_groups

    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    CONFIG_PATH = BASE_DIR / "configs" / "config.yaml"
    CKPT_PATH = BASE_DIR / "checkpoints" / "model_best.pt"
    MODEL_DIR = BASE_DIR / "models"
    EMBED_DIR = BASE_DIR / "embeddings"
    XML_DIR = BASE_DIR / "data" / "openi" / "xml" / "NLMCXR_reports" / "ecgen-radiology"
    DICOM_ROOT = BASE_DIR / "data" / "openi" / "dicom"
    LABELS_CSV = BASE_DIR / "outputs" / "openi_labels_final.csv"

    cfg = Config.load(CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    combined_groups = {**disease_groups, **normal_groups, **finding_groups, **symptom_groups}
    label_cols = list(combined_groups.keys())

    # make both engines: train and test
    engine_train = make_retrieval_engine(
        str(EMBED_DIR / "train_joint_embeddings.npy"),
        str(EMBED_DIR / "train_ids.json"),
        method="dls",
        link_threshold=0.5,
        max_links=10
    )
    engine_test = make_retrieval_engine(
        str(EMBED_DIR / "test_joint_embeddings.npy"),
        str(EMBED_DIR / "test_ids.json"),
        method="dls",
        link_threshold=0.5,
        max_links=10
    )

    # model (for producing joint embedding of query)
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
    model.eval()

    # parsed_records -> build dataloader
    parsed_records = parse_openi_xml(XML_DIR, DICOM_ROOT, combined_groups=combined_groups)
    test_loader = build_dataloader(records=parsed_records, batch_size=1, mean=0.5, std=0.25, shuffle=False)

    # load embedding banks (for computing embedding diversity)
    train_embs = np.load(EMBED_DIR / "train_joint_embeddings.npy")
    with open(EMBED_DIR / "train_ids.json") as f:
        train_ids = json.load(f)
    id_to_emb_train = {rid: train_embs[i] for i, rid in enumerate(train_ids)}

    test_embs = np.load(EMBED_DIR / "test_joint_embeddings.npy")
    with open(EMBED_DIR / "test_ids.json") as f:
        test_ids = json.load(f)
    id_to_emb_test = {rid: test_embs[i] for i, rid in enumerate(test_ids)}

    # try to load labels table for label diversity (optional)
    labels_df = None
    try:
        labels_df = pd.read_csv(LABELS_CSV).set_index("id")
        print(f"[INFO] Loaded labels dataframe with {len(labels_df)} rows.")
    except Exception:
        print(f"[WARN] Could not load labels CSV at {LABELS_CSV}. Label diversity will be estimated from provided label lists if available.")

    results = []
    for batch in tqdm(test_loader, desc="Running Retrieval"):
        qid = batch["id"][0]
        img = batch["image"].to(device)
        ids = batch["input_ids"].to(device)
        mask = batch["attn_mask"].to(device)

        with torch.no_grad():
            out = model(img, ids, mask, return_attention=False)
            q_vec = out["joint_emb"][0].cpu().numpy()

        # -- retrieve from train DB
        ret_train_ids, dists_train = engine_train.retrieve(q_vec, K=k)
        emb_train = np.array([id_to_emb_train[r] for r in ret_train_ids if r in id_to_emb_train])
        diversity_train = compute_embedding_diversity(emb_train)

        # collect labels for train retrievals (if labels_df exists)
        train_label_sets = []
        if labels_df is not None:
            for rid in ret_train_ids:
                if rid in labels_df.index:
                    row = labels_df.loc[rid]
                    labs = [
                        c for c, v in row.items()
                        if pd.notna(v)
                        and (str(v).strip() in ["1", "True", "true"])
                    ]
                    train_label_sets.append(labs)
                else:
                    train_label_sets.append([])
        label_diversity_train = compute_label_diversity_from_labels(train_label_sets)

        # -- retrieve from test DB
        ret_test_ids, dists_test = engine_test.retrieve(q_vec, K=k)
        emb_test = np.array([id_to_emb_test[r] for r in ret_test_ids if r in id_to_emb_test])
        diversity_test = compute_embedding_diversity(emb_test)

        test_label_sets = []
        if labels_df is not None:
            for rid in ret_test_ids:
                if rid in labels_df.index:
                    row = labels_df.loc[rid]
                    # Safer label detection
                    labs = [
                        c for c, v in row.items()
                        if pd.notna(v)
                        and (str(v).strip() in ["1", "True", "true"])
                    ]
                    test_label_sets.append(labs)
                else:
                    test_label_sets.append([])
        label_diversity_test = compute_label_diversity_from_labels(test_label_sets)

        # build retrieval lists for JSON (no heavy compare metrics here)
        retrieval_train = [{"rid": rid, "dist": float(d)} for rid, d in zip(ret_train_ids, dists_train)]
        retrieval_test = [{"rid": rid, "dist": float(d)} for rid, d in zip(ret_test_ids, dists_test)]

        results.append({
            "qid": qid,
            "query_report": batch.get("report_text", [None])[0] if isinstance(batch.get("report_text"), list) else None,
            "retrieval_train": retrieval_train,
            "retrieval_test": retrieval_test,
            "retrieval_diversity_train": float(diversity_train),
            "retrieval_diversity_test": float(diversity_test),
            "retrieval_label_diversity_train": float(label_diversity_train),
            "retrieval_label_diversity_test": float(label_diversity_test),
        })

    return results

# ------------------------
# Top-level runner (no argparse)
# ------------------------
def run_analysis(json_path: str, out_dir: str, run_retrieval: bool = False, k: int = 5):
    json_path = Path(json_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if run_retrieval:
        print("[INFO] Running retrieval experiment (dual engines)...")
        data = run_retrieval_experiment(k=k)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[INFO] Retrieval JSON saved to: {json_path}")
    else:
        print("[INFO] Loading retrieval JSON from disk...")
        data = load_json(str(json_path))

    # Q->R flatten
    df_qr = flatten_retrievals(data)
    if not df_qr.empty:
        qr_out = out_dir / "retrieval_metrics_flat.csv"
        df_qr.to_csv(qr_out, index=False)
        print(f"[INFO] Saved Q->R flattened CSV: {qr_out}")
        summary_qr = summary_all_metrics(df_qr)
        if not summary_qr.empty:
            summary_qr.to_csv(out_dir / "retrieval_metrics_summary.csv")
            print(f"[INFO] Saved Q->R metrics summary CSV")
    else:
        print("[WARN] No Q->R rows to save.")

    # R->R flatten
    df_rr = flatten_rr(data)
    if not df_rr.empty:
        rr_out = out_dir / "retrieval_retrieval_metrics.csv"
        df_rr.to_csv(rr_out, index=False)
        print(f"[INFO] Saved R->R CSV: {rr_out}")
        summary_rr = summary_all_metrics(df_rr)
        if not summary_rr.empty:
            summary_rr.to_csv(out_dir / "retrieval_retrieval_summary.csv")
            print(f"[INFO] Saved R->R metrics summary CSV")
    else:
        print("[INFO] No retrieval-level (R->R) metrics found in JSON.")

    # Save a copy of the full JSON report
    full_json_out = out_dir / json_path.name
    with open(full_json_out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[INFO] Full JSON report copied to: {full_json_out}")

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    JSON_PATH = BASE_DIR / "retrieval_diversity_score" / "retrieval_report.json"
    OUT_DIR = BASE_DIR / "retrieval_diversity_score" / "retrieval_reports"
    # Set run_retrieval=True to compute retrievals (slow), or False to parse an existing JSON
    run_analysis(json_path=JSON_PATH, out_dir=OUT_DIR, run_retrieval=True, k=5)
