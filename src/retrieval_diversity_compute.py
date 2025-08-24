import json
from pathlib import Path
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np

def load_json(path: str) -> List[Dict[str,Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected JSON root to be a list of query objects.")
    return data

# ------------------------
# Query → Retrieval parsing
# ------------------------
def flatten_retrievals(data: List[Dict[str,Any]]) -> pd.DataFrame:
    """Flatten the JSON into a long DataFrame: one row per retrieval entry (Q to R)."""
    rows = []
    for q in data:
        qid = q.get("qid")
        qreport = q.get("query_report")
        retrievals = q.get("retrieval", []) or []
        for rank, r in enumerate(retrievals, start=1):
            base = {
                "qid": qid,
                "query_report": qreport,
                "rid": r.get("rid"),
                "dist": r.get("dist"),
                "rank": rank,
                "report": r.get("report"),
                "attn_txt_path": r.get("attn_txt_path"),
                "attn_img_path": r.get("attn_img_path"),
                "attn_comb_path": r.get("attn_comb_path"),
                "ig_path": r.get("ig_path"),
                "gradcam_path": r.get("gradcam_path"),
                "error": r.get("error"),
            }
            cm = r.get("compare_metrics") or {}
            for k, v in cm.items():
                try:
                    base[k] = float(v) if v is not None else np.nan
                except Exception:
                    base[k] = np.nan
            rows.append(base)
    df = pd.DataFrame(rows)
    id_cols = ["qid", "rid", "rank", "dist", "report",
               "query_report", "error",
               "attn_txt_path", "attn_img_path", "attn_comb_path",
               "ig_path", "gradcam_path"]
    metric_cols = [c for c in df.columns if c not in id_cols]
    ordered = id_cols + sorted(metric_cols)
    ordered = [c for c in ordered if c in df.columns]
    return df[ordered]

# ------------------------
# Retrieval → Retrieval parsing
# ------------------------
def flatten_rr(data: List[Dict[str,Any]]) -> pd.DataFrame:
    """
    Flatten retrieval to retrieval metrics at query-level (one row per qid).
    Works with keys like retrieval_overlap_iou5 / retrieval_diversity_score.
    """
    rows = []
    for q in data:
        qid = q.get("qid")
        row = {"qid": qid}
        found = False
        for key in ["retrieval_overlap_iou5", "retrieval_diversity_score"]:
            if key in q:
                try:
                    row[key] = float(q[key]) if q[key] is not None else np.nan
                except Exception:
                    row[key] = np.nan
                found = True
        if found:
            rows.append(row)
    return pd.DataFrame(rows)

# ------------------------
# Utility functions
# ------------------------
def discover_metrics(df: pd.DataFrame) -> List[str]:
    non_metric = {"qid","rid","rank","dist","report","query_report","error",
                  "attn_txt_path","attn_img_path","attn_comb_path","ig_path","gradcam_path"}
    return [c for c in df.columns if c not in non_metric]

def get_metric_scores(df: pd.DataFrame, metric: str, dropna: bool = True) -> pd.DataFrame:
    if metric not in df.columns:
        raise KeyError(f"Metric '{metric}' not found. Available: {discover_metrics(df)}")
    subset = df[["qid","rid","rank","dist","report", metric]].copy()
    subset = subset.rename(columns={metric: "score"})
    if dropna:
        subset = subset[subset["score"].notna()]
    return subset.sort_values(["qid","rank"])

def summary_all_metrics(df: pd.DataFrame) -> pd.DataFrame:
    metrics = discover_metrics(df)
    rows = []
    total = len(df)
    for m in metrics:
        s = pd.to_numeric(df[m], errors="coerce")
        cnt = int(s.count())
        rows.append({
            "metric": m,
            "mean": float(s.mean()) if cnt>0 else np.nan,
            "std": float(s.std()) if cnt>1 else np.nan,
            "median": float(s.median()) if cnt>0 else np.nan,
            "min": float(s.min()) if cnt>0 else np.nan,
            "max": float(s.max()) if cnt>0 else np.nan,
            "count_nonnull": cnt,
            "total_rows": total,
            "pct_missing": 100.0 * (1.0 - cnt/total) if total>0 else np.nan
        })
    return pd.DataFrame(rows).set_index("metric").sort_index()

def aggregate_per_qid(df: pd.DataFrame, metric: str, agg_funcs: Optional[List[str]] = None) -> pd.DataFrame:
    if metric not in df.columns:
        raise KeyError(f"Metric '{metric}' not found.")
    if agg_funcs is None:
        agg_funcs = ['mean','median','std','min','max','count']
    s = df[["qid", metric]].copy()
    s[metric] = pd.to_numeric(s[metric], errors="coerce")
    group = s.groupby("qid").agg({metric: agg_funcs})
    group.columns = ["_".join(col) if isinstance(col, tuple) else col for col in group.columns.values]
    group = group.reset_index()
    counts = df.groupby("qid").size().rename("total_retrieved").reset_index()
    nonnulls = df.groupby("qid")[metric].apply(lambda x: x.notna().sum()).rename("nonnull_count").reset_index()
    group = group.merge(counts, on="qid", how="left").merge(nonnulls, on="qid", how="left")
    group["pct_missing"] = 100.0 * (1.0 - group["nonnull_count"] / group["total_retrieved"])
    return group

def top_k_by_metric(df: pd.DataFrame, metric: str, k: int = 1, higher_is_better: bool = True) -> pd.DataFrame:
    if metric not in df.columns:
        raise KeyError(f"Metric '{metric}' not found.")
    s = df[["qid","rid","rank","dist","report",metric]].copy()
    s[metric] = pd.to_numeric(s[metric], errors="coerce")
    s = s[s[metric].notna()].copy()
    s = s.sort_values(["qid", metric], ascending=[True, not higher_is_better])
    topk = s.groupby("qid").head(k).reset_index(drop=True)
    topk["k_rank"] = topk.groupby("qid").cumcount()+1
    return topk

def save_df(df: pd.DataFrame, outpath: str) -> None:
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outpath, index=False)
    print(f"Saved: {outpath}")

# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Parse retrieval JSON and extract metric results (Q to R and R and R).")
    parser.add_argument("--json", type=str, default="retrieval_diversity_score/retrieval_reports/retrieval_report.json",
                        help="Path to retrieval JSON (list of queries).")
    parser.add_argument("--out-dir", type=str, default="retrieval_diversity_score/retrieval_reports/",
                        help="Directory to save flattened CSV and summaries.")
    args = parser.parse_args()

    data = load_json(args.json)

    # ---- Q→R ----
    df = flatten_retrievals(data)
    print(f"[Q to R] Flattened rows: {len(df)}  unique qids: {df['qid'].nunique() if 'qid' in df else 0}")
    outdir = Path(args.out_dir); outdir.mkdir(parents=True, exist_ok=True)
    save_df(df, str(outdir / "retrieval_metrics_flat.csv"))
    summary = summary_all_metrics(df)
    summary.to_csv(outdir / "retrieval_metrics_summary.csv")
    print(summary.head(20))

    # ---- R↔R ----
    df_rr = flatten_rr(data)
    if not df_rr.empty:
        print(f"[R and R] Rows: {len(df_rr)}  unique qids: {df_rr['qid'].nunique()}")
        save_df(df_rr, str(outdir / "retrieval_retrieval_metrics.csv"))
        summary_rr = summary_all_metrics(df_rr)
        summary_rr.to_csv(outdir / "retrieval_retrieval_summary.csv")
        print(summary_rr.head(20))
    else:
        print("[R and R] No retrieval to retrieval metrics found in JSON.")

    # Example: per-qid aggregation on Q→R metric
    example_metrics = [m for m in ["txt2img_pearson","img2txt_pearson",
                                   "final_patch_map_iou_5","final_patch_map_iou_20"]
                       if m in discover_metrics(df)]
    for m in example_metrics:
        agg = aggregate_per_qid(df, m)
        save_df(agg, str(outdir / f"per_qid_agg__{m}.csv"))

    # Example: top-1 by pearson if exists
    metric_for_topk = "final_patch_map_pearson" if "final_patch_map_pearson" in df.columns else None
    if metric_for_topk:
        top1 = top_k_by_metric(df, metric_for_topk, k=1, higher_is_better=True)
        save_df(top1, str(outdir / f"top1_by__{metric_for_topk}.csv"))
        print("Top-1 sample:")
        print(top1.head(10))
