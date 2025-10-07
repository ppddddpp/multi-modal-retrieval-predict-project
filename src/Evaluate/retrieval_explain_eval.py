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
import time
from typing import List

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

from Helpers import Config
from Model import MultiModalRetrievalModel
from Retrieval import make_retrieval_engine
from DataHandler import parse_openi_xml, build_dataloader
from Helpers.retrieval_metrics import precision_at_k, mean_average_precision, mean_reciprocal_rank
from Helpers.helper import compare_maps
from LabelData import disease_groups, normal_groups, finding_groups, symptom_groups


# === PATH SETUP ===
BASE_DIR    = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = BASE_DIR / "configs" / "config.yaml"
GT_DIR      = BASE_DIR / "ground_truths"
SPLIT_DIR   = BASE_DIR / "splited_data"
CKPT_PATH   = BASE_DIR / "checkpoints" / "model_best.pt"
MODEL_DIR   = BASE_DIR / "models"
XML_DIR     = BASE_DIR / "data" / "openi" / "xml" / "NLMCXR_reports" / "ecgen-radiology"
DICOM_ROOT  = BASE_DIR / "data" / "openi" / "dicom"
EMBED_DIR   = BASE_DIR / "embeddings"


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise TypeError(f"Unsupported type for attention map: {type(x)}")


def retrieval_explain_eval_predict(k=10, combined_groups=None):
    """
    Evaluate retrieval metrics (P@k, mAP, MRR)
    and explainability alignment (Pearson, Spearman, IoU on top-F)
    using model.predict() which returns retrieval + explainability outputs.
    """
    if combined_groups is not None:
        print("[INFO] Using provided combined_groups")
    else:
        combined_groups = {**disease_groups, **normal_groups, **finding_groups, **symptom_groups}

    cfg = Config.load(CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_cols = list(combined_groups.keys())

    # === Load ground-truth relevance ===
    with open(GT_DIR / "test_relevance.json") as f:
        gt_general = json.load(f)
    with open(GT_DIR / "test_to_train_relevance.json") as f:
        gt_historical = json.load(f)

    # === Create retrieval engine with test embeddings ===
    print("Test embeddings:", np.load(EMBED_DIR / "test_joint_embeddings.npy").shape)
    retriever = make_retrieval_engine(
        str(EMBED_DIR / "test_joint_embeddings.npy"),
        str(EMBED_DIR / "test_ids.json"),
        method="dls",
        link_threshold=0.5,
        max_links=10
    )

    # === Model ===
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
        training=False,
        retriever=retriever
    ).to(device)
    model.eval()

    print(f"[INFO] Retriever embedding shape: {model.retriever.embs.shape}")

    # === Data Preparation ===
    label_cols = list(disease_groups.keys()) + list(normal_groups.keys())
    df_test = pd.read_csv(SPLIT_DIR / "openi_test_labeled.csv")
    with open(SPLIT_DIR / "test_split_ids.json") as f:
        test_ids = json.load(f)
    df_test = df_test[df_test["id"].isin(test_ids)].reset_index(drop=True)

    parsed_records = parse_openi_xml(XML_DIR, DICOM_ROOT, combined_groups=combined_groups)
    labels_df = pd.read_csv(BASE_DIR / "outputs" / "openi_labels_final.csv").set_index("id")

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

    test_records = [r for r in records if r["id"] in test_ids]
    test_loader = build_dataloader(
        records=test_records,
        batch_size=cfg.batch_size,
        mean=0.5,
        std=0.25,
        shuffle=False
    )

    # === Metrics Buffers ===
    all_ret_gen, all_rel_gen = [], []
    gen_times = []
    pearsons, spearmans, iou05s, iou20s = [], [], [], []

    # === Evaluation Loop ===
    for batch in tqdm(test_loader, desc="Evaluating Retrieval + Explainability"):
        img = batch["image"].to(device)
        ids = batch["input_ids"].to(device)
        mask = batch["attn_mask"].to(device)
        qids = batch["id"]

        with torch.no_grad():
            t0 = time.perf_counter()
            outputs = model.predict(
                image=img,
                input_ids=ids,
                attention_mask=mask,
                K=k,
                explain=True
            )
            gen_times.append(time.perf_counter() - t0)

        for i, qid in enumerate(qids):
            ret_ids = outputs["retrieval_ids"][i]
            attn_map = outputs.get("attention_map", None)

            gt_rel = gt_general.get(qid, [])
            all_ret_gen.append(ret_ids)
            all_rel_gen.append(gt_rel)

            # === Debugging retrieval alignment ===
            if not ret_ids:
                print("        No retrieved IDs for this query")
            elif not gt_rel:
                print("        No ground truth relevance for this query")

            # === Explainability Alignment ===
            try:
                if attn_map is None:
                    continue

                # --- Handle query attention map ---
                if isinstance(attn_map, dict):
                    q_map = attn_map.get("final_patch_map")
                    if q_map is None:
                        q_map = list(attn_map.values())[-1]
                else:
                    q_map = attn_map[i]

                q_map = to_numpy(q_map)

                if not ret_ids:
                    continue

                top_id = ret_ids[0]
                r_rec = next((r for r in test_records if r["id"] == top_id), None)
                if not r_rec:
                    continue

                # --- Build retrieved sample batch ---
                r_loader = build_dataloader([r_rec], batch_size=1, mean=0.5, std=0.25, shuffle=False)
                r_batch = next(iter(r_loader))

                with torch.no_grad():
                    r_outputs = model.predict(
                        image=r_batch["image"].to(device),
                        input_ids=r_batch["input_ids"].to(device),
                        attention_mask=r_batch["attn_mask"].to(device),
                        K=k,
                        explain=True
                    )

                r_attn = r_outputs.get("attention_map")

                if isinstance(r_attn, dict):
                    r_map = r_attn.get("final_patch_map")
                    if r_map is None:
                        r_map = list(r_attn.values())[-1]
                else:
                    r_map = r_attn[0]

                r_map = to_numpy(r_map)

                # --- Compare maps ---

                cm05 = compare_maps(q_map, r_map, topk_frac=0.05)
                cm20 = compare_maps(q_map, r_map, topk_frac=0.20)

                # record correlation metrics using the smaller-topk map
                pearsons.append(cm05.get("pearson", np.nan))
                spearmans.append(cm05.get("spearman", np.nan))

                # record IoU for each fraction from the matching call
                iou05s.append(cm05.get("iou_top5pct", np.nan))
                iou20s.append(cm20.get("iou_top20pct", np.nan))

            except Exception as e:
                print(f"[WARN] Explainability failed for {qid}: {type(e).__name__} - {e}")
                continue

    # === Retrieval Metrics ===
    p_gen  = np.mean([precision_at_k(r, rel, k=k) for r, rel in zip(all_ret_gen, all_rel_gen)])
    map_gen = mean_average_precision(all_ret_gen, all_rel_gen, k=k)
    mrr_gen = mean_reciprocal_rank(all_ret_gen, all_rel_gen)
    avg_gen_ms = 1000 * np.mean(gen_times)

    # === Explainability Metrics ===
    exp_results = {
        "pearson_mean": np.mean(pearsons) if pearsons else 0.0,
        "spearman_mean": np.mean(spearmans) if spearmans else 0.0,
        "iou05_mean": np.mean(iou05s) if iou05s else 0.0,
        "iou20_mean": np.mean(iou20s) if iou20s else 0.0,
    }

    # === Print Summary ===
    print("\n=== Retrieval Results ===")
    print(f"Generalization (test to test)  P@{k}: {p_gen:.4f}, mAP: {map_gen:.4f}, MRR: {mrr_gen:.4f}")
    print(f"Avg Query Time: {avg_gen_ms:.2f} ms")

    print("\n=== Explainability Alignment ===")
    print(f"Pearson: {exp_results['pearson_mean']:.4f}, Spearman: {exp_results['spearman_mean']:.4f}")
    print(f"IoU@0.05: {exp_results['iou05_mean']:.4f}, IoU@0.20: {exp_results['iou20_mean']:.4f}")

    # === Save Results ===
    result_dir = BASE_DIR / "retrieval_eval_result"
    result_dir.mkdir(exist_ok=True)
    result_path = result_dir / f"eval_explain_predict_k{k}.txt"

    with open(result_path, "w", encoding="utf-8") as f:
        f.write("=== Retrieval ===\n")
        f.write(f"Generalization (test to test)  P@{k}: {p_gen:.4f}, mAP: {map_gen:.4f}, MRR: {mrr_gen:.4f}\n")
        f.write(f"Avg Query Time: {avg_gen_ms:.2f} ms\n\n")
        f.write("=== Explainability Alignment ===\n")
        for key, val in exp_results.items():
            f.write(f"{key}: {val:.4f}\n")

    print(f"\n[INFO] Results saved to: {result_path}")


if __name__ == "__main__":
    retrieval_explain_eval_predict(k=5)
