from pathlib import Path
import sys
try:
    base = Path(__file__).resolve().parent.parent
except NameError:
    base = Path.cwd().parent
sys.path.append(str(base))

import torch
import numpy as np
import pandas as pd
import json
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score, f1_score

from Helpers import Config, safe_roc_auc, safe_avg_precision, log_and_print
from Model import MultiModalRetrievalModel
from DataHandler import build_dataloader, parse_openi_xml
from LabelData import disease_groups, normal_groups, finding_groups, symptom_groups

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = BASE_DIR / "configs" / "config.yaml"
CKPT_PATH   = BASE_DIR / "checkpoints" / "model_best.pt"
SPLIT_DIR   = BASE_DIR / "splited_data"
XML_DIR     = BASE_DIR / "data" / "openi" / "xml" / "NLMCXR_reports" / "ecgen-radiology"
DICOM_ROOT  = BASE_DIR / "data" / "openi" / "dicom"
LABELS_CSV  = BASE_DIR / "outputs" / "openi_labels_final.csv"
CSV_EVAL_SAVE_PATH = BASE_DIR / "eval_test_csvs"

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

def eval_on_test(use_fixed_threshold=False, fixed_threshold=0.5, split="test"):
    """
    Evaluate model on `split` ("val" or "test") and compute same metrics as training:
     - per-class AUROC, AP, Precision, Recall, F1, thresholds
     - macro & micro aggregated metrics (AUROC, AP, F1, prec, rec)
    Returns metrics dict.
    """
    log_file = CSV_EVAL_SAVE_PATH / f"{split}_eval_log.txt"

    # Use the exact same label ordering as training
    label_cols = list(disease_groups.keys()) + list(normal_groups.keys()) + list(finding_groups.keys()) + list(symptom_groups.keys())
    log_and_print(f"[INFO] Evaluating with {len(label_cols)} label columns (first 10): {label_cols[:10]}", log_file=log_file)

    cfg = Config.load(CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load split ids
    split_file = "val_split_ids.json" if split == "val" else "test_split_ids.json"
    with open(SPLIT_DIR / split_file) as f:
        split_ids = set(json.load(f))

    df_labels = pd.read_csv(LABELS_CSV).set_index("id")
    parsed_records = parse_openi_xml(XML_DIR, DICOM_ROOT, combined_groups={**disease_groups, **normal_groups, **finding_groups, **symptom_groups})

    # align
    records = []
    for rec in parsed_records:
        rec_id = rec["id"]
        if rec_id in split_ids and rec_id in df_labels.index:
            try:
                label_vec = df_labels.loc[rec_id, label_cols].astype(int).tolist()
                if len(label_vec) != len(label_cols):
                    log_and_print(f"[WARN] Skipping {rec_id}: label vector length mismatch",  log_file=log_file)
                    continue
                rec["labels"] = label_vec
                records.append(rec)
            except Exception as e:
                log_and_print(f"[WARN] Skipping {rec_id}: {e}", log_file=log_file)

    log_and_print(f"[INFO] Loaded {len(records)} records for split={split}", log_file=log_file)

    # dataloader
    test_loader = build_dataloader(records=records, batch_size=cfg.batch_size, mean=0.5, std=0.25, shuffle=False)

    # build model (do NOT auto-load checkpoint inside ctor; we'll load manually to check)
    model = MultiModalRetrievalModel(
        joint_dim=cfg.joint_dim,
        num_heads=cfg.num_heads,
        num_fusion_layers=cfg.num_fusion_layers,
        num_classes=len(label_cols),
        fusion_type=cfg.fusion_type,
        swin_ckpt_path=BASE_DIR / "models" / "swin_checkpoint.safetensors",
        bert_local_dir=BASE_DIR / "models" / "clinicalbert_local",
        checkpoint_path=CKPT_PATH,   
        device=device,
        use_shared_ffn=cfg.use_shared_ffn,
        use_cls_only=cfg.use_cls_only,
        training=False
    ).to(device)

    model.to(device)
    model.eval()

    # inference collect logits -> probs
    all_ids = []
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            img = batch["image"].to(device)
            ids_batch = batch["input_ids"].to(device)
            mask = batch["attn_mask"].to(device)
            lbls = batch["labels"].cpu().numpy()
            batch_ids = batch.get("id", None)

            outputs = model(img, ids_batch, mask, return_attention=False)
            logits = outputs["logits"]  # raw logits (B, C)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_labels.append(lbls)
            all_logits.append(probs)
            if batch_ids is not None:
                all_ids.extend(batch_ids)

    if not all_labels:
        raise RuntimeError("No records processed. Check split files and label CSV.")

    y_true = np.vstack(all_labels)         # (N, C)
    y_prob = np.vstack(all_logits)         # (N, C)

    # thresholds
    if use_fixed_threshold:
        best_ts = np.full(y_prob.shape[1], fixed_threshold)
    else:
        # same thresholding used in training
        best_ts = _find_best_thresholds(y_true, y_prob)

    y_pred_bin = (y_prob > best_ts[None, :]).astype(int)

    # per-class metrics
    class_aucs = safe_roc_auc(y_true, y_prob, label_names=label_cols)
    class_aps  = safe_avg_precision(y_true, y_prob, label_names=label_cols)

    class_precs = precision_score(y_true, y_pred_bin, average=None, zero_division=0)
    class_recs  = recall_score(y_true, y_pred_bin, average=None, zero_division=0)
    class_f1s   = f1_score(y_true, y_pred_bin, average=None, zero_division=0)

    # aggregated metrics (matching train)
    macro_auc = float(np.nanmean(class_aucs))
    macro_ap  = float(np.nanmean(class_aps))
    try:
        micro_ap = float(average_precision_score(y_true, y_prob, average='micro'))
    except Exception:
        micro_ap = float("nan")

    macro_f1 = float(f1_score(y_true, y_pred_bin, average='macro', zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred_bin, average='micro', zero_division=0))
    macro_prec = float(precision_score(y_true, y_pred_bin, average='macro', zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred_bin, average='macro', zero_division=0))
    micro_prec = float(precision_score(y_true, y_pred_bin, average='micro', zero_division=0))
    micro_rec = float(recall_score(y_true, y_pred_bin, average='micro', zero_division=0))

    # Print per-class lines (optional): uncomment if you want per-class
    log_and_print(f"{'Label':40s}  AUROC   AP      Prec   Rec    F1    Threshold", log_file=log_file)
    for i, label in enumerate(label_cols):
        auc = class_aucs[i] if not np.isnan(class_aucs[i]) else 0.0
        ap  = class_aps[i]  if not np.isnan(class_aps[i])  else 0.0
        log_and_print(f"{label:40s}  {auc:6.3f}  {ap:6.3f}  {class_precs[i]:6.3f}  {class_recs[i]:6.3f}  {class_f1s[i]:6.3f}  {best_ts[i]:5.3f}", log_file=log_file)

    # Print summary (same style as training)
    log_and_print("\n[SUMMARY]", log_file=log_file)
    log_and_print(f"[MACRO]  F1: {macro_f1:.4f}  AUROC: {macro_auc:.4f}  AP: {macro_ap:.4f}", log_file=log_file)
    log_and_print(f"[MICRO]  F1: {micro_f1:.4f}  AP(micro): {micro_ap:.4f}  Prec: {micro_prec:.4f}  Rec: {micro_rec:.4f}", log_file=log_file)

    # save CSV (id / true / prob / pred)
    CSV_EVAL_SAVE_PATH.mkdir(exist_ok=True)
    df_ids = pd.DataFrame({"id": all_ids})
    df_true = pd.DataFrame(y_true, columns=[f"true_{c}" for c in label_cols])
    df_prob = pd.DataFrame(y_prob, columns=[f"prob_{c}" for c in label_cols])
    df_pred = pd.DataFrame(y_pred_bin, columns=[f"pred_{c}" for c in label_cols])
    df_eval = pd.concat([df_ids, df_true, df_prob, df_pred], axis=1)
    out_path = CSV_EVAL_SAVE_PATH / f"eval_{split}_detailed.csv"
    df_eval.to_csv(out_path, index=False)
    print(f"[INFO] Saved evaluation CSV to: {out_path}")

    metrics = {
        "split": split,
        "num_samples": int(y_true.shape[0]),
        "num_classes": int(y_true.shape[1]),
        "macro_auc": macro_auc,
        "macro_ap": macro_ap,
        "macro_f1": macro_f1,
        "micro_ap": micro_ap,
        "micro_f1": micro_f1,
        "macro_prec": macro_prec,
        "macro_rec": macro_rec,
        "micro_prec": micro_prec,
        "micro_rec": micro_rec,
        "per_class": {
            "labels": label_cols,
            "auroc": [float(x) for x in class_aucs],
            "ap": [float(x) for x in class_aps],
            "prec": [float(x) for x in class_precs],
            "rec": [float(x) for x in class_recs],
            "f1": [float(x) for x in class_f1s],
            "thresholds": [float(x) for x in best_ts]
        }
    }
    return metrics

if __name__ == "__main__":
    # run on val to validate reproducing training logged numbers, then on test
    print("Running eval on VAL (use this to compare with WandB's val metrics)...")
    m_val = eval_on_test(use_fixed_threshold=False, split="val")
    print("\nNow running eval on TEST...")
    m_test = eval_on_test(use_fixed_threshold=False, split="test")
