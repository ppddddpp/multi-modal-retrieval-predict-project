import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_fscore_support

from config import Config
from model import MultiModalRetrievalModel
from dataParser import parse_openi_xml
from dataLoader import build_dataloader
from labeledData import disease_groups, normal_groups

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "configs" / "config.yaml"
CKPT_PATH   = BASE_DIR / "checkpoints" / "model_best.pt"
SPLIT_DIR  = BASE_DIR / "splited_data"
XML_DIR    = BASE_DIR / "data" / "openi" / "xml" / "NLMCXR_reports" / "ecgen-radiology"
DICOM_ROOT = BASE_DIR / "data" / "openi" / "dicom"
LABELS_CSV = BASE_DIR / "outputs" / "openi_labels_final.csv"

def main():

    # Load config and device
    cfg = Config.load(CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load label columns
    label_cols = list(disease_groups.keys()) + list(normal_groups.keys())

    # Load test IDs and filter records
    with open(SPLIT_DIR / "test_split_ids.json") as f:
        test_ids = set(json.load(f))

    df_labels = pd.read_csv(LABELS_CSV).set_index("id")
    parsed_records = parse_openi_xml(XML_DIR, DICOM_ROOT)

    # Align records
    records = []
    for rec in parsed_records:
        rec_id = rec["id"]
        if rec_id in test_ids and rec_id in df_labels.index:
            try:
                label_vec = df_labels.loc[rec_id, label_cols].astype(int).tolist()
                if len(label_vec) != len(label_cols):
                    print(f"[WARN] Skipping {rec_id}: label vector length mismatch")
                    continue
                rec["labels"] = label_vec
                records.append(rec)
            except Exception as e:
                print(f"[WARN] Skipping {rec_id}: {e}")


    print(f"[INFO] Loaded {len(records)} test records")

    # Build test dataloader
    test_loader = build_dataloader(
        records=records,
        batch_size=cfg.batch_size,
        mean=0.5,
        std=0.25,
        shuffle=False
    )

    # Load model
    model = MultiModalRetrievalModel(
        joint_dim=cfg.joint_dim,
        num_heads=cfg.num_heads,
        num_classes=len(label_cols),
        fusion_type=cfg.fusion_type,
        swin_ckpt_path=BASE_DIR / "models" / "swin_checkpoint.safetensors",
        bert_local_dir=BASE_DIR / "models" / "clinicalbert_local",
        checkpoint_path=str(CKPT_PATH),
        device=device,
        training=False
    ).to(device)

    model.eval()

    # Run on test set
    y_true, y_prob = [], []

    with torch.no_grad():
        for batch in test_loader:
            img  = batch["image"].to(device)
            ids  = batch["input_ids"].to(device)
            mask = batch["attn_mask"].to(device)
            lbls = batch["labels"].cpu().numpy()

            output = model.predict(img, ids, mask, threshold=0.5, explain=False)
            probs = output["probs"]

            y_true.extend(lbls)
            y_prob.extend(probs)

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob > 0.5).astype(int)

    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    auroc = [
        roc_auc_score(y_true[:, i], y_prob[:, i])
        if np.any(y_true[:, i]) else 0.0
        for i in range(y_true.shape[1])
    ]

    # Print per-label results
    print(f"{'Label':40s}  Prec   Rec    F1     AUROC")
    for i, label in enumerate(label_cols):
        print(f"{label:40s}  {prec[i]:.3f}  {rec[i]:.3f}  {f1[i]:.3f}  {auroc[i]:.3f}")

    # Macro average
    macro_f1 = np.mean(f1)
    macro_auc = np.mean(auroc)
    print(f"\n[MACRO] F1: {macro_f1:.4f}  AUROC: {macro_auc:.4f}")

if __name__ == "__main__":
    main()