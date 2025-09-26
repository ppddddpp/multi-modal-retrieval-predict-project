from pathlib import Path
import sys
try:
    base = Path(__file__).resolve().parent.parent
except NameError:
    base = Path.cwd().parent
sys.path.append(str(base))

import json
from collections import defaultdict
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from .dataParser import parse_openi_xml
from LabelData import disease_groups, normal_groups, finding_groups, symptom_groups

# Resolve paths...
try:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
except NameError:
    BASE_DIR = Path.cwd().parent.parent

XML_DIR    = BASE_DIR / 'data' / 'openi' / 'xml' / 'NLMCXR_reports' / 'ecgen-radiology'
DICOM_ROOT = BASE_DIR / 'data' / 'openi' / 'dicom'
LABEL_CSV  = BASE_DIR / 'outputs' / 'openi_labels_final_cleaned.csv'
SPLIT_DIR  = BASE_DIR / 'splited_data'
SPLIT_DIR.mkdir(parents=True, exist_ok=True)


def train_val_test_split(xml_dir=None, dicom_dir=None, combined_groups=None,
                         label_csv=None, split_dir=None, seed=42,
                         split_ratio=(0.8, 0.1, 0.1)):
    """Create report-level multilabel stratified train/val/test splits.
       Saves train/val/test id json files and labeled CSVs (using cleaned label CSV).
    """
    if xml_dir is None: xml_dir = XML_DIR
    if dicom_dir is None: dicom_dir = DICOM_ROOT
    if combined_groups is None:
        raise ValueError("Provide combined_groups")
    if label_csv is None: label_csv = LABEL_CSV
    if split_dir is None: split_dir = SPLIT_DIR

    if not Path(label_csv).exists():
        raise FileNotFoundError(f"Label CSV not found: {label_csv}")

    # --- Parse records grouped by report_text ---
    records = parse_openi_xml(xml_dir, dicom_dir, combined_groups)
    report_to_records = defaultdict(list)
    for rec in records:
        report_to_records[rec["report_text"]].append(rec)
    reports = list(report_to_records.keys())
    if len(reports) == 0:
        raise RuntimeError("No reports parsed. Check parse_openi_xml output.")

    # --- Load labels CSV ---
    labels_df = pd.read_csv(label_csv).set_index("id")
    label_cols = (list(disease_groups.keys()) +
                  list(normal_groups.keys()) +
                  list(finding_groups.keys()) +
                  list(symptom_groups.keys()))

    # --- Build per-report label vectors ---
    report_label_vecs = []
    report_id_lists = []
    missing_ids = 0
    for text in reports:
        recs = report_to_records[text]
        ids = [r["id"] for r in recs]
        report_id_lists.append(ids)
        rows = labels_df.reindex(ids)  # safe: missing ids -> NaN
        if rows.isnull().any(axis=None):
            missing_ids += int(rows.isnull().any(axis=1).sum())
        vec = (rows[label_cols].fillna(0).sum(axis=0) > 0).astype(int).values
        report_label_vecs.append(vec)
    if missing_ids:
        print(f"[WARN] {missing_ids} image IDs from parsed records not found in {label_csv} (treated as zeros).")

    report_label_vecs = np.vstack(report_label_vecs)  # (R, C)

    # --- Stratified splitting ---
    train_frac, val_frac, test_frac = split_ratio
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("split_ratio must sum to 1.0")

    # train vs temp
    msss1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=(1 - train_frac), random_state=seed)
    train_idx, temp_idx = next(msss1.split(np.zeros(len(report_label_vecs)), report_label_vecs))

    # temp -> val/test
    temp_labels = report_label_vecs[temp_idx]
    val_rel = val_frac / (val_frac + test_frac)
    msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=(1 - val_rel), random_state=seed+1)
    val_rel_idx, test_rel_idx = next(msss2.split(np.zeros(len(temp_idx)), temp_labels))
    val_idx = temp_idx[val_rel_idx]
    test_idx = temp_idx[test_rel_idx]

    train_reports = [reports[i] for i in train_idx]
    val_reports   = [reports[i] for i in val_idx]
    test_reports  = [reports[i] for i in test_idx]

    # flatten to record ids
    def flatten_ids(report_list):
        return [r["id"] for rpt in report_list for r in report_to_records[rpt]]
    train_ids = [str(x) for x in flatten_ids(train_reports)]
    val_ids   = [str(x) for x in flatten_ids(val_reports)]
    test_ids  = [str(x) for x in flatten_ids(test_reports)]

    # --- Save ID splits ---
    split_dir = Path(split_dir)
    with open(split_dir / "train_split_ids.json", "w") as f: json.dump(train_ids, f)
    with open(split_dir / "val_split_ids.json", "w") as f: json.dump(val_ids, f)
    with open(split_dir / "test_split_ids.json", "w") as f: json.dump(test_ids, f)

    # --- Save labeled CSVs (only IDs present in CSV) ---
    df = pd.read_csv(label_csv)
    available_ids = set(df["id"].astype(str))
    def keep_present(ids): return [i for i in ids if i in available_ids]
    train_ids_present = keep_present(train_ids)
    val_ids_present   = keep_present(val_ids)
    test_ids_present  = keep_present(test_ids)

    df[df["id"].astype(str).isin(train_ids_present)].to_csv(split_dir / "openi_train_labeled.csv", index=False)
    df[df["id"].astype(str).isin(val_ids_present)].to_csv(split_dir / "openi_val_labeled.csv", index=False)
    df[df["id"].astype(str).isin(test_ids_present)].to_csv(split_dir / "openi_test_labeled.csv", index=False)

    # --- Diagnostics ---
    df_indexed = df.set_index("id")
    def per_label_counts(ids):
        arr = df_indexed.reindex(ids)[label_cols].fillna(0).values if len(ids) > 0 else np.zeros((0, len(label_cols)))
        return arr.sum(axis=0).astype(int)

    train_counts = per_label_counts(train_ids_present)
    val_counts   = per_label_counts(val_ids_present)
    test_counts  = per_label_counts(test_ids_present)

    diag = pd.DataFrame({
        "label": label_cols,
        "train_pos": train_counts,
        "val_pos": val_counts,
        "test_pos": test_counts
    }).sort_values("val_pos")

    print("Split sizes (records):", len(train_ids_present), len(val_ids_present), len(test_ids_present))
    print("Per-label positives in validation (bottom 40):")
    print(diag[["label", "val_pos"]].head(40).to_string(index=False))
    zeros = diag[diag.val_pos == 0].label.tolist()
    if zeros:
        print(f"[WARN] {len(zeros)} labels have 0 positives in validation. Consider larger val, k-fold, or merging very-rare labels.")

    print("Saved stratified report-level splits and CSVs to:", split_dir)
    return train_ids_present, val_ids_present, test_ids_present


if __name__ == "__main__":
    combined_groups = {
        **disease_groups,
        **finding_groups,
        **symptom_groups,
        **normal_groups
    }
    train_ids, val_ids, test_ids = train_val_test_split(
        xml_dir=XML_DIR,
        dicom_dir=DICOM_ROOT,
        combined_groups=combined_groups,
        label_csv=LABEL_CSV,
        split_dir=SPLIT_DIR,
        seed=42,
        split_ratio=(0.8, 0.1, 0.1)
    )
