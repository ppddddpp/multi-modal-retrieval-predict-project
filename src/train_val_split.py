import json
from collections import defaultdict
from dataParser import parse_openi_xml
import os
from pathlib import Path
import random
import pandas as pd

# Resolve paths...
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd().parent

XML_DIR    = BASE_DIR / 'data' / 'openi' / 'xml' / 'NLMCXR_reports' / 'ecgen-radiology'
DICOM_ROOT = BASE_DIR / 'data' / 'openi' / 'dicom'
LABEL_CSV   = BASE_DIR / 'outputs' / 'openi_labels_final.csv'
SPLIT_DIR   = BASE_DIR / 'splited_data'
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

def collect_ids(report_set):
    return [rec["id"] for rpt in report_set for rec in report_to_records[rpt]]

if __name__ == '__main__':
    # Load parsed data
    records = parse_openi_xml(XML_DIR, DICOM_ROOT)

    # Group by report
    report_to_records = defaultdict(list)
    for rec in records:
        report_to_records[rec["report_text"]].append(rec)

    # Shuffle reports
    reports = list(report_to_records.keys())
    random.seed(42)
    random.shuffle(reports)

    # Splitting 80/10/10
    n = len(reports)
    p1 = int(n * 0.8)      # end of train
    p2 = int(n * 0.9)      # end of val

    train_reports = set(reports[:p1])
    val_reports   = set(reports[p1:p2])
    test_reports  = set(reports[p2:])

    # Build splits
    train_records, val_records, test_records = [], [], []
    for report in reports:
        if report in train_reports:  
            train_records.extend(report_to_records[report])
        elif report in val_reports:    
            val_records.extend(report_to_records[report])
        else:                          
            test_records.extend(report_to_records[report])

    # Summary
    print(f"Total records: {len(records)}")
    print(f"Train split:   {len(train_records)}")
    print(f"Val split:     {len(val_records)}")
    print(f"Test split:    {len(test_records)}")
    print(f"Unique reports: train={len(train_reports)}, val={len(val_reports)}, test={len(test_reports)}")

    train_ids = set([r["id"] for r in train_records])
    val_ids   = set([r["id"] for r in val_records])
    test_ids  = set([r["id"] for r in test_records])
    intersection = train_ids & val_ids & test_ids
    print(f"Train/Val/Test ID overlap: {len(intersection)}")

    # Save split IDs to disk
    with open(os.path.join(SPLIT_DIR, "train_split_ids.json"), "w") as f:
        json.dump([r["id"] for r in train_records], f)
    with open(os.path.join(SPLIT_DIR, "val_split_ids.json"), "w") as f:
        json.dump([r["id"] for r in val_records], f)
    with open(os.path.join(SPLIT_DIR, "test_split_ids.json"), "w") as f:
        json.dump([r["id"] for r in test_records], f)

    labels_df = pd.read_csv(LABEL_CSV)

    df_train = labels_df[labels_df["id"].isin(train_ids)].copy()
    df_val   = labels_df[labels_df["id"].isin(val_ids)].copy()
    df_test  = labels_df[labels_df["id"].isin(test_ids)].copy()

    # Save the labeled splits
    df_train.to_csv(SPLIT_DIR / "openi_train_labeled.csv", index=False)
    df_val.to_csv(SPLIT_DIR / "openi_val_labeled.csv", index=False)
    df_test.to_csv(SPLIT_DIR / "openi_test_labeled.csv", index=False)

    print(f"Total reports: {len(records)}")
    print(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")
    print(f"Train/Val/Test ID overlap: {len(set(train_ids) & set(val_ids) & set(test_ids))}")