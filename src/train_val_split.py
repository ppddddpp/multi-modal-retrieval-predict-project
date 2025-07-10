import json
from collections import defaultdict
from dataParser import parse_openi_xml
import os
from pathlib import Path
import random

# Resolve paths...
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd().parent

XML_DIR    = BASE_DIR / 'data' / 'openi' / 'xml' / 'NLMCXR_reports' / 'ecgen-radiology'
DICOM_ROOT = BASE_DIR / 'data' / 'openi' / 'dicom'

if __name__ == '__main__':
    print("Please run this script from the root directory of the repository.")

    # Create split directory
    split_dir = "splited_data"
    os.makedirs(split_dir, exist_ok=True)

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

    # Spliting 80/20
    split_point = int(len(reports) * 0.8)
    train_reports = set(reports[:split_point])

    # Build splits
    train_records, val_records = [], []
    for report in reports:
        if report in train_reports:
            train_records.extend(report_to_records[report])
        else:
            val_records.extend(report_to_records[report])

    # Summary
    print(f"Total records: {len(records)}")
    print(f"Train split:   {len(train_records)}")
    print(f"Val split:     {len(val_records)}")
    print(f"Unique reports: train={len(train_reports)}, val={len(reports) - len(train_reports)}")

    train_ids = set([r["id"] for r in train_records])
    val_ids   = set([r["id"] for r in val_records])
    intersection = train_ids & val_ids
    print(f"Train/Val ID overlap: {len(intersection)}")

    # Save split IDs to disk
    with open(os.path.join(split_dir, "train_split_ids.json"), "w") as f:
        json.dump([r["id"] for r in train_records], f)

    with open(os.path.join(split_dir, "val_split_ids.json"), "w") as f:
        json.dump([r["id"] for r in val_records], f)
