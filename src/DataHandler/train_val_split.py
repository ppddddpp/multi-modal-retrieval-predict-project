from pathlib import Path
import sys
try:
    base = Path(__file__).resolve().parent.parent
except NameError:
    base = Path.cwd().parent
sys.path.append(str(base))
import json
from collections import defaultdict
from .dataParser import parse_openi_xml
from pathlib import Path
import random
import pandas as pd
from LabelData import disease_groups, normal_groups, finding_groups, symptom_groups

# Resolve paths...
try:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
except NameError:
    BASE_DIR = Path.cwd().parent.parent

XML_DIR    = BASE_DIR / 'data' / 'openi' / 'xml' / 'NLMCXR_reports' / 'ecgen-radiology'
DICOM_ROOT = BASE_DIR / 'data' / 'openi' / 'dicom'
LABEL_CSV   = BASE_DIR / 'outputs' / 'openi_labels_final_cleaned.csv'
SPLIT_DIR   = BASE_DIR / 'splited_data'
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

def train_val_test_split(xml_dir=None, dicom_dir=None, combined_groups=None, label_csv=None, split_dir=None, seed=42, split_ratio=[0.8, 0.1, 0.1]):
    """
    Split parsed OpenI XML reports into train, validation, and test sets.

    Parameters
    ----------
    xml_dir : str
        Path to folder containing individual .xml report files
    dicom_dir : str
        Root folder where .dcm files live (possibly nested)
    combined_groups : dict of str to list of str
        Dictionary where keys are disease/normal group names and values are lists of labels
    label_csv : str
        Path to the CSV file containing labels
    split_dir : str
        Path to save the train, validation, and test CSV files to
    seed : int
        Random seed for shuffling reports
    split_ratio : list of float
        Split ratio for train, validation, and test sets

    Returns
    -------
    None

    Saves the labeled train, validation, and test sets to separate CSV files
    """
    if xml_dir is None:
        xml_dir = XML_DIR
    if dicom_dir is None:
        dicom_dir = DICOM_ROOT
    if combined_groups is None:
        raise ValueError("Please provide a least a list of disease groups and normal groups to label the report with.")
    if label_csv is None:
        label_csv = LABEL_CSV
    if split_dir is None:
        split_dir = SPLIT_DIR
    
    # Load parsed data
    records = parse_openi_xml(xml_dir, dicom_dir, combined_groups)

    def collect_ids(report_set):
        return [rec["id"] for rpt in report_set for rec in report_to_records[rpt]]

    # Group by report
    report_to_records = defaultdict(list)
    for rec in records:
        report_to_records[rec["report_text"]].append(rec)

    # Shuffle reports
    reports = list(report_to_records.keys())
    random.seed(seed if seed is not None else 42)
    random.shuffle(reports)

    # Splitting 80/10/10
    n = len(reports)
    p1 = int(n * split_ratio[0])                           # end of train
    p2 = int(n * split_ratio[0] + n * split_ratio[1])      # end of val

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
    overlap = (train_ids & val_ids) | (train_ids & test_ids) | (val_ids & test_ids)
    print(f"Train/Val/Test ID overlap: {len(overlap)}")

    # Save split IDs to disk
    with open(split_dir / "train_split_ids.json", "w") as f:
        json.dump([r["id"] for r in train_records], f)
    with open(split_dir / "val_split_ids.json", "w") as f:
        json.dump([r["id"] for r in val_records], f)
    with open(split_dir / "test_split_ids.json", "w") as f:
        json.dump([r["id"] for r in test_records], f)

    labels_df = pd.read_csv(label_csv)

    df_train = labels_df[labels_df["id"].isin(train_ids)].copy()
    df_val   = labels_df[labels_df["id"].isin(val_ids)].copy()
    df_test  = labels_df[labels_df["id"].isin(test_ids)].copy()

    # Save the labeled splits
    df_train.to_csv(split_dir / "openi_train_labeled.csv", index=False)
    df_val.to_csv(split_dir / "openi_val_labeled.csv", index=False)
    df_test.to_csv(split_dir / "openi_test_labeled.csv", index=False)

    print(f"Total unique reports: {len(reports)}")
    print(f"Total records (DICOMs): {len(records)}")
    print(f"Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")

if __name__ == "__main__":
    combined_groups = {
        **disease_groups,
        **finding_groups,
        **symptom_groups,
        **normal_groups
    }
    train_val_test_split(combined_groups=combined_groups)