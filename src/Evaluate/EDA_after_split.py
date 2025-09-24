from pathlib import Path
import sys
try:
    base = Path(__file__).resolve().parent.parent
except NameError:
    base = Path.cwd().parent
sys.path.append(str(base))
import numpy as np
import matplotlib.pyplot as plt
import json
from DataHandler import parse_openi_xml
from LabelData import disease_groups, normal_groups, finding_groups, symptom_groups
import os
import pandas as pd

try:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
except NameError:
    BASE_DIR = Path.cwd().parent.parent
XML_DIR = BASE_DIR / 'data' / 'openi' / 'xml' / 'NLMCXR_reports' / 'ecgen-radiology'
DICOM_ROOT = BASE_DIR / 'data' / 'openi' / 'dicom'
SPLIT_DIR = BASE_DIR / 'splited_data'
LABEL_CSV = BASE_DIR / 'outputs' / 'openi_labels_final_cleaned.csv'

def avg_labels_per_image(records, field='labels'):
    mat = np.array([r[field] for r in records])
    return mat.sum(axis=1).mean()

def label_distribution(records, field='labels'):
    mat = np.array([r[field] for r in records])
    return mat.sum(axis=0)

def get_eda_after_split(xml_dir=XML_DIR, dicom_root=DICOM_ROOT, 
                        split_dir=SPLIT_DIR, label_csv=LABEL_CSV, 
                        combined_groups=None):
    # Label setup
    combined_group_temp = {
        **disease_groups, 
        **normal_groups,
        **finding_groups,
        **symptom_groups
        }
    combined_groups = combined_group_temp if combined_groups is None else combined_groups
    label_cols = sorted(combined_groups.keys())

    # Load final labels
    labels_df = pd.read_csv(label_csv).set_index("id")

    # Parse raw XML records
    raw_records = parse_openi_xml(xml_dir, dicom_root, combined_groups=combined_groups)

    # Merge labels into parsed records
    labeled_records = []
    for rec in raw_records:
        rec_id = rec["id"]
        if rec_id in labels_df.index:
            rec["labels"] = labels_df.loc[rec_id, label_cols].tolist()
            labeled_records.append(rec)

    # Load split IDs
    with open(split_dir / 'train_split_ids.json') as f: train_ids = set(json.load(f))
    with open(split_dir / 'val_split_ids.json')   as f: val_ids   = set(json.load(f))
    with open(split_dir / 'test_split_ids.json')  as f: test_ids  = set(json.load(f))

    # Filter records into splits
    train_records = [r for r in labeled_records if r['id'] in train_ids]
    val_records   = [r for r in labeled_records if r['id'] in val_ids]
    test_records  = [r for r in labeled_records if r['id'] in test_ids]

    print(f"Train size: {len(train_records)}, Val size: {len(val_records)}, Test size: {len(test_records)}")

    # Compute distributions
    train_counts = label_distribution(train_records)
    val_counts   = label_distribution(val_records)
    test_counts  = label_distribution(test_records)

    x = np.arange(len(label_cols))
    plt.figure(figsize=(14,6))
    plt.bar(x-0.3, train_counts, width=0.25, label='Train')
    plt.bar(x,      val_counts,   width=0.25, label='Val')
    plt.bar(x+0.3,  test_counts,  width=0.25, label='Test')
    plt.xticks(x, label_cols, rotation=45, ha="right")
    plt.ylabel("Number of Cases")
    plt.title("Final (Verified) Disease Label Distribution Across Splits")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Print avg labels/image
    print(f"Average labels/image:")
    print(f"  Train: {avg_labels_per_image(train_records):.2f}")
    print(f"  Val:   {avg_labels_per_image(val_records):.2f}")
    print(f"  Test:  {avg_labels_per_image(test_records):.2f}")

    # Save to Markdown report
    with open(SPLIT_DIR / "split_stats_verified.md", "w") as f:
        f.write(f"# Final Labeled Data Split Summary (80/10/10)\n\n")
        f.write(f"- Train records: {len(train_records)}\n")
        f.write(f"- Val   records: {len(val_records)}\n")
        f.write(f"- Test  records: {len(test_records)}\n\n")
        for name, t, v, te in zip(label_cols, train_counts, val_counts, test_counts):
            f.write(f"- **{name}** -> train: {t}, val: {v}, test: {te}\n")
        f.write(f"\n- Avg labels/image -> train: {avg_labels_per_image(train_records):.2f}, "
                f"val: {avg_labels_per_image(val_records):.2f}, "
                f"test: {avg_labels_per_image(test_records):.2f}\n")