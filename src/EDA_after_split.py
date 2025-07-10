import numpy as np
import matplotlib.pyplot as plt
import json
from dataParser import parse_openi_xml
from pathlib import Path
import os

def avg_labels_per_image(records):
    label_matrix = np.array([r['labels'] for r in records])
    return label_matrix.sum(axis=1).mean()

def label_distribution(records, label_names):
    label_matrix = np.array([r['labels'] for r in records])
    return label_matrix.sum(axis=0)

if __name__ == "__main__":
    # Resolve paths...
    try:
        BASE_DIR = Path(__file__).resolve().parent.parent
    except NameError:
        BASE_DIR = Path.cwd().parent

    XML_DIR    = BASE_DIR / 'data' / 'openi' / 'xml' / 'NLMCXR_reports' / 'ecgen-radiology'
    DICOM_ROOT = BASE_DIR / 'data' / 'openi' / 'dicom'
    split_dir = BASE_DIR / 'splited_data'
    train_set = split_dir / 'train_split_ids.json'
    val_set = split_dir / 'val_split_ids.json'

    # Load all records
    records = parse_openi_xml(XML_DIR, DICOM_ROOT)

    # Load split IDs
    with open(train_set) as f:
        train_ids = set(json.load(f))
    with open(val_set) as f:
        val_ids = set(json.load(f))

    # Filter records into splits
    train_records = [r for r in records if r['id'] in train_ids]
    val_records   = [r for r in records if r['id'] in val_ids]

    print(f"Train size: {len(train_records)}, Val size: {len(val_records)}")

    train_ids = set([r["id"] for r in train_records])
    val_ids   = set([r["id"] for r in val_records])
    intersection = train_ids & val_ids
    print(f"Train/Val ID overlap: {len(intersection)}")

    # Label distribution
    FINDINGS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
                "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule",
                "Pleural_Thickening", "Pneumonia", "Pneumothorax"]

    train_counts = label_distribution(train_records, FINDINGS)
    val_counts   = label_distribution(val_records, FINDINGS)

    x = np.arange(len(FINDINGS))
    plt.figure(figsize=(12,5))
    plt.bar(x - 0.2, train_counts, width=0.4, label='Train')
    plt.bar(x + 0.2, val_counts, width=0.4, label='Val')
    plt.xticks(x, FINDINGS, rotation=45, ha="right")
    plt.ylabel("Number of Cases")
    plt.title("Label Distribution in Train vs Val")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Train: avg labels/image = {avg_labels_per_image(train_records):.2f}")
    print(f"Val:   avg labels/image = {avg_labels_per_image(val_records):.2f}")

    # Save split stats
    with open(os.path.join(split_dir, "split_stats.md"), "w") as f:
        f.write(f"# Data Split Summary\n\n")
        f.write(f"- Total train records: {len(train_records)}\n")
        f.write(f"- Total val records: {len(val_records)}\n")
        f.write(f"- Unique label counts (train):\n")
        for lbl, cnt in zip(FINDINGS, train_counts):
            f.write(f"  - {lbl}: {cnt}\n")
        f.write(f"\n- Unique label counts (val):\n")
        for lbl, cnt in zip(FINDINGS, val_counts):
            f.write(f"  - {lbl}: {cnt}\n")