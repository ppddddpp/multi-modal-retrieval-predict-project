from dataParser import parse_openi_xml
from labeledData import disease_groups, normal_groups
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tensorDICOM import DICOMImagePreprocessor

# Resolve paths...
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd().parent

XML_DIR    = BASE_DIR / 'data' / 'openi' / 'xml' / 'NLMCXR_reports' / 'ecgen-radiology'
DICOM_ROOT = BASE_DIR / 'data' / 'openi' / 'dicom'
MODEL_PLACE = BASE_DIR / "models"
os.environ['TRANSFORMERS_CACHE'] = str(MODEL_PLACE)

if __name__ == '__main__':
    # Load parsed data
    records = parse_openi_xml(XML_DIR, DICOM_ROOT)

    # Build label matrix from the new 'disease_vector' field
    label_matrix = np.array([rec['labels'] for rec in records])
    label_sums   = label_matrix.sum(axis=0)

    combined_groups = {
        **disease_groups,
        **normal_groups
    }

    # Get label names in the same order as the vector
    label_names = sorted(combined_groups.keys())

    # Plot normal vs abnormal
    normal_idx = label_names.index("Normal")
    n_strict_normal = sum(
        vec[normal_idx] == 1 and sum(vec) == 1
        for vec in label_matrix
    )
    n_abnormal = sum(
        any(vec[i] for i in range(len(vec)) if i != normal_idx)
        for vec in label_matrix
    )
    print(f"Strict Normal samples (only 'Normal' = 1): {n_strict_normal}")
    print(f"Abnormal samples (any disease group = 1): {n_abnormal}")

    plt.figure()
    plt.pie(
        [n_strict_normal, n_abnormal],
        labels=["Normal", "Abnormal"],
        autopct="%1.1f%%",
        startangle=90
    )
    plt.title("Normal vs Abnormal Cases")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

    # Plot total cases per disease
    plt.figure(figsize=(12,4))
    plt.bar(label_names, label_sums)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Cases")
    plt.title("Disease Label Frequencies")
    plt.tight_layout()
    plt.show()

    # Count number of labels per image
    num_labels_per_sample = label_matrix.sum(axis=1)
    print("Average labels/image:", num_labels_per_sample.mean())
    print("Max labels on one image:", num_labels_per_sample.max())

    plt.figure()
    plt.hist(num_labels_per_sample, bins=np.arange(num_labels_per_sample.max()+2)-0.5, rwidth=0.8)
    plt.xlabel("Number of labels per image")
    plt.ylabel("Frequency")
    plt.title("Multi-label Distribution")
    plt.tight_layout()
    plt.show()

    # Check for shared reports
    report_map = defaultdict(list)
    for rec in records:
        report_map[rec['report_text']].append(rec['id'])

    shared = [ids for ids in report_map.values() if len(ids) > 1]
    print(f"→ Unique reports: {len(report_map)}")
    print(f"→ Reports shared by multiple images: {len(shared)}")
    print(f"→ Avg images per reused report: {np.mean([len(ids) for ids in shared]):.2f}")

    # Debug a sample DICOM
    dp = DICOMImagePreprocessor()
    sample = records[0]
    arr = dp.load_raw_array(sample['dicom_path'])

    plt.figure()
    plt.imshow(arr, cmap="gray")
    plt.title("DICOM Image")
    plt.axis("off")
    plt.show()

    # Print sample report and its labels
    print("--- Report ---")
    print(sample["report_text"])
    print("\n--- Labels ---")
    print({
        name: val
        for name, val in zip(label_names, sample["labels"])
        if val
    })
