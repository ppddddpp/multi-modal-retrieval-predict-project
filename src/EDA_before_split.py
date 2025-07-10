from dataParser import parse_openi_xml
from pathlib import Path
import os
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
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

    # Count labels
    label_counts = np.array([rec['labels'] for rec in records])
    label_sums = label_counts.sum(axis=0)

    FINDINGS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
                "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass", "Nodule",
                "Pleural_Thickening", "Pneumonia", "Pneumothorax"]

    plt.figure(figsize=(12,4))
    plt.bar(FINDINGS, label_sums)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Cases")
    plt.title("Label Frequencies")
    plt.tight_layout()
    plt.show()

    # Count number of labels per image
    num_labels_per_sample = label_counts.sum(axis=1)
    print("Average labels/image:", num_labels_per_sample.mean())
    print("Max labels on one image:", num_labels_per_sample.max())

    plt.hist(num_labels_per_sample, bins=np.arange(8), rwidth=0.8)
    plt.xlabel("Number of labels per image")
    plt.ylabel("Frequency")
    plt.title("Multi-label distribution")

    # Check for shared reports
    report_map = defaultdict(list)
    for rec in records:
        report_map[rec['report_text']].append(rec['id'])

    shared = [v for v in report_map.values() if len(v) > 1]
    print(f"→ Unique reports: {len(report_map)}")
    print(f"→ Reports shared by multiple images: {len(shared)}")
    print(f"→ Avg images per reused report: {np.mean([len(v) for v in shared]):.2f}")

    # Debug DICOM
    dp = DICOMImagePreprocessor()
    sample = records[0]
    arr = dp.load_raw_array(sample['dicom_path'])

    plt.imshow(arr, cmap="gray")
    plt.title("DICOM Image")
    plt.axis("off")
    plt.show()

    print("--- Report ---")
    print(sample["report_text"])
    print("--- Labels ---")
    print(dict(zip(FINDINGS, sample["labels"])))