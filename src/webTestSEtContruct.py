import os
import json
import shutil
import random
from pathlib import Path
import pydicom
from dataParser import parse_openi_xml

BASE_DIR = Path(__file__).resolve().parent.parent
XML_DIR = BASE_DIR / "data/openi/xml/NLMCXR_reports/ecgen-radiology"
DICOM_DIR = BASE_DIR / "data/openi/dicom"
SPLIT_DIR = BASE_DIR / "splited_data"
OUTPUT_DIR = BASE_DIR / "web_test_set"
OUTPUT_DICOM = OUTPUT_DIR / "dicom"
OUTPUT_REPORTS = OUTPUT_DIR / "reports"

OUTPUT_DICOM.mkdir(parents=True, exist_ok=True)
OUTPUT_REPORTS.mkdir(parents=True, exist_ok=True)

def main():
    # Load test IDs
    with open(SPLIT_DIR / "test_split_ids.json") as f:
        test_ids = json.load(f)

    # Parse all records
    records = parse_openi_xml(XML_DIR, DICOM_DIR)
    record_map = {r["id"]: r for r in records}

    # Choose N samples
    N = 10
    sampled_ids = random.sample(test_ids, N)
    meta_info = {}

    for rid in sampled_ids:
        if rid not in record_map:
            continue
        rec = record_map[rid]
        dicom_src = rec["dicom_path"]
        report_text = rec["report_text"]

        # Copy DICOM file
        dicom_dst = OUTPUT_DICOM / f"{rid}.dcm"
        shutil.copy2(dicom_src, dicom_dst)

        # Save report text
        with open(OUTPUT_REPORTS / f"{rid}.txt", "w", encoding="utf-8") as f:
            f.write(report_text.strip())

        meta_info[rid] = {
            "dicom": str(dicom_dst.name),
            "report": str(f"{rid}.txt")
        }

    # Save metadata
    with open(OUTPUT_DIR / "meta.json", "w") as f:
        json.dump(meta_info, f, indent=2)

    print(f"Saved {len(meta_info)} test samples to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()