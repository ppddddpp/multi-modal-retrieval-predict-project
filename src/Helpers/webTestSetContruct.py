import json
import shutil
import random
from pathlib import Path
from DataHandler import parse_openi_xml

BASE_DIR = Path(__file__).resolve().parent.parent.parent
XML_DIR = BASE_DIR / "data/openi/xml/NLMCXR_reports/ecgen-radiology"
DICOM_DIR = BASE_DIR / "data/openi/dicom"
SPLIT_DIR = BASE_DIR / "splited_data"
OUTPUT_DIR = BASE_DIR / "web_test_set"

def create_test_set_for_web(xml_dir=XML_DIR, dicom_dir=DICOM_DIR, split_dir=SPLIT_DIR, output_dir=OUTPUT_DIR, combined_groups=None, num_samples=10):
    """
    Creates a test set for the web application by sampling N records from the test split IDs.

    Parameters
    ----------
    xml_dir : str
        Path to folder containing individual .xml report files
    dicom_dir : str
        Root folder where .dcm files live (possibly nested)
    split_dir : str
        Path to folder containing train, validation, and test split IDs
    output_dir : str
        Path to save the test set to
    combined_groups : dict of str to list of str
        Dictionary where keys are disease/normal group names and values are lists of labels
    num_samples : int
        Number of test samples to generate

    Returns
    -------
    None

    Saves a JSON file with the following format: {<rid>: {<dicom_path>, <report_text>}} to the output directory
    """
    # Load test IDs
    with open(split_dir / "test_split_ids.json") as f:
        test_ids = json.load(f)

    if combined_groups is None:
        raise ValueError("Please provide a least a list of disease groups and normal groups to label the report with.")

    # Parse all records
    records = parse_openi_xml(xml_dir, dicom_dir, combined_groups=combined_groups)
    record_map = {r["id"]: r for r in records}

    # Choose N samples
    sampled_ids = random.sample(test_ids, num_samples)
    meta_info = {}
    output_dir = OUTPUT_DIR if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dcm = output_dir / "dicom"
    output_dcm.mkdir(parents=True, exist_ok=True)
    output_reports = output_dir / "reports"
    output_reports.mkdir(parents=True, exist_ok=True)

    for rid in sampled_ids:
        if rid not in record_map:
            continue
        rec = record_map[rid]
        dicom_src = rec["dicom_path"]
        report_text = rec["report_text"]

        # Copy DICOM file
        dicom_dst = output_dcm / f"{rid}.dcm"
        shutil.copy2(dicom_src, dicom_dst)

        # Save report text
        with open(output_reports / f"{rid}.txt", "w", encoding="utf-8") as f:
            f.write(report_text.strip())

        meta_info[rid] = {
            "dicom": str(dicom_dst.name),
            "report": str(f"{rid}.txt")
        }

    # Save metadata
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta_info, f, indent=2)

    print(f"Saved {len(meta_info)} test samples to: {output_dir}")

if __name__ == "__main__":
    create_test_set_for_web()