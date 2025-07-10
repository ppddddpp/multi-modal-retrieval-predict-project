import os
import glob
import xml.etree.ElementTree as ET
import re

# ----------------------------
# Rule-based CheXpert-style labeler
# ----------------------------
FINDINGS = {
    "Atelectasis": ["atelectasis"],
    "Cardiomegaly": ["cardiomegaly", "enlarged cardiac", "cardiac enlargement"],
    "Consolidation": ["consolidation"],
    "Edema": ["edema", "interstitial fluid"],
    "Effusion": ["pleural effusion", "effusion"],
    "Emphysema": ["emphysema"],
    "Fibrosis": ["fibrosis"],
    "Hernia": ["hernia"],
    "Infiltration": ["infiltrate", "infiltration"],
    "Mass": ["mass"],
    "Nodule": ["nodule"],
    "Pleural_Thickening": ["pleural thickening"],
    "Pneumonia": ["pneumonia"],
    "Pneumothorax": ["pneumothorax"]
}

NEGATIONS = [
    r'no (evidence of|sign of|indication of)?\s*{}',
    r'without (any )?{}',
    r'negative for {}',
    r'{} is absent',
    r'absence of {}'
]

def label_report(text):
    """
    Labels a given radiology report text with the 14 findings defined in
    FINDINGS. The function returns a list of 14 binary labels, corresponding
    to the order of the findings in FINDINGS. The value of each label is 1 if
    the finding is present in the report and 0 if it is not. The function is
    case-insensitive.

    Parameters
    ----------
    text : str
        The radiology report text to label.

    Returns
    -------
    labels : list of int
        The 14 binary labels corresponding to the findings in FINDINGS.
    """
    labels = {}
    text = text.lower()
    for finding, keywords in FINDINGS.items():
        found = 0
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', text):
                # check for negation
                is_negated = any(re.search(pattern.format(re.escape(kw)), text) for pattern in NEGATIONS)
                if not is_negated:
                    found = 1
                    break
        labels[finding] = found
    return list(labels.values())  # 14 labels

def parse_openi_xml(xml_dir, dicom_root):
    """
    Parse OpenI XML reports and match to corresponding DICOM files

    Parameters
    ----------
    xml_dir : str
        Path to folder containing individual .xml report files
    dicom_root : str
        Root folder where .dcm files live (possibly nested)

    Returns
    -------
    records : list of dicts
        Each dict has: 'id', 'dicom_path', 'report_text', 'labels' (14-dim vector)
    """
    all_dcms = glob.glob(os.path.join(dicom_root, '**', '*.dcm'), recursive=True)
    dcm_map = {os.path.splitext(os.path.basename(p))[0]: p for p in all_dcms}

    print(f"[INFO] Found {len(os.listdir(xml_dir))} XML files in {xml_dir}")
    print(f"[INFO] Found {len(all_dcms)} DICOM files in {dicom_root}")

    records = []

    for fname in os.listdir(xml_dir):
        if not fname.endswith('.xml'):
            continue

        xml_path = os.path.join(xml_dir, fname)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for img_tag in root.findall('parentImage'):
            raw_id = img_tag.attrib.get('id')  # e.g. CXR3_1_IM-1384-2001
            if not raw_id:
                continue

            # Normalize ID and match to DICOM
            if raw_id.startswith("CXR") and "_" in raw_id:
                parts = raw_id[3:].split("_", 1)
                if len(parts) == 2:
                    image_id = parts[0] + "_" + parts[1]
                else:
                    continue
            else:
                continue

            dcm_path = dcm_map.get(image_id)
            if not dcm_path:
                continue

            # Extract full report text
            abstract_parts = [n.text.strip() for n in root.findall('.//AbstractText') if n.text]
            if not abstract_parts:
                title = root.findtext('.//ArticleTitle') or ""
                abstract_parts = [title.strip()]
            report = " ".join(abstract_parts)

            # Generate 14-dim label vector
            label_vector = label_report(report)

            records.append({
                'id': image_id,
                'dicom_path': dcm_path,
                'report_text': report,
                'labels': label_vector
            })

    print(f"[INFO] Loaded {len(records)} records.")
    return records
