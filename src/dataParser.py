import sys
import os
import glob
import xml.etree.ElementTree as ET
import spacy
from spacy.matcher import PhraseMatcher
from negspacy.negation import Negex
from pathlib import Path
import subprocess
import shutil
from labeledData import disease_groups, normal_groups

BASE_DIR    = Path(__file__).resolve().parent.parent
MODEL_PLACE = BASE_DIR / "models"
MODEL_PLACE.mkdir(exist_ok=True)

combined_groups = {
    **disease_groups,
    **normal_groups
}

# If the model isn’t already in models/, download it there
try:
    spacy.load("en_core_sci_sm")
except OSError:
    print("[INFO] en_core_sci_sm not found, downloading...")
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz"
    ], check=True)

# Load the model
import spacy
nlp = spacy.load("en_core_sci_sm")
print("[INFO] Loaded model from:", nlp.path)

# ----------------------------
# Set up SpaCy + Negex
# ----------------------------
nlp = spacy.load("en_core_sci_sm")
nlp.add_pipe("negex", config={"ent_types": ["MATCH"]})

# ----------------------------
# Build a PhraseMatcher for all disease keywords
# ----------------------------
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
keyword_to_group = {}  # flat lookup
for group, kws in combined_groups.items():
    for kw in kws:
        doc = nlp.make_doc(kw)
        matcher.add(group, [doc])
        keyword_to_group[kw.lower()] = group

# ----------------------------
# Build a keyword to disease_group map
# ----------------------------
keyword_to_group = {
    kw.lower(): group
    for group, kws in combined_groups.items()
    for kw in kws
}

def label_report_diseases(text):
    """
    Label a given radiology report text with the disease groups and normal groups in `combined_groups`.

    Parameters
    ----------
    text : str
        The radiology report text to label.

    Returns
    -------
    dict of str to int
        A dictionary mapping each disease group to 1 if the group is present in
        the text or 0 if it is not.
    """
    doc = nlp(text)
    # initialize all groups to 0
    labels = {group: 0 for group in combined_groups}

    matches = matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        group = nlp.vocab.strings[match_id]  # the group name we added
        # negspacy marks span._.negex = True if it's negated
        if not span._.negex:
            labels[group] = 1

    return labels

def label_vector(text):
    """
    Return a binary label vector corresponding to the disease groups in `text`

    The vector is ordered by the sorted keys of `disease_groups` and `normal_groups` and contains
    binary values (0 or 1) indicating the presence or absence of each group in
    the text.

    Parameters
    ----------
    text : str
        The text to be analyzed

    Returns
    -------
    list of int
        A binary label vector
    """
    ordered = sorted(combined_groups.keys())
    lbls = label_report_diseases(text)
    return [lbls[g] for g in ordered]

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

            # Label diseases
            vec = label_vector(report)

            ordered_labels = sorted(combined_groups.keys())
            normal_idx = ordered_labels.index("Normal")

            is_normal = vec[normal_idx] == 1 and sum(vec) == 1
            is_abnormal = any(vec[i] for i in range(len(vec)) if i != normal_idx)

            records.append({
                'id': image_id,
                'dicom_path': dcm_path,
                'report_text': report,
                'labels': vec,
                'is_normal': is_normal,
                'is_abnormal': is_abnormal
            })

    print(f"[INFO] Loaded {len(records)} records.")
    return records
