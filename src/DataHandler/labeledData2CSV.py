from pathlib import Path
import sys
try:
    base = Path(__file__).resolve().parent.parent
except NameError:
    base = Path.cwd().parent
sys.path.append(str(base))
import pandas as pd
from .dataParser import parse_openi_xml
from pathlib import Path
import numpy as np
from LabelData import disease_groups, normal_groups, finding_groups, symptom_groups

# Resolve paths...
try:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
except NameError:
    BASE_DIR = Path.cwd().parent

XML_DIR    = BASE_DIR / 'data' / 'openi' / 'xml' / 'NLMCXR_reports' / 'ecgen-radiology'
DICOM_ROOT = BASE_DIR / 'data' / 'openi' / 'dicom'
MODEL_PLACE = BASE_DIR / "models"

def label2CSV(xml_dir=None, dicom_dir=None, out_path=None, combined_groups=None):
    """
    Convert parsed OpenI XML reports to a CSV file with labels.

    Parameters
    ----------
    xml_dir : str
        Path to folder containing individual .xml report files
    dicom_dir : str
        Root folder where .dcm files live (possibly nested)
    out_path : str
        Path to save the output CSV file to
    combined_groups : dict of str to list of str
        Dictionary where keys are disease/normal group names and values are lists of labels

    Returns
    -------
    None

    Saves a CSV file with the following columns: ['id', 'report_text'] + label_names
    """
    if xml_dir is None:
        xml_dir = XML_DIR
    if dicom_dir is None:
        dicom_dir = DICOM_ROOT
    if combined_groups is None:
        raise ValueError("Please provide a least a list of disease groups and normal groups to label the report with.")
    
    label_names  = sorted(combined_groups.keys())
    records      = parse_openi_xml(xml_dir=xml_dir, dicom_root=dicom_dir, combined_groups=combined_groups)
    
    # Build label matrix
    label_matrix = np.array([rec['labels'] for rec in records])

    # Build a pandas DataFrame
    df = pd.DataFrame(label_matrix, columns=label_names)
    df.insert(0, 'report_text', [rec['report_text'] for rec in records])
    df.insert(0, 'id',          [rec['id']          for rec in records])

    # Save to CSV
    out_path = BASE_DIR / 'outputs' / 'openi_labels.csv' if out_path is None else out_path
    out_path.parent.mkdir(exist_ok=True)
    df.to_csv(out_path, index=False, encoding='utf-8')

    print(f"Wrote {len(df)} rows with {len(label_names)} labels to {out_path}")

if __name__ == "__main__":
    combined_groups = {
        **disease_groups,
        **finding_groups,
        **symptom_groups,
        **normal_groups
    }
    label2CSV(combined_groups=combined_groups)