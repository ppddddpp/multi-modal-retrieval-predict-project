import pandas as pd
from dataParser import parse_openi_xml
from labeledData import disease_groups, normal_groups
from pathlib import Path
import os
import numpy as np

# Resolve paths...
try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path.cwd().parent

XML_DIR    = BASE_DIR / 'data' / 'openi' / 'xml' / 'NLMCXR_reports' / 'ecgen-radiology'
DICOM_ROOT = BASE_DIR / 'data' / 'openi' / 'dicom'
MODEL_PLACE = BASE_DIR / "models"

if __name__ == '__main__':
    combined_groups = {
        **disease_groups,
        **normal_groups
    }
    label_names  = sorted(combined_groups.keys())
    records      = parse_openi_xml(XML_DIR, DICOM_ROOT)
    
    # Build label matrix
    label_matrix = np.array([rec['labels'] for rec in records])

    # Build a pandas DataFrame
    df = pd.DataFrame(label_matrix, columns=label_names)
    df.insert(0, 'report_text', [rec['report_text'] for rec in records])
    df.insert(0, 'id',          [rec['id']          for rec in records])

    # Save to CSV
    out_path = BASE_DIR / 'outputs' / 'openi_labels.csv'
    out_path.parent.mkdir(exist_ok=True)
    df.to_csv(out_path, index=False, encoding='utf-8')

    print(f"Wrote {len(df)} rows with {len(label_names)} labels to {out_path}")
