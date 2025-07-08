import os
import xml.etree.ElementTree as ET
import glob

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
    records : list
        List of dicts where each dict has the keys 'id', 'dicom_path', 'report_text', and 'mesh_labels'
    """
    all_dcms = glob.glob(os.path.join(dicom_root, '**', '*.dcm'), recursive=True)
    dcm_map = { os.path.splitext(os.path.basename(p))[0]: p for p in all_dcms }

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
            raw_id = img_tag.attrib.get('id')
            if not raw_id:
                continue

            # --- Normalize ID ---
            image_id = raw_id.replace("CXR", "")
            parts = image_id.split('_', 1)
            if len(parts) == 2:
                image_id = parts[1]
            else:
                continue

            # Match to DICOM file
            dcm_path = dcm_map.get(image_id)
            if not dcm_path:
                continue

            # Extract report text
            report = root.findtext('AbstractText') or ""
            report = report.strip()

            # MeSH terms
            mesh = [m.text for m in root.findall('.//MeshHeading/DescriptorName')]

            records.append({
                'id':          image_id,
                'dicom_path':  dcm_path,
                'report_text': report,
                'mesh_labels': mesh
            })

    print(f"[INFO] Loaded {len(records)} records.")
    return records
