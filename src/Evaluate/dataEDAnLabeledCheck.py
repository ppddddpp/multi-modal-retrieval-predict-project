from pathlib import Path
import sys
try:
    base = Path(__file__).resolve().parent.parent
except NameError:
    base = Path.cwd().parent
sys.path.append(str(base))
from collections import Counter, defaultdict
import xml.etree.ElementTree as ET
from LabelData import disease_groups, device_groups, finding_groups, symptom_groups, technical_groups, normal_groups, anatomy_groups
import pandas as pd
import matplotlib.pyplot as plt

try:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
except NameError:
    BASE_DIR = Path.cwd().parent.parent

def edaLabeledCheck(xml_dir=None, 
                    save_dir=None, 
                    min_count_threshold=100):
    """
    This function checks the XML files in the given directory for MeSH labels,
    and outputs statistics on the number of MeSH labels, the number of
    unmapped MeSH labels, and the number of duplicate MeSH labels.

    Args:
        xml_dir (str): The directory containing the XML files to check.
        save_dir (str): The directory to save the output CSV file.
        min_count_threshold (int): The minimum count for a MeSH label to be plotted.

    Returns:
        None
    """
    # Point to the real directory (relative to project root)
    xml_dir = BASE_DIR / 'data' / 'openi' / 'xml' / 'NLMCXR_reports' / 'ecgen-radiology' if xml_dir is None else xml_dir

    files = list(xml_dir.glob("*.xml"))
    print("\nFound XML files:", len(files))

    existing_ids = {int(p.stem) for p in xml_dir.glob("*.xml")}
    expected_ids = set(range(1, 4000))
    missing_ids = sorted(expected_ids - existing_ids)
    print("\nMissing XML files:", missing_ids)
    print("\nCount missing:", len(missing_ids))

    mesh_counter = Counter()
    for fn in files:
        root = ET.parse(fn).getroot()
        for term_node in root.findall('.//MeSH/*'):
            raw = term_node.text or ""
            label = raw.split('/')[0].strip().lower()
            if label:
                mesh_counter[label] += 1

    print("\nUnique MeSH labels:", len(mesh_counter))
    print(mesh_counter)

    mesh_terms = list(mesh_counter.keys())
    print("\nUnique MeSH labels:", len(mesh_terms))

    print("\nMeSH terms:")
    print(mesh_terms)

    # Put all of individual group‑dicts into one single dict
    all_groups = {
        **disease_groups,
        **finding_groups,
        **symptom_groups,
        **device_groups,
        **technical_groups,
        **normal_groups,
        **anatomy_groups
    }

    # Normalize both mesh_terms and all group terms to lowercase
    mesh_terms_lc = [t.strip().lower() for t in mesh_terms]
    all_groups_lc = {
        grp: [t.strip().lower() for t in terms]
        for grp, terms in all_groups.items()
    }

    # Build reverse map: term → list of groups it belongs to
    reverse_map = defaultdict(list)
    for grp, terms in all_groups_lc.items():
        for term in terms:
            reverse_map[term].append(grp)

    # Find unmapped MeSH terms
    unmapped = [t for t in mesh_terms_lc if t not in reverse_map]

    # Find duplicates in the original mesh_terms
    mesh_dups = {t for t in mesh_terms_lc if mesh_terms_lc.count(t) > 1}

    # Find duplicates across all labels (terms assigned more than once to different groups)
    labeled_dups = {t for t, grps in reverse_map.items() if len(grps) > 1}

    # Find “extra” labels (terms in your groups that aren’t in mesh_terms)
    extra_labeled = [t for t in reverse_map if t not in mesh_terms_lc]

    # Summary printout
    print(f"\nTotal MeSH terms           : {len(set(mesh_terms_lc))}")
    print(f"\nLabeled MeSH terms         : {len(set(mesh_terms_lc) - set(unmapped))}")
    print(f"\nUnmapped MeSH terms        : {len(unmapped)}")
    print(f"{unmapped}")
    print(f"\nDuplicate in mesh_terms    : {mesh_dups or 'None'}")
    print(f"\nTerms assigned to multiple groups: {labeled_dups or 'None'}")
    print(f"\nExtra labeled terms not in mesh_terms: {extra_labeled or 'None'}")

    # Detailed per-group term‑counts
    print(f"\nTotal group: {len(all_groups_lc)}")
    print("\nGroup term counts:")
    for grp, terms in all_groups_lc.items():
        print(f" - {grp}: {len(terms)}")

    # Detailed per-diseases term‑counts
    print(f"\nTotal diseases group: {len(disease_groups)}")
    print("\nGroup term counts:")
    for grp, terms in disease_groups.items():
        print(f" - {grp}: {len(terms)}")

    # Detailed per-findings term‐counts
    print(f"\nTotal finding group: {len(finding_groups)}")
    print("\nGroup term counts:")
    for grp, terms in finding_groups.items():
        print(f" - {grp}: {len(terms)}")

    # Detailed per-symptoms term‐counts
    print(f"\nTotal symptoms group: {len(symptom_groups)}")
    print("\nGroup term counts:")
    for grp, terms in symptom_groups.items():
        print(f" - {grp}: {len(terms)}")

    df = pd.DataFrame(mesh_counter.items(), columns=['MeSH_Label', 'Count'])

    # Filter for labels with count >= min_count_threshold
    df_above_threshold = df[df['Count'] >= min_count_threshold].sort_values(by='Count', ascending=False)

    # Save counts to CSV
    save_dir = BASE_DIR / 'eda_data' if save_dir is None else save_dir
    csv_path = save_dir / f'mesh_labels_count_gte_{min_count_threshold}.csv' 
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Saved MeSH label counts to {csv_path}")

    # Save bar plot
    if not df[df['Count'] >= min_count_threshold].empty:
        plt.figure(figsize=(12, 8))
        plt.barh(df_above_threshold['MeSH_Label'], df_above_threshold['Count'], color='purple')
        plt.xlabel('Count')
        plt.ylabel('MeSH Label')
        plt.title(f'MeSH Labels with Count >= {min_count_threshold}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        img_path = save_dir / f'mesh_labels_count_gte_{min_count_threshold}.png'
        plt.savefig(img_path)
        plt.close()
        print(f"Saved bar chart to {img_path}")
