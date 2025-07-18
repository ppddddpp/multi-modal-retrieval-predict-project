from pathlib import Path
from collections import Counter, defaultdict
import xml.etree.ElementTree as ET
from labeledData import disease_groups
from labeledData import device_groups
from labeledData import technical_groups
from labeledData import normal_groups
from labeledData import anatomy_groups

if __name__ == "__main__":
    # Point to the real directory (relative to project root)
    try:
        BASE_DIR = Path(__file__).resolve().parent.parent
    except NameError:
        BASE_DIR = Path.cwd().parent
    
    xml_dir = BASE_DIR / 'data' / 'openi' / 'xml' / 'NLMCXR_reports' / 'ecgen-radiology'

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