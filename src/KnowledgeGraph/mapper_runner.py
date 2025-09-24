from pathlib import Path
import sys
try:
    base = Path(__file__).resolve().parent.parent
except NameError:
    base = Path.cwd().parent
sys.path.append(str(base))
from ontology_mapper import OntologyMapper
from LabelData import disease_groups, normal_groups, finding_groups, symptom_groups
from Helpers import log_and_print

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ONTO_DIR = BASE_DIR / 'data' / "ontologies"
LOG_DIR = BASE_DIR / 'mapped' 

mapper = OntologyMapper(use_doid=True, use_radlex=True, use_bioportal=True)
log_file = LOG_DIR / 'mapped_log.txt'

def print_unmapped(nested_mapping, title="", log_file=None):
    """Print unmapped labels per group, one per line."""
    log_and_print(f"\n=== Unmapped Labels Report ({title}) ===", log_file=log_file)
    for group, mapping in nested_mapping.items():
        unmapped = [lbl for lbl, oid in mapping.items()
                    if oid is None or str(oid).startswith("LOCAL:")]
        if unmapped:
            log_and_print(f"\n{group} ({len(unmapped)} unmapped):", log_file=log_file)
            for lbl in unmapped:
                log_and_print(f"  - {lbl}", log_file=log_file)

# ------------------ Diseases ------------------
nested_disease = mapper.map_grouped_labels(disease_groups)
mapper.report_group_coverage(nested_disease)
mapper.save_mapping(nested_disease, ONTO_DIR / "disease_label2ontology.json")
print_unmapped(nested_disease, "Disease Groups", log_file=log_file)

nested_finding = mapper.map_grouped_labels(finding_groups)
mapper.report_group_coverage(nested_finding)
mapper.save_mapping(nested_finding, ONTO_DIR / "finding_label2ontology.json")
print_unmapped(nested_finding, "Finding Groups", log_file=log_file)

nested_normal = mapper.map_grouped_labels(normal_groups)
mapper.report_group_coverage(nested_normal)
mapper.save_mapping(nested_normal, ONTO_DIR / "normal_label2ontology.json")
print_unmapped(nested_normal, "Normal Groups", log_file=log_file)

nested_symptom = mapper.map_grouped_labels(symptom_groups)
mapper.report_group_coverage(nested_symptom)
mapper.save_mapping(nested_symptom, ONTO_DIR / "symptom_label2ontology.json")
print_unmapped(nested_symptom, "Symptom Groups", log_file=log_file)