from pathlib import Path
import json, csv, re
from typing import List, Optional, Tuple, Dict
from DataHandler import parse_openi_xml
from LabelData import disease_groups, normal_groups, finding_groups, symptom_groups
from Helpers import _sanitize_node

BASE_DIR = Path(__file__).resolve().parent.parent.parent
KG_DIR = BASE_DIR / "knowledge_graph"
ONTO_DIR = BASE_DIR / "data" / "ontologies"
KG_DIR.mkdir(parents=True, exist_ok=True)


class KGBuilder:
    """
    Build node/relation vocabularies and triples CSV for training KG embeddings.
    Supports:
      - Dataset-only (OpenI)
      - Ontology-only (RadLex / DOID)
      - Hybrid (dataset + mappings + ontology)
    """

    def __init__(self, out_dir: str = "knowledge_graph", combined_groups: Optional[Dict[str, str]] = None):
        """
        Initialize KGBuilder with output directory and optional combined group labels.

        Args:
            out_dir (str): Output directory for triples.csv and node/relation vocabularies (default: "knowledge_graph")
            combined_groups (Optional[Dict[str, str]]): Optional dictionary of combined group labels (default: all groups combined)
        """
        self.out_dir = Path(out_dir) if Path(out_dir).is_absolute() else (BASE_DIR / out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.triples: List[Tuple[str, str, str, float, str]] = []
        self.node2id: Dict[str, int] = {}
        self.relation2id: Dict[str, int] = {}
        self.entity_meta: Dict[int, str] = {}

        self.combined_groups = {**disease_groups, **normal_groups, **finding_groups, **symptom_groups} \
                                if combined_groups is None else combined_groups
        self.label_names = sorted(self.combined_groups.keys())

    # ---------- dataset-driven triples ----------
    def build_from_parsed(self, xml_dir: str, dicom_root: str, min_label_conf: float = 0.0):
        records = parse_openi_xml(xml_dir, dicom_root, combined_groups=self.combined_groups)
        for r in records:
            rid = f"report:{r['id']}"
            iid = f"image:{r['id']}"
            self.triples.append((rid, "REPORT_OF", iid, 1.0, "extracted"))

            vec = r.get("labels", [])
            for idx, v in enumerate(vec):
                if v:
                    label = self.label_names[idx]
                    lbl_node = f"label:{_sanitize_node(label)}"
                    self.triples.append((rid, "HAS_LABEL", lbl_node, 1.0, "extracted"))

    # ---------- ontology mapping triples ----------
    def add_ontology_mapping(self, mapping_path: Path):
        if not mapping_path.exists():
            print(f"[KGBuilder] Mapping file not found: {mapping_path}")
            return
        mapping = json.loads(mapping_path.read_text(encoding="utf8"))
        for group, labels in mapping.items():
            for label, ont in labels.items():
                lbl_node = f"label:{_sanitize_node(label)}"
                if ont and not str(ont).startswith("LOCAL:"):
                    ont_node = f"onto:{_sanitize_node(ont)}"
                    self.triples.append((lbl_node, "MAPPED_TO", ont_node, 1.0, "mapping"))

    # ---------- ontology-native triples ----------
    def add_doid(self, doid_path: Path):
        if not doid_path.exists():
            print(f"[KGBuilder] DOID not found: {doid_path}")
            return
        with doid_path.open(encoding="utf8") as f:
            cur_id = None
            for line in f:
                line = line.strip()
                if line == "[Term]":
                    cur_id = None
                    continue
                if line.startswith("id: DOID:"):
                    cur_id = line.split("id: ")[1]
                elif line.startswith("is_a:"):
                    parent = line.split("is_a: ")[1].split()[0]
                    if cur_id and parent:
                        self.triples.append((f"doid:{cur_id}", "is_a", f"doid:{parent}", 1.0, "doid"))

    def add_radlex(self, radlex_path: Path):
        if not radlex_path.exists():
            print(f"[KGBuilder] RadLex not found: {radlex_path}")
            return
        text = radlex_path.read_text(encoding="utf8")
        for sub, parent in re.findall(r'rdf:about=".*?(RID\d+)".*?<rdfs:subClassOf rdf:resource=".*?(RID\d+)"', text, flags=re.S):
            self.triples.append((f"radlex:{sub}", "is_a", f"radlex:{parent}", 1.0, "radlex"))

    # ---------- curated CSV ----------
    def add_curated_csv(self, csv_path: str):
        p = Path(csv_path)
        if not p.exists():
            raise FileNotFoundError(csv_path)
        with p.open(newline='', encoding='utf8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                s = row['s']; r = row['r']; o = row['o']
                conf = float(row.get('confidence', 1.0))
                self.triples.append((s, r, o, conf, "curated"))

    # ---------- vocab building ----------
    def _ensure_vocab(self):
        next_n, next_r = 0, 0
        for s, r, o, conf, src in self.triples:
            for ent in (s, o):
                if ent not in self.node2id:
                    self.node2id[ent] = next_n
                    self.entity_meta[next_n] = ent
                    next_n += 1
            if r not in self.relation2id:
                self.relation2id[r] = next_r
                next_r += 1

    # ---------- save ----------
    def save(self):
        self._ensure_vocab()
        tpath = self.out_dir / "triples.csv"
        with tpath.open('w', newline='', encoding='utf8') as f:
            writer = csv.writer(f)
            writer.writerow(['s_id', 'r_id', 'o_id', 'confidence', 'source'])
            for s, r, o, conf, src in self.triples:
                writer.writerow([self.node2id[s], self.relation2id[r], self.node2id[o], conf, src])

        (self.out_dir / "node2id.json").write_text(json.dumps(self.node2id, indent=2, ensure_ascii=False))
        (self.out_dir / "relation2id.json").write_text(json.dumps(self.relation2id, indent=2, ensure_ascii=False))
        (self.out_dir / "entity_meta.json").write_text(json.dumps(self.entity_meta, indent=2, ensure_ascii=False))
        print(f"[KGBuilder] wrote: {tpath} and maps to {self.out_dir}")

    # ---------- build entrypoint ----------
    def build(self, xml_dir: Optional[str] = None, dicom_root: Optional[str] = None,
              curated_csv: Optional[str] = None, mode: str = "hybrid"):
        """
        mode: "dataset", "ontology", "hybrid"
        """
        if mode == "dataset":
            self.build_from_parsed(xml_dir, dicom_root)

        elif mode == "ontology":
            self.add_doid(ONTO_DIR / "doid.obo")
            self.add_radlex(ONTO_DIR / "RadLex.owl")

        elif mode == "hybrid":
            self.build_from_parsed(xml_dir, dicom_root)
            for mp in ["disease_label2ontology.json", "finding_label2ontology.json",
                       "normal_label2ontology.json", "symptom_label2ontology.json"]:
                self.add_ontology_mapping(ONTO_DIR / mp)
            self.add_doid(ONTO_DIR / "doid.obo")
            self.add_radlex(ONTO_DIR / "RadLex.owl")
        else:
            raise ValueError(f"Unknown KG mode: {mode}")

        if curated_csv:
            self.add_curated_csv(curated_csv)

        self.save()

    # ---------- ensure triples exist ----------
    @classmethod
    def ensure_exists(cls, xml_dir: Optional[str] = None, dicom_root: Optional[str] = None,
                      curated_csv: Optional[str] = None, mode: str = "dataset"):
        tpath = KG_DIR / "triples.csv"
        if not tpath.exists():
            print(f"[KGBuilder] triples.csv not found, building KG in mode={mode}...")
            builder = cls()
            builder.build(xml_dir=xml_dir, dicom_root=dicom_root,
                          curated_csv=curated_csv, mode=mode)
        else:
            print(f"[KGBuilder] triples.csv already exists -> using cached KG (mode={mode})")
