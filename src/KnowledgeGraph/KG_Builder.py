from pathlib import Path
import json, csv, re, os
from tqdm import tqdm
import numpy as np
from typing import List, Optional, Tuple, Dict
from DataHandler import parse_openi_xml
from Model.fusion import Backbones
from DataHandler.tensorDICOM import DICOMImagePreprocessor
import torch
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
        self.out_dir = Path(out_dir) if Path(out_dir).is_absolute() else (BASE_DIR / out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.triples: List[Tuple[str, str, str, float, str]] = []
        self.node2id: Dict[str, int] = {}
        self.relation2id: Dict[str, int] = {}
        self.entity_meta: Dict[int, dict] = {}  # final export
        self.entity_meta_raw: Dict[str, dict] = {}  # ontology metadata

        self.combined_groups = {**disease_groups, **normal_groups, **finding_groups, **symptom_groups} \
                                if combined_groups is None else combined_groups
        self.label_names = sorted(self.combined_groups.keys())

    # ---------- dataset-driven triples ----------
    def build_from_parsed(self,
                        xml_dir: str,
                        dicom_root: str,
                        min_label_conf: float = 0.0,
                        save_feats_path: str = "kg_image_feats.pt",
                        backbone_name: str = "swin_base_patch4_window7_224",
                        backbone_type: str = "swin",   # "swin" or "cnn" per Backbones constructor
                        device: str = "cpu"):
        """
        Build KG triples from parsed records and extract + save global image features.

        - xml_dir, dicom_root: as before (records should include an identifier to map to dcm file)
        - save_feats_path: path to save the feature dict (torch .pt recommended)
        - backbone_name: model name passed to Backbones
        - backbone_type: "swin" or "cnn" (Backbones uses img_backbone param)
        - device: 'cpu' or 'cuda'
        """
        device = device if isinstance(device, str) else str(device)
        device = torch.device(device if torch.cuda.is_available() and device.startswith('cuda') else 'cpu')

        print("[KGBuilder] Building triples and extracting image features...")
        records = parse_openi_xml(xml_dir, dicom_root, combined_groups=self.combined_groups)

        if backbone_type == "cnn":
            backbone_name = "resnet50"
        if backbone_type == "swin":
            backbone_name = "swin_base_patch4_window7_224"

        device = torch.device(device)
        preproc = DICOMImagePreprocessor(output_size=(224, 224), augment=False)
        backbone = Backbones(img_backbone=backbone_type, swin_model_name=backbone_name, pretrained=True)
        backbone.to(device)
        backbone.eval()

        image_feats = {}  # will hold "image:{id}" -> np.array(feature_dim,)

        with torch.no_grad():
            for r in tqdm(records, desc="Dataset records"):
                rid = f"report:{r['id']}"
                iid = f"image:{r['id']}"
                self.triples.append((rid, "REPORT_OF", iid, 1.0, "extracted"))

                # Determine DICOM path for this record (adapt if your record uses a different key)
                if "dicom_path" in r and r["dicom_path"]:
                    dcm_path = os.path.join(dicom_root, r["dicom_path"])
                else:
                    # fallback: try <id>.dcm in dicom_root
                    dcm_path = os.path.join(dicom_root, f"{r['id']}.dcm")

                # load & preprocess DICOM -> tensor (C, H, W)
                try:
                    img_tensor = preproc.load(dcm_path)  # returns torch.Tensor (C,H,W)
                except Exception as e:
                    print(f"[KGBuilder] WARNING: failed to load DICOM for id={r['id']} @ {dcm_path}: {e}")
                    continue

                # ensure batch dim and move to device
                if img_tensor.dim() == 3:
                    inp = img_tensor.unsqueeze(0).to(device)  # (1, C, H, W)
                else:
                    inp = img_tensor.to(device)

                # extract pooled global image vector using Backbones.extract_global
                try:
                    feat = backbone.extract_global(inp)  # (B, D)
                    feat = feat.squeeze(0).cpu()         # (D,)  keep as torch.Tensor
                except Exception as e:
                    # fallback: try forward() and take img_global
                    try:
                        (img_global, img_patches), _ = backbone.forward(inp)
                        feat = img_global.squeeze(0).cpu()   # keep as torch.Tensor
                    except Exception as e2:
                        print(f"[KGBuilder] ERROR: failed to extract features for {dcm_path}: {e} / {e2}")
                        continue

                image_feats[iid] = feat

                # labels handling remains unchanged
                vec = r.get("labels", [])
                for idx, v in enumerate(vec):
                    if v:
                        label = self.label_names[idx]
                        lbl_node = f"label:{_sanitize_node(label)}"
                        if label in disease_groups:
                            rel = "HAS_DISEASE"
                        elif label in symptom_groups:
                            rel = "HAS_SYMPTOM"
                        elif label in finding_groups:
                            rel = "HAS_FINDING"
                        elif label in normal_groups:
                            rel = "HAS_NORMAL"
                        else:
                            rel = "HAS_LABEL"

                        self.triples.append((rid, rel, lbl_node, 1.0, "extracted"))

        if save_feats_path is None:
            save_feats_path = "kg_image_feats.pt"
        save_path = Path(save_feats_path)
        if not save_path.is_absolute():
            save_path = self.out_dir / save_path

        # Save features
        try:
            # ensure parent dir exists
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(image_feats, str(save_path))
            print(f"[KGBuilder] Saved {len(image_feats)} image features to {save_path}")
        except Exception as e:
            # fallback to numpy (pickled) at same target folder
            np_save_path = save_path.with_suffix(".npy")
            try:
                np.save(np_save_path, image_feats, allow_pickle=True)
                print(f"[KGBuilder] torch.save failed ({e}). Saved features to {np_save_path}")
            except Exception as e2:
                print(f"[KGBuilder] Failed to save image features to {np_save_path}: {e2}")

        print(f"[KGBuilder] Dataset triples added: {len(self.triples)}")

    # ---------- ontology mapping triples ----------
    def add_ontology_mapping(self, mapping_path: Path):
        if not mapping_path.exists():
            print(f"[KGBuilder] Mapping file not found: {mapping_path}")
            return
        mapping = json.loads(mapping_path.read_text(encoding="utf8"))

        for group, labels in tqdm(mapping.items(), desc=f"Ontology mapping {mapping_path.name}"):
            for label, ont in labels.items():
                lbl_node = f"label:{_sanitize_node(label)}"
                if ont and not str(ont).startswith("LOCAL:"):

                    # normalize ontology node name
                    if str(ont).startswith("DOID"):
                        ont_node = f"doid:{ont.replace('DOID:', '')}"
                        rel = "MAPPED_TO_DOID"
                    elif str(ont).startswith("RADLEX") or str(ont).startswith("RID"):
                        rid = ont.replace("RADLEX:", "").replace("RID", "")
                        ont_node = f"radlex:RID{rid}"
                        rel = "MAPPED_TO_RADLEX"
                    else:
                        ont_node = f"onto:{_sanitize_node(str(ont))}"
                        rel = "MAPPED_TO"

                    # base triple
                    self.triples.append((lbl_node, rel, ont_node, 1.0, "mapping"))

                    # add synonym links if available in ontology metadata
                    if ont_node in self.entity_meta_raw:
                        synonyms = self.entity_meta_raw[ont_node].get("synonyms", [])
                        for syn in synonyms:
                            syn_node = f"syn:{_sanitize_node(syn)}"
                            self.triples.append((syn_node, "SYNONYM_OF", ont_node, 1.0, "mapping"))

    # ---------- ontology-native triples ----------
    def add_doid(self, doid_path: Path):
        if not doid_path.exists():
            print(f"[KGBuilder] DOID not found: {doid_path}")
            return

        print(f"[KGBuilder] Parsing DOID from {doid_path.name}...")
        with doid_path.open(encoding="utf8") as f:
            lines = f.readlines()

        cur_id, cur_label, cur_syns, cur_xrefs, cur_def = None, None, [], [], None
        for line in tqdm(lines, desc="DOID terms"):
            line = line.strip()

            if line == "[Term]":
                if cur_id:
                    node = f"doid:{cur_id}"
                    self.entity_meta_raw[node] = {
                        "id": node, "label": cur_label,
                        "synonyms": cur_syns, "xrefs": cur_xrefs, "definition": cur_def
                    }
                    for syn in cur_syns:
                        syn_node = f"syn:{_sanitize_node(syn)}"
                        self.triples.append((syn_node, "SYNONYM_OF", node, 1.0, "doid"))
                    for xref in cur_xrefs:
                        xref_node = f"xref:{_sanitize_node(xref)}"
                        self.triples.append((node, "XREF", xref_node, 1.0, "doid"))
                cur_id, cur_label, cur_syns, cur_xrefs, cur_def = None, None, [], [], None
                continue

            if line.startswith("id: DOID:"):
                cur_id = line.split("id: ")[1]
            elif line.startswith("name:"):
                cur_label = line.split("name: ")[1]
            elif line.startswith("synonym:"):
                m = re.search(r"\"(.*?)\"", line)
                if m:
                    cur_syns.append(m.group(1))
            elif line.startswith("xref:"):
                cur_xrefs.append(line.split("xref: ")[1])
            elif line.startswith("def:"):
                m = re.search(r"\"(.*?)\"", line)
                if m:
                    cur_def = m.group(1)
            elif line.startswith("is_a:"):
                parent = line.split("is_a: ")[1].split()[0]
                if cur_id and parent:
                    self.triples.append((f"doid:{cur_id}", "is_a", f"doid:{parent}", 1.0, "doid"))

        # save last term if not already flushed
        if cur_id:
            node = f"doid:{cur_id}"
            self.entity_meta_raw[node] = {
                "id": node, "label": cur_label,
                "synonyms": cur_syns, "xrefs": cur_xrefs, "definition": cur_def
            }
            for syn in cur_syns:
                syn_node = f"syn:{_sanitize_node(syn)}"
                self.triples.append((syn_node, "SYNONYM_OF", node, 1.0, "doid"))
            for xref in cur_xrefs:
                xref_node = f"xref:{_sanitize_node(xref)}"
                self.triples.append((node, "XREF", xref_node, 1.0, "doid"))

        print(f"[KGBuilder] DOID triples added: {len(self.triples)}")

    def add_radlex(self, radlex_path: Path):
        if not radlex_path.exists():
            print(f"[KGBuilder] RadLex not found: {radlex_path}")
            return

        print(f"[KGBuilder] Parsing RadLex from {radlex_path.name}...")
        triples_before = len(self.triples)

        with radlex_path.open(encoding="utf8") as f:
            buf = []
            for line in tqdm(f, desc="RadLex scan"):
                buf.append(line.strip())
                if "</owl:Class>" in line:  # end of a class block
                    block = " ".join(buf)
                    buf = []

                    sub_match = re.search(r'rdf:about=".*?(RID\d+)"', block)
                    label_match = re.search(r'<rdfs:label>(.*?)</rdfs:label>', block)
                    parent_match = re.search(r'rdfs:subClassOf rdf:resource=".*?(RID\d+)"', block)
                    synonym_matches = re.findall(r'<oboInOwl:hasExactSynonym>(.*?)</oboInOwl:hasExactSynonym>', block)

                    if sub_match:
                        rid = sub_match.group(1)
                        node = f"radlex:{rid}"
                        self.entity_meta_raw[node] = {
                            "id": node,
                            "label": label_match.group(1) if label_match else None,
                            "synonyms": synonym_matches,
                            "definition": None
                        }
                        for syn in synonym_matches:
                            syn_node = f"syn:{_sanitize_node(syn)}"
                            self.triples.append((syn_node, "SYNONYM_OF", node, 1.0, "radlex"))
                        if parent_match:
                            parent = parent_match.group(1)
                            self.triples.append((node, "is_a", f"radlex:{parent}", 1.0, "radlex"))

        print(f"[KGBuilder] RadLex triples added: {len(self.triples) - triples_before}")

    # ---------- curated CSV ----------
    def add_curated_csv(self, csv_path: str):
        p = Path(csv_path)
        if not p.exists():
            raise FileNotFoundError(csv_path)

        print(f"[KGBuilder] Adding curated triples from {p.name}...")
        with p.open(newline='', encoding='utf8') as f:
            reader = list(csv.DictReader(f))

        for row in tqdm(reader, desc="Curated CSV"):
            s = row['s']; r = row['r']; o = row['o']
            conf = float(row.get('confidence', 1.0))
            self.triples.append((s, r, o, conf, "curated"))

        print(f"[KGBuilder] Curated triples added: {len(self.triples)}")

    # ---------- vocab building ----------
    def _ensure_vocab(self):
        next_n, next_r = 0, 0
        for s, r, o, conf, src in self.triples:
            for ent in (s, o):
                if ent not in self.node2id:
                    self.node2id[ent] = next_n
                    if ent in self.entity_meta_raw:
                        self.entity_meta[next_n] = self.entity_meta_raw[ent]
                    else:
                        self.entity_meta[next_n] = {"id": ent}
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

        # write JSON maps with utf-8 encoding
        (self.out_dir / "node2id.json").write_text(json.dumps(self.node2id, indent=2, ensure_ascii=False), encoding='utf-8')
        (self.out_dir / "relation2id.json").write_text(json.dumps(self.relation2id, indent=2, ensure_ascii=False), encoding='utf-8')
        (self.out_dir / "entity_meta.json").write_text(json.dumps(self.entity_meta, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f"[KGBuilder] wrote: {tpath} and maps to {self.out_dir}")

    # ---------- build entrypoint ----------
    def build(self, xml_dir: Optional[str] = None, dicom_root: Optional[str] = None,
                curated_csv: Optional[str] = None, mode: str = "hybrid",
                save_feats_path: Optional[str] = None, backbone_type: str = "swin", device: str = "cuda"):
        """
        mode: "dataset", "ontology", "hybrid"
        """
        if mode == "dataset":
            self.build_from_parsed(xml_dir=xml_dir, dicom_root=dicom_root, save_feats_path=save_feats_path, 
                                    backbone_type=backbone_type, device=device)

        elif mode == "ontology":
            self.add_doid(ONTO_DIR / "doid.obo")
            self.add_radlex(ONTO_DIR / "RadLex.owl")

        elif mode == "hybrid":
            self.build_from_parsed(xml_dir, dicom_root, save_feats_path=save_feats_path, backbone_type=backbone_type)
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
                        curated_csv: Optional[str] = None, mode: str = "hybrid",
                        save_feats_path: Optional[str] = None, backbone_type: str = "swin",
                        device="cuda"):
        tpath = KG_DIR / "triples.csv"
        if not tpath.exists():
            print(f"[KGBuilder] triples.csv not found, building KG in mode={mode}...")
            builder = cls()
            builder.build(xml_dir=xml_dir, dicom_root=dicom_root,
                            curated_csv=curated_csv, mode=mode, save_feats_path=save_feats_path, 
                            backbone_type=backbone_type, device=device)
        else:
            print(f"[KGBuilder] triples.csv already exists -> using cached KG (mode={mode})")
