
import json
import re
from pathlib import Path
import numpy as np
import torch
from typing import Optional, Dict, Any, List
from LabelData.labeledData import disease_groups, normal_groups, finding_groups, symptom_groups

def _sanitize_label_for_node(s: str) -> str:

    """Sanitization consistent with KG node naming: 'label:<sanitized>'"""
    s2 = s.strip()
    s2 = re.sub(r'\s+', '_', s2)
    s2 = re.sub(r'[^A-Za-z0-9_:.-]', '', s2)
    return s2

def _find_node_embeddings_file(kg_dir: Path) -> Optional[Path]:
    # prefer best / epoch-sorted
    cands = sorted(kg_dir.glob("node_embeddings_best.npy"))
    if not cands:
        cands = sorted(kg_dir.glob("node_embeddings_epoch*.npy"))
    if not cands:
        cands = sorted(kg_dir.glob("node_embeddings*.npy"))
    return cands[-1] if cands else None

def _try_import_label_lists() -> Optional[List[str]]:
    try:
        names = set()
        names.update(disease_groups.keys())
        names.update(normal_groups.keys())
        names.update(finding_groups.keys())
        names.update(symptom_groups.keys())
        return sorted(names)
    except Exception:
        return None

def ensure_label_embeddings(base_dir: Path,
                            kg_subdir: str = "knowledge_graph",
                            out_name: str = "label_embeddings.pt",
                            label_names: Optional[List[str]] = None,
                            force_recreate: bool = False) -> Dict[str, Any]:
    """
    Ensure knowledge_graph/label_embeddings.pt exists. If missing (or force_recreate=True),
    build it using node_embeddings*.npy + node2id.json.
    Returns the loaded or newly-created label_emb_dict (mapping label_name -> numpy array or torch tensor).
    Raises FileNotFoundError if necessary KG artifacts are missing.
    """
    kg_dir = Path(base_dir) / kg_subdir
    kg_dir.mkdir(parents=False, exist_ok=True)
    out_path = kg_dir / out_name

    # If file exists and no force -> load and return
    if out_path.exists() and not force_recreate:
        try:
            data = torch.load(out_path)
            print(f"[INFO] Loaded existing label embeddings from {out_path} (len={len(data)})")
            return data
        except Exception as e:
            print(f"[WARN] Failed to load existing {out_path}: {e} -- will attempt to recreate")

    # locate node embeddings and node2id
    emb_file = _find_node_embeddings_file(kg_dir)
    node2id_file = kg_dir / "node2id.json"
    if not emb_file or not node2id_file.exists():
        raise FileNotFoundError(
            f"Missing KG artifacts: node_embeddings file found: {bool(emb_file)}, node2id.json exists: {node2id_file.exists()}. "
            "Run KG build/training first or provide label_embeddings.pt manually."
        )

    node_emb = np.load(emb_file)  # shape (N, D)

    print(f"[INFO] Using node embeddings file: {emb_file} (shape={node_emb.shape})")

    with open(node2id_file, "r", encoding="utf8") as f:
        node2id = json.load(f)

    # determine label names
    if label_names is None:
        label_names = _try_import_label_lists()
    if label_names is None:
        # fallback: infer labels from node2id keys "label:..."
        inferred = []
        for k in node2id.keys():
            if isinstance(k, str) and k.startswith("label:"):
                raw = k[len("label:"):]
                # reverse the sanitization somewhat (underscores -> spaces)
                inferred.append(raw.replace("_", " "))
        if inferred:
            print(f"[INFO] Inferred {len(inferred)} label names from node2id keys.")
            label_names = sorted(set(inferred))
        else:
            raise RuntimeError("Could not determine label names (no supplied list, import failed, and no 'label:' nodes found).")

    # build label -> embedding dict
    out_dict = {}
    dim = node_emb.shape[1]
    for lab in label_names:
        node_key = f"label:{_sanitize_label_for_node(lab)}"
        if node_key in node2id:
            idx = int(node2id[node_key])
            out_dict[lab] = node_emb[idx].astype(np.float32)  # numpy array; torch.load will return this same structure
        else:
            # fallback: zero vector (explicit warning)
            out_dict[lab] = np.zeros((dim,), dtype=np.float32)
            print(f"[WARN] Label node not found in node2id: {node_key} -> saving zero vector for label '{lab}'")

    # save as torch .pt so existing code can torch.load it
    torch.save(out_dict, out_path)
    print(f"[INFO] Created label embeddings file at {out_path} (labels: {len(out_dict)})")
    return out_dict
