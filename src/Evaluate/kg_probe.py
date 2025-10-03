from pathlib import Path
import sys
try:
    base = Path(__file__).resolve().parent.parent
except NameError:
    base = Path.cwd().parent
sys.path.append(str(base))
import numpy as np
import json, csv, os

try:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
except NameError:
    BASE_DIR = Path.cwd().parent.parent
KG_DIR = BASE_DIR / "knowledge_graph"
EMB_PATH = KG_DIR / "node_embeddings_best.npy"
NAMES_PATH = KG_DIR / "node2id.json"

def load_embeddings(path):
    """Load numpy embeddings (N, D)."""
    return np.load(path)


def load_names(path):
    """
    Load node names aligned with embeddings.
    Supports .txt, .npy, .json, .csv.
    Special case: node2id.json (mapping node->id).
    """
    p = Path(path)

    # text file: one name per line
    if p.suffix == ".txt":
        return [line.strip() for line in p.read_text(encoding="utf8").splitlines()]

    # npy array of names
    if p.suffix == ".npy":
        return list(np.load(p))

    # json
    if p.suffix == ".json":
        with open(p, "r", encoding="utf8") as f:
            j = json.load(f)
        # if already a list
        if isinstance(j, list):
            return j
        # if dict
        if isinstance(j, dict):
            # detect node2id mapping (values numeric)
            if all(isinstance(v, int) or (isinstance(v, str) and str(v).isdigit()) for v in j.values()):
                max_id = max(int(v) for v in j.values())
                id2node = ["<UNK>"] * (max_id + 1)
                for node, nid in j.items():
                    id2node[int(nid)] = node
                return id2node
            # fallback: treat as dict of id->name
            try:
                keys = sorted(j.keys(), key=lambda x: int(x))
                return [j[k] for k in keys]
            except Exception:
                return list(j.values())

    # csv
    if p.suffix == ".csv":
        out = []
        with open(p, newline="", encoding="utf8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if "name" in r:
                    out.append(r["name"])
                elif "node" in r:
                    out.append(r["node"])
                else:
                    out.append(list(r.values())[0])
        return out

    raise RuntimeError(f"Unsupported name file: {p}")


def normalize_rows(X):
    """Row-normalize embeddings for cosine similarity."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    return X / norms


def topk_by_index(emb, names, idx, k=10):
    """Find top-k neighbors of embedding[idx]."""
    X = normalize_rows(emb)
    v = X[idx : idx + 1]
    sims = (v @ X.T).reshape(-1)
    order = np.argsort(sims)[::-1]
    res = []
    for i in order:
        if i == idx:
            continue
        res.append((i, names[i] if i < len(names) else str(i), float(sims[i])))
        if len(res) >= k:
            break
    return res


def find_name_indices(names, query, topn=10):
    """Find indices matching query (exact or substring, case-insensitive)."""
    q = query.lower()
    idxs = [i for i, n in enumerate(names) if q == n.lower()]
    if idxs:
        return idxs
    idxs = [i for i, n in enumerate(names) if q in n.lower()]
    if idxs:
        return idxs[:topn]
    idxs = [i for i, n in enumerate(names) if any(w == q for w in n.lower().split())]
    return idxs[:topn]


def topk_by_name(emb, names, query, k=10):
    """Find neighbors given a query string."""
    idxs = find_name_indices(names, query)
    if not idxs:
        print(f"[WARN] No match for '{query}'. Try a different query or inspect names.")
        return {}
    out = {}
    for idx in idxs:
        out[idx] = topk_by_index(emb, names, idx, k=k)
    return out


if __name__ == "__main__":
    if not os.path.exists(EMB_PATH):
        print(f"Embeddings file not found: {EMB_PATH}")
        sys.exit(1)
    if not os.path.exists(NAMES_PATH):
        print(f"Node names file not found: {NAMES_PATH}")
        sys.exit(1)

    emb = load_embeddings(EMB_PATH)
    names = load_names(NAMES_PATH)

    print("Loaded embeddings:", emb.shape, "names:", len(names))
    if emb.shape[0] != len(names):
        print(f"[WARN] Mismatch: {emb.shape[0]} embeddings vs {len(names)} names!")

    # Example queries
    queries = ["pneumonia", "copd_emphysema", "Normal", "bronchiectasis", "fibrosis_ild", "infection_pneumonia"]

    for q in queries:
        print("\n=== Query:", q, "===")
        res = topk_by_name(emb, names, q, k=10)
        for idx, neighs in res.items():
            print(f"\nNode index {idx} name='{names[idx]}' neighbors:")
            for i, nm, score in neighs:
                print(f"  idx={i:5d}  sim={score:.4f}   name={nm}")
