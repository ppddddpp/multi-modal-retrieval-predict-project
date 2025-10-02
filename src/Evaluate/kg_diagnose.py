from pathlib import Path
import sys
try:
    base = Path(__file__).resolve().parent.parent
except NameError:
    base = Path.cwd().parent
sys.path.append(str(base))
import numpy as np, json, sys
from sklearn.metrics.pairwise import cosine_similarity

try:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
except NameError:
    BASE_DIR = Path.cwd().parent.parent
KG_DIR = BASE_DIR / "knowledge_graph"

def exists_and_list():
    print("Files in KG dir:", KG_DIR.exists(), list(KG_DIR.glob("*"))[:10])

def load_node_maps():
    n2id = json.load(open(KG_DIR / "node2id.json", "r", encoding="utf8"))
    rid  = json.load(open(KG_DIR / "relation2id.json", "r", encoding="utf8")) if (KG_DIR/"relation2id.json").exists() else {}
    print("node2id size:", len(n2id), "relation2id size:", len(rid))
    return n2id, rid

def find_emb_file():
    for name in ["node_embeddings_best.npy"]:
        p = KG_DIR / name
        if p.exists(): return p
    # fallback
    cands = sorted(KG_DIR.glob("node_embeddings_epoch*.npy")) or sorted(KG_DIR.glob("node_embeddings*.npy"))
    return cands[-1] if cands else None

def load_embeddings(p):
    emb = np.load(p)
    print("emb shape:", emb.shape, "nan count:", np.isnan(emb).sum())
    norms = np.linalg.norm(emb, axis=1)
    print("norms: min", norms.min(), "median", np.median(norms), "max", norms.max())
    return emb

def topk_for_label(label_key, node2id, id2node, emb, k=10):
    if label_key not in node2id:
        print("Label node not in map:", label_key); return
    idx = node2id[label_key]
    v = emb[[idx]]
    sims = cosine_similarity(v, emb)[0]
    top = sims.argsort()[::-1][1:k+1]  # skip self
    return [(id2node[i], float(sims[i])) for i in top]

if __name__=="__main__":
    exists_and_list()
    try:
        node2id, rel2id = load_node_maps()
    except FileNotFoundError as e:
        print("Missing node2id/relation2id:", e); sys.exit(1)

    emb_file = find_emb_file()
    if not emb_file:
        print("No node_embeddings file found in knowledge_graph/ — run KGTrainer first."); sys.exit(1)
    print("Using embeddings file:", emb_file)
    emb = load_embeddings(emb_file)

    # build id->node
    max_idx = max(node2id.values())
    id2node = [None] * (max_idx + 1)
    for k,v in node2id.items(): id2node[v] = k

    # check a few label nodes (example: 'label:Normal' or infer one)
    some_labels = [k for k in node2id.keys() if k.startswith("label:")][:5]
    print("Sample label nodes:", some_labels)
    for lab in some_labels:
        print("Top neighbors for", lab, ":", topk_for_label(lab, node2id, id2node, emb, k=10))
