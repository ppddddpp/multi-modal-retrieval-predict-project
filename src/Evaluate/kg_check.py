from pathlib import Path
import sys
try:
    base = Path(__file__).resolve().parent.parent
except NameError:
    base = Path.cwd().parent
sys.path.append(str(base))
import numpy as np, json
from sklearn.metrics.pairwise import cosine_similarity

try:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
except NameError:
    BASE_DIR = Path.cwd().parent.parent

KG = BASE_DIR/"knowledge_graph"
node2id = json.load(open(KG/"node2id.json","r",encoding="utf8"))
emb = np.load(KG/"node_embeddings_best.npy")

# Basic stats
norms = np.linalg.norm(emb, axis=1)
zero_mask = norms == 0
print("Total nodes:", emb.shape[0])
print("Zero-norm nodes:", zero_mask.sum(), "  (%0.3f%%)" % (100*zero_mask.sum()/emb.shape[0]))
print("Norms: min", norms.min(), "median", np.median(norms), "max", norms.max())

# Which node names are zero?
id2node = {v:k for k,v in node2id.items()}
zeros = [id2node[i] for i in range(len(norms)) if zero_mask[i]]
print("Example zero-nodes (first 50):", zeros[:50])

# Label coverage: percent of label: nodes that are zero
label_nodes = [k for k in node2id if k.startswith("label:")]
label_zero = sum(1 for k in label_nodes if norms[node2id[k]]==0)
print("Label nodes total:", len(label_nodes), "label zero vectors:", label_zero)

# Optionally: check distribution of cosine similarities between random label pairs
import random
labels_sample = random.sample(label_nodes, min(200, len(label_nodes)))
pairs = []
for i in range(len(labels_sample)-1):
    a = emb[node2id[labels_sample[i]]]
    b = emb[node2id[labels_sample[i+1]]]
    pairs.append(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))
print("Sample label-label cosine: min, median, max:", np.min(pairs), np.median(pairs), np.max(pairs))

# small neighbour function
def topk(node, k=10):
    idx = node2id.get(node)
    if idx is None:
        return None
    sims = cosine_similarity(emb[[idx]], emb)[0]
    top = sims.argsort()[::-1][1:k+1]
    return [(id2node[i], float(sims[i])) for i in top]

print("Top neighbours for some labels:", topk('label:Normal')[:5])
