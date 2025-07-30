import numpy as np
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
EMBEDDINGS_DIR = BASE_DIR / "embeddings"

train_emb = np.load(EMBEDDINGS_DIR / "train_joint_embeddings.npy")
val_emb = np.load(EMBEDDINGS_DIR / "val_joint_embeddings.npy")
merged_emb = np.concatenate([train_emb, val_emb], axis=0)
np.save(EMBEDDINGS_DIR / "trainval_joint_embeddings.npy", merged_emb)

with open(EMBEDDINGS_DIR / "train_ids.json") as f1, open(EMBEDDINGS_DIR / "val_ids.json") as f2:
    train_ids = json.load(f1)
    val_ids = json.load(f2)
    merged_ids = train_ids + val_ids

with open(EMBEDDINGS_DIR / "trainval_ids.json", "w") as fout:
    json.dump(merged_ids, fout)

print(f"Saved merged embeddings to: {EMBEDDINGS_DIR / 'trainval_joint_embeddings.npy'}")
print(f"Saved merged IDs to:        {EMBEDDINGS_DIR / 'trainval_ids.json'}")
