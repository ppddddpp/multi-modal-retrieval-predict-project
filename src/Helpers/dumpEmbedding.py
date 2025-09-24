import numpy as np
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
EMBEDDINGS_DIR = BASE_DIR / "embeddings"

def createDumpEmbedding(base_dir, embeddings_dir):
    """
    Creates a merged dump of train and validation embeddings and IDs.

    Parameters
    ----------
    base_dir : Path
        The base directory of the project.
    embeddings_dir : Path
        The directory where the embeddings are stored.

    Returns
    -------
    None
    """
    if not base_dir:
        base_dir = BASE_DIR
    if not embeddings_dir:
        embeddings_dir = EMBEDDINGS_DIR

    train_emb = np.load(embeddings_dir / "train_joint_embeddings.npy")
    val_emb = np.load(embeddings_dir / "val_joint_embeddings.npy")
    merged_emb = np.concatenate([train_emb, val_emb], axis=0)
    np.save(embeddings_dir / "trainval_joint_embeddings.npy", merged_emb)

    with open(embeddings_dir / "train_ids.json") as f1, open(embeddings_dir / "val_ids.json") as f2:
        train_ids = json.load(f1)
        val_ids = json.load(f2)
        merged_ids = train_ids + val_ids

    with open(EMBEDDINGS_DIR / "trainval_ids.json", "w") as fout:
        json.dump(merged_ids, fout)

    print(f"Saved merged embeddings to: {embeddings_dir / 'trainval_joint_embeddings.npy'}")
    print(f"Saved merged IDs to:        {embeddings_dir / 'trainval_ids.json'}")
