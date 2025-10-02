import numpy as np
import json
import torch
from typing import Tuple, List, Optional, Dict, Any, Set
from pathlib import Path
import pandas as pd
from KnowledgeGraph.label_attention import LabelAttention

# Base dirs
BASE_DIR        = Path(__file__).resolve().parent.parent.parent
EMBEDDINGS_DIR  = BASE_DIR / "embeddings"
FEATURES_PATH   = BASE_DIR / "featureDBs"
KG_DIR_DEFAULT  = BASE_DIR / "knowledge_graph"
LABELS_CSV_DEF  = BASE_DIR / "outputs" / "openi_labels_final.csv"

class Reranker:
    """
    Lightweight reranker that combines:
      - embedding cosine (from joint embeddings),
      - label Jaccard (from outputs/openi_labels_final.csv),
      - KG-based cosine (from knowledge_graph node embeddings).
    Usage:
      rer = Reranker(kg_dir=..., labels_csv=...)
      rer.rerank(query_id, candidate_ids, candidate_embs)
    """

    def __init__(
        self,
        kg_dir: Optional[Path] = None,
        labels_csv: Optional[Path] = None,
        alpha: float = 0.6,
        beta: float = 0.25,
        gamma: float = 0.15,
        preload_record_kg: bool = True,
    ):
        self.kg_dir = Path(kg_dir) if kg_dir else KG_DIR_DEFAULT
        self.labels_csv = Path(labels_csv) if labels_csv else LABELS_CSV_DEF
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # load KG artifacts
        self.kg = self._load_kg(self.kg_dir)
        # load labels
        self.labels_df = pd.read_csv(self.labels_csv, index_col="id")
        self.labels_df.index = self.labels_df.index.astype(str)

        # optional precompute record_kg_vectors if matching embeddings id list exists
        self.record_kg_vectors = None
        self.record_kg_id2idx = None
        if preload_record_kg:
            try:
                self._try_precompute_record_kg()
            except Exception:
                # non-fatal; we can compute on the fly per record
                self.record_kg_vectors = None
                self.record_kg_id2idx = None

        self.attn_model = None
        model_path = BASE_DIR / "label attention model" / "label_attention_model.pt"
        if model_path.exists():
            d_kge = self.kg["node_emb"].shape[1]
            self.attn_model = LabelAttention(d_emb=d_kge)
            self.attn_model.load_state_dict(torch.load(model_path, map_location="cpu"))
            self.attn_model.eval()

    def _load_kg(self, kg_dir: Path) -> Dict[str, Any]:
        """
        Loads Knowledge Graph (KG) node embeddings from the given directory.

        Specifically, it loads:
        - node2id.json (mapping node names to indices)
        - node embeddings (best checkpoint if available; otherwise, latest epoch)

        Returns a dictionary with the following keys:
        - "node2id": mapping node names to indices
        - "id2node": mapping indices back to node names
        - "node_emb": node embeddings (numpy array, shape (N, D))
        """
        node2id_path = kg_dir / "node2id.json"
        if not node2id_path.exists():
            raise FileNotFoundError(f"KG node2id.json not found at {node2id_path}")
        node2id = json.load(open(node2id_path, "r", encoding="utf8"))

        # prefer best checkpoint
        best_files = sorted(kg_dir.glob("node_embeddings_best.npy"))
        if best_files:
            node_emb_file = best_files[-1]
        else:
            # fall back to epoch checkpoints
            node_emb_files = sorted(kg_dir.glob("node_embeddings_epoch*.npy"))
            if not node_emb_files:
                node_emb_files = sorted(kg_dir.glob("node_embeddings*.npy"))
            if not node_emb_files:
                raise FileNotFoundError("No .npy embeddings found in KG dir")
            node_emb_file = node_emb_files[-1]

        node_emb = np.load(node_emb_file)
        node_emb = node_emb / (np.linalg.norm(node_emb, axis=1, keepdims=True) + 1e-12)

        # build id->node list
        max_idx = max(node2id.values())
        id2node = [None] * (max_idx + 1)
        for k, v in node2id.items():
            id2node[v] = k

        print(f"[Reranker] loaded KG embeddings from {node_emb_file}")
        return {"node2id": node2id, "id2node": id2node, "node_emb": node_emb}

    # -------------------------
    # helpers
    # -------------------------
    @staticmethod
    def safe_cos(a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None:
            return 0.0
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    @staticmethod
    def jaccard_sets(a: Set[str], b: Set[str]) -> float:
        if not a and not b:
            return 0.0
        inter = len(a & b); uni = len(a | b)
        return 0.0 if uni == 0 else inter / uni

    @staticmethod
    def minmax_scale_list(x_list: List[float]) -> List[float]:
        arr = np.array(x_list, dtype=float)
        if arr.size == 0:
            return arr.tolist()
        lo = float(np.nanmin(arr)); hi = float(np.nanmax(arr))
        if hi - lo == 0:
            return [0.0] * len(arr)
        return ((arr - lo) / (hi - lo)).tolist()

    def get_record_label_set(self, rec_id: str) -> Set[str]:
        """
        Returns the set of labels for the given record ID.

        If the record ID is not found in the labels dataframe, returns an empty set.

        Otherwise, returns a set of labels for which the value in the dataframe is 1.
        """
        if str(rec_id) not in self.labels_df.index:
            return set()
        row = self.labels_df.loc[str(rec_id)]
        return set([c for c, v in row.items() if int(v) == 1])

    def get_record_kg_vec(self, rec_id: str) -> np.ndarray:
        """
        Returns a Knowledge Graph (KG) node embedding vector for the given record ID.
        
        First, it tries to look up the record ID in the node2id mapping.
        If not found, it falls back to looking up the lower-cased record ID.
        If still not found, it uses label attention pooling to get a weighted average of label node embeddings.
        """
        node2id = self.kg["node2id"]; node_emb = self.kg["node_emb"]
        k1 = f"report:{rec_id}"
        if k1 in node2id:
            return node_emb[node2id[k1]]
        if str(rec_id) in node2id:
            return node_emb[node2id[str(rec_id)]]

        # fallback: attention-pooled label node embeddings
        labels = self.get_record_label_set(rec_id)
        if not labels:
            return np.zeros(node_emb.shape[1], dtype=float)

        vecs = []
        for lab in labels:
            cands = [f"label:{lab}", lab, lab.lower(), lab.replace(" ", "_")]
            for ck in cands:
                if ck in node2id:
                    vecs.append(node_emb[node2id[ck]])
                    break
        if not vecs:
            return np.zeros(node_emb.shape[1], dtype=float)

        label_embs = np.stack(vecs, axis=0)  # [n_labels, d]

        if self.attn_model is not None:
            with torch.no_grad():
                # shape: [1, n_labels, d]
                label_embs_t = torch.from_numpy(label_embs).float().unsqueeze(0)
                pooled, _ = self.attn_model(label_embs_t)  # [1, d]
            return pooled.squeeze(0).numpy()
        else:
            return label_embs.mean(axis=0)

    def _try_precompute_record_kg(self):
        """
        If a typical 'trainval_ids.json' exists, precompute record-level KG vectors to speed rerank.
        """
        ids_file = EMBEDDINGS_DIR / "trainval_ids.json"
        if not ids_file.exists():
            return
        with open(ids_file, "r") as f:
            ids = json.load(f)
        rec_vecs = np.zeros((len(ids), self.kg["node_emb"].shape[1]), dtype=float)
        for i, rid in enumerate(ids):
            rec_vecs[i] = self.get_record_kg_vec(str(rid))
        # save to KG dir for reuse
        out_np = self.kg_dir / "record_kg_vectors.npy"
        np.save(out_np, rec_vecs)
        self.record_kg_vectors = rec_vecs
        self.record_kg_id2idx = {str(rid): i for i, rid in enumerate(ids)}

    def rerank(
        self,
        query_id: str,
        candidate_ids: List[str],
        candidate_embs: Optional[np.ndarray] = None,
        candidate_emb_lookup: Optional[Dict[str, np.ndarray]] = None,
        topk: Optional[int] = None,
        query_emb: Optional[np.ndarray] = None, 
    ) -> List[Tuple[str, float, float, float, float]]:
        """
        Rerank a list of candidate records based on their embeddings, labels, and KG vectors.

        Args:
            query_id (str): ID of the query record
            candidate_ids (List[str]): IDs of the candidate records
            candidate_embs (Optional[np.ndarray]): embeddings of the candidate records
            candidate_emb_lookup (Optional[Dict[str, np.ndarray]]): lookup table for candidate embeddings
            topk (Optional[int]): number of top records to return
            query_emb (Optional[np.ndarray]): embedding of the query record

        Returns:
            List[Tuple[str, float, float, float, float]]: a list of tuples containing the ID of the record, the final reranking score, the cosine similarity score, the Jaccard similarity score, and the KG similarity score.
        """
        N = len(candidate_ids)
        # get candidate embeddings (build from lookup if needed)
        if candidate_embs is None:
            if candidate_emb_lookup is not None:
                candidate_embs = np.vstack([candidate_emb_lookup.get(str(cid), 
                                        np.zeros(next(iter(candidate_emb_lookup.values())).shape, dtype=float))
                                        for cid in candidate_ids])
            else:
                raise ValueError("Please provide candidate_embs or candidate_emb_lookup.")

        # basic shape check
        if candidate_embs.shape[0] != N:
            raise ValueError("candidate_embs rows must match candidate_ids length")

        # embedding (cosine) scores
        # Resolve query embedding (multiple fallbacks)
        q_emb = None
        if candidate_emb_lookup is not None and str(query_id) in candidate_emb_lookup:
            q_emb = candidate_emb_lookup[str(query_id)]
        elif query_emb is not None:
            q_emb = query_emb
        else:
            # try find query inside candidate_ids and use that row from candidate_embs
            for i, cid in enumerate(candidate_ids):
                if str(cid) == str(query_id):
                    q_emb = candidate_embs[i]
                    break

        if q_emb is None:
            raise ValueError(
                "Query embedding not found. Provide candidate_emb_lookup keyed by query_id, "
                "or include the query_id in candidate_ids with matching candidate_embs, "
                "or pass query_emb explicitly."
            )

        emb_scores = [self.safe_cos(q_emb, candidate_embs[i]) for i in range(N)]

        # label (Jaccard) scores
        q_labels = self.get_record_label_set(query_id)
        lab_scores = []
        for cid in candidate_ids:
            lab_scores.append(self.jaccard_sets(q_labels, self.get_record_label_set(cid)))

        # KG scores
        kg_scores = []
        if self.record_kg_vectors is not None and self.record_kg_id2idx is not None:
            q_idx = self.record_kg_id2idx.get(str(query_id), None)
            q_kg = self.record_kg_vectors[q_idx] if q_idx is not None else self.get_record_kg_vec(str(query_id))
            for cid in candidate_ids:
                c_idx = self.record_kg_id2idx.get(str(cid), None)
                c_kg = self.record_kg_vectors[c_idx] if c_idx is not None else self.get_record_kg_vec(str(cid))
                kg_scores.append(self.safe_cos(q_kg, c_kg))
        else:
            q_kg = self.get_record_kg_vec(str(query_id))
            for cid in candidate_ids:
                c_kg = self.get_record_kg_vec(str(cid))
                kg_scores.append(self.safe_cos(q_kg, c_kg))

        # normalize and combine
        emb_n = np.array(self.minmax_scale_list(emb_scores))
        lab_n = np.array(self.minmax_scale_list(lab_scores))
        kg_n  = np.array(self.minmax_scale_list(kg_scores))
        final = self.alpha * emb_n + self.beta * lab_n + self.gamma * kg_n

        ranked_idx = np.argsort(final)[::-1]
        if topk:
            ranked_idx = ranked_idx[:topk]
        out = []
        for i in ranked_idx:
            out.append((candidate_ids[i], float(final[i]), float(emb_n[i]), float(lab_n[i]), float(kg_n[i])))
        return out