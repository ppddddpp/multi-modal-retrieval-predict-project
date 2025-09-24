import abc
import numpy as np
import json
from typing import Tuple, List, Optional, Dict, Any, Set
from sklearn.metrics.pairwise import cosine_similarity
import heapq
from pathlib import Path
import pickle
import pandas as pd

# Base dirs
BASE_DIR        = Path(__file__).resolve().parent.parent.parent
EMBEDDINGS_DIR  = BASE_DIR / "embeddings"
FEATURES_PATH   = BASE_DIR / "featureDBs"
KG_DIR_DEFAULT  = BASE_DIR / "knowledge_graph"
LABELS_CSV_DEF  = BASE_DIR / "outputs" / "openi_labels_final.csv"

# ---------------------------
# Reranker class (KG + label + embedding)
# ---------------------------
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

    def _load_kg(self, kg_dir: Path) -> Dict[str, Any]:
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
        if str(rec_id) not in self.labels_df.index:
            return set()
        row = self.labels_df.loc[str(rec_id)]
        return set([c for c, v in row.items() if int(v) == 1])

    def get_record_kg_vec(self, rec_id: str) -> np.ndarray:
        """
        Try in order:
          1) node key "report:<id>"
          2) raw id as key
          3) average label node embeddings (fallback)
        """
        node2id = self.kg["node2id"]; node_emb = self.kg["node_emb"]
        k1 = f"report:{rec_id}"
        if k1 in node2id:
            return node_emb[node2id[k1]]
        if str(rec_id) in node2id:
            return node_emb[node2id[str(rec_id)]]

        # fallback: average label node embeddings
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
        return np.mean(np.stack(vecs, axis=0), axis=0)

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
        topk: Optional[int] = None
    ) -> List[Tuple[str, float, float, float, float]]:
        """
        Return list of tuples: (candidate_id, final_score, emb_score, lab_score, kg_score)
        - candidate_embs: np.array shape (N, D) in same order as candidate_ids (fast)
        - OR candidate_emb_lookup: dict id->embedding
        - If none provided, raises.
        """
        N = len(candidate_ids)
        # 1) get candidate embeddings
        if candidate_embs is None:
            if candidate_emb_lookup is not None:
                candidate_embs = np.vstack([candidate_emb_lookup[str(cid)] for cid in candidate_ids])
            else:
                raise ValueError("Please provide candidate_embs or candidate_emb_lookup.")

        # 2) embedding (cosine) scores
        # query embedding must be provided via candidate_emb_lookup keyed by query_id or precomputed arrays upstream
        if candidate_emb_lookup and str(query_id) in candidate_emb_lookup:
            q_emb = candidate_emb_lookup[str(query_id)]
        else:
            # query embedding must be passed in candidate_embs (rare); otherwise reranker can't compute emb similarity
            # as an alternative user can call rerank with candidate_emb_lookup where query is included
            raise ValueError("Query embedding not found in candidate_emb_lookup. Provide it keyed by query_id.")

        emb_scores = [self.safe_cos(q_emb, candidate_embs[i]) for i in range(N)]

        # 3) label (Jaccard) scores
        q_labels = self.get_record_label_set(query_id)
        lab_scores = []
        for cid in candidate_ids:
            lab_scores.append(self.jaccard_sets(q_labels, self.get_record_label_set(cid)))

        # 4) KG scores
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

        # 5) normalize and combine
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

class RetrievalEngine(abc.ABC):
    """
    Abstract base class for retrieval engines.
    Provides a convenience id->index mapping and embedding access.
    """

    def __init__(self, features_path: str, ids_path: str):
        # load embeddings and IDs once
        self.embs = np.load(features_path).astype("float32")  # shape (N, D)
        with open(ids_path, "r") as f:
            self.ids = json.load(f)                             # length N
        # id->index map for quick lookup (keys as strings)
        self.id2idx = {str(self.ids[i]): i for i in range(len(self.ids))}
        # small sanity
        assert self.embs.shape[0] == len(self.ids), "embeddings count != ids count"

    @abc.abstractmethod
    def retrieve(self, query_emb: np.ndarray, K: int = 5, **kwargs) -> Tuple[List[str], List[float]]:
        """
        Given a query embedding (D,) or (1,D), return top-K IDs and their scores.
        """
        pass

    def get_embeddings_for_ids(self, ids: List[str]) -> np.ndarray:
        """Return embeddings in same order as ids (zeros if missing)."""
        rows = []
        for _id in ids:
            idx = self.id2idx.get(str(_id), None)
            if idx is None:
                rows.append(np.zeros(self.embs.shape[1], dtype=self.embs.dtype))
            else:
                rows.append(self.embs[idx])
        return np.vstack(rows)


class DLSRetrievalEngine(RetrievalEngine):
    """
    DenseLinkSearch implementation:
    - Precomputes a cosine-similarity neighbor graph above a threshold.
    - On query, does a sublinear graph-traversal scan.
    """

    def __init__(
        self,
        features_path: str,
        ids_path: str,
        link_threshold: float = 0.5,
        max_links: int = 10,
        fdb_path: Optional[str] = None,
        name: Optional[str] = None,
    ):
        super().__init__(features_path, ids_path)

        # decide on a single cache path
        if fdb_path:
            self.fdb_path = Path(fdb_path)
        else:
            if name:
                self.fdb_path = FEATURES_PATH / name
            else:
                stem = Path(features_path).stem
                self.fdb_path = FEATURES_PATH / f"{stem}_link_graph.pkl"

        # ensure parent dir exists
        self.fdb_path.parent.mkdir(parents=True, exist_ok=True)

        # try load, rebuild on error or size mismatch
        rebuild = True
        if self.fdb_path.exists():
            try:
                with open(self.fdb_path, "rb") as f:
                    graph = pickle.load(f)
                if len(graph) == self.embs.shape[0]:
                    self.link_graph = graph
                    rebuild = False
            except Exception:
                rebuild = True

        if rebuild:
            self.link_graph = self._build_link_graph(link_threshold, max_links)
            with open(self.fdb_path, "wb") as f:
                pickle.dump(self.link_graph, f)

    def _build_link_graph(self, threshold: float, max_links: int) -> List[List[int]]:
        """
        Build an adjacency list of each node’s top-similarity neighbors above threshold.
        """
        sim = cosine_similarity(self.embs)        # (N, N)
        np.fill_diagonal(sim, -1)                 # ignore self-similarity

        graph = []
        for i in range(sim.shape[0]):
            # sort by descending similarity
            nbrs = np.argsort(sim[i])[::-1]
            # filter and truncate
            selected = [int(j) for j in nbrs if sim[i, j] >= threshold][:max_links]
            graph.append(selected)
        return graph

    def retrieve(
        self,
        query_emb: np.ndarray,
        K: int = 5,
        seed_size: int = 5,
        max_steps: int = 100,
        candidate_multiplier: int = 10,
        reranker: Optional["Reranker"] = None,
        query_id: Optional[str] = None,
        rerank_topk: Optional[int] = None
    ) -> Tuple[List[str], List[float]]:
        """
        Sublinear DLS retrieval via greedy graph-walk using a heap-based candidate pool.

        Optional reranker: pass a Reranker instance and query_id (str) to perform final reranking.
        If reranker is provided, it will be called with the top candidates and their embeddings.
        """
        q = query_emb.astype("float32").reshape(-1)
        N = self.embs.shape[0]

        if len(self.link_graph) != N:
            raise RuntimeError(
                f"Link graph size {len(self.link_graph)} != embeddings {N}. "
                "Rebuild or delete your pickle."
            )

        # --- Seed selection ---
        seeds = np.random.choice(N, size=min(seed_size, N), replace=False).tolist()
        visited = set(seeds)

        # --- Initial scoring: heap stores (-similarity, idx) so heapq gives max-sim first
        heap = []
        q_norm = np.linalg.norm(q) + 1e-6
        for idx in seeds:
            emb = self.embs[idx]
            sim = float(emb @ q / (np.linalg.norm(emb) * q_norm + 1e-12))
            heapq.heappush(heap, (-sim, idx))

        R = max(candidate_multiplier * K, seed_size)
        steps = 0

        # --- Greedy graph walk ---
        while steps < max_steps and heap:
            # Pop best candidate
            neg_sim, best_idx = heapq.heappop(heap)
            best_sim = -neg_sim
            improved = False

            # Expand neighbors
            for nbr in self.link_graph[best_idx]:
                if nbr < 0 or nbr >= N or nbr in visited:
                    continue
                visited.add(nbr)
                nbr_emb = self.embs[nbr]
                nbr_sim = float(nbr_emb @ q / (np.linalg.norm(nbr_emb) * q_norm + 1e-12))
                heapq.heappush(heap, (-nbr_sim, nbr))
                improved = True

            # Keep heap size bounded
            if len(heap) > R:
                heap = heapq.nsmallest(R, heap)  # keeps best R (lowest neg_sim)
                heapq.heapify(heap)

            if not improved:
                break

            steps += 1

        # --- Get top-K sorted by similarity ---
        topk = heapq.nsmallest(K, heap)
        topk = sorted([(-neg_sim, idx) for neg_sim, idx in topk], reverse=True)

        ids = [self.ids[idx] for _, idx in topk]
        scores = [sim for sim, _ in topk]

        cand_embs = self.get_embeddings_for_ids(ids)

        # build lookup that maps id->embedding and also include the query embedding
        cand_lookup = {str(rid): emb for rid, emb in zip(ids, cand_embs)}
        # try to get query embedding from engine if it exists there; else use the query vector q
        if query_id is not None and str(query_id) in self.id2idx:
            cand_lookup[str(query_id)] = self.embs[self.id2idx[str(query_id)]]
        else:
            cand_lookup[str(query_id)] = q 

        # --- optional reranking ---
        if reranker is not None and query_id is not None:
            # fetch candidate embeddings quickly
            cand_embs = self.get_embeddings_for_ids(ids)
            reranked = reranker.rerank(
                query_id=query_id,
                candidate_ids=ids,
                candidate_embs=cand_embs,
                candidate_emb_lookup=cand_lookup,
                topk=rerank_topk or K
            )
            # reranked: list of tuples (id, final_score, emb_score, lab_score, kg_score)
            ids = [t[0] for t in reranked]
            scores = [t[1] for t in reranked]

        return ids, scores

def make_retrieval_engine(
    features_path: str,
    ids_path: str,
    method: str = "dls",
    **kwargs
) -> RetrievalEngine:
    method = method.lower()
    if method == "dls":
        return DLSRetrievalEngine(
            features_path, ids_path,
            link_threshold=kwargs.get("link_threshold", 0.5),
            max_links=kwargs.get("max_links", 10),
            fdb_path=kwargs.get("fdb_path", None),
            name=kwargs.get("name", None)
        )
    else:
        raise ValueError(f"Unknown retrieval method: {method}")

# ---------------------------
# Minimal usage example
# ---------------------------
if __name__ == "__main__":
    # instantiate retrieval engine (uses your embeddings/ids)
    feats = str(EMBEDDINGS_DIR / "trainval_joint_embeddings.npy")
    idsf = str(EMBEDDINGS_DIR / "trainval_ids.json")
    engine = make_retrieval_engine(feats, idsf, method="dls", link_threshold=0.5, max_links=10)

    # instantiate reranker
    rer = Reranker(kg_dir=KG_DIR_DEFAULT, labels_csv=LABELS_CSV_DEF, alpha=0.6, beta=0.25, gamma=0.15)

    # Example: get query embedding (pick first id for demo)
    query_id = engine.ids[0]
    q_emb = engine.embs[0]

    # get top-50 candidates from DLS
    cand_ids, cand_scores = engine.retrieve(q_emb, K=50, seed_size=10, max_steps=200)

    # get embeddings for these candidates (fast)
    cand_embs = engine.get_embeddings_for_ids(cand_ids)

    # rerank using KG+labels
    ranked = rer.rerank(query_id=query_id, candidate_ids=cand_ids, candidate_embs=cand_embs, topk=20)
    print("Top 5 after rerank:", ranked[:5])
