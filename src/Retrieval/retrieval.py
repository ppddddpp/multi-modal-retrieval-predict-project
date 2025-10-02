import abc
import numpy as np
import json
from typing import Tuple, List, Optional
from sklearn.metrics.pairwise import cosine_similarity
import heapq
from pathlib import Path
import pickle
from .reranker import Reranker

# Base dirs
BASE_DIR        = Path(__file__).resolve().parent.parent.parent
EMBEDDINGS_DIR  = BASE_DIR / "embeddings"
FEATURES_PATH   = BASE_DIR / "featureDBs"
KG_DIR_DEFAULT  = BASE_DIR / "knowledge_graph"
LABELS_CSV_DEF  = BASE_DIR / "outputs" / "openi_labels_final.csv"

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
        Builds a dense link graph based on cosine similarity above a given threshold.
        Each node is associated with a list of its max_links most similar neighbors.
        The result is a list of lists of node indices, where each inner list represents
        the neighbors of a node in descending order of similarity.
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
        Retrieves top-K IDs and their scores based on a query embedding.

        - First generates a set of seed nodes based on cosine similarity.
        - Then performs a greedy graph walk to expand the set of nodes.
        - Finally sorts the nodes by their similarity and returns the top-K.

        Optional reranking can be performed using a Reranker object.
        This reranks the top-K candidates based on a combination of
        embedding cosine, label Jaccard, and KG-based cosine.

        Parameters:
        - query_emb: query embedding (D,) or (1,D)
        - K: number of top candidates to return (default 5)
        - seed_size: initial number of seed nodes (default 5)
        - max_steps: maximum number of steps in the greedy graph walk (default 100)
        - candidate_multiplier: multiplier for heap size (default 10)
        - reranker: optional Reranker object for reranking (default None)
        - query_id: optional query ID for reranking (default None)
        - rerank_topk: optional number of top candidates to rerank (default None)

        Returns:
        - A list of top-K IDs
        - A list of top-K scores
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
    """
    Returns a RetrievalEngine object based on the method specified.

    Parameters:
    - features_path: path to the embedding features file (numpy .npy file)
    - ids_path: path to the IDs file (json file with list of IDs)
    - method: string specifying the retrieval method (default "dls")
    - link_threshold: float specifying the minimum cosine similarity for a link (default 0.5)
    - max_links: int specifying the maximum number of links per node (default 10)
    - fdb_path: path to the link graph file (pickle file, default None)
    - name: string specifying the name of the retrieval engine (default None)

    Returns:
    - A RetrievalEngine object based on the specified method
    """
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
    ranked = rer.rerank(
        query_id=query_id,
        candidate_ids=cand_ids,
        candidate_embs=cand_embs,
        query_emb=q_emb,
        topk=20
    )
    print("Top 5 after rerank:", ranked[:5])
