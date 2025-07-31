import abc
import numpy as np
import json
from typing import Tuple, List
from sklearn.metrics.pairwise import cosine_similarity
import os
from pathlib import Path
import pickle

# Base dirs
BASE_DIR        = Path(__file__).resolve().parent.parent
EMBEDDINGS_DIR  = BASE_DIR / "embeddings"
FEATURES_PATH   = BASE_DIR / "featureDBs"


class RetrievalEngine(abc.ABC):
    """
    Abstract base class for retrieval engines.
    """

    def __init__(self, features_path: str, ids_path: str):
        # load embeddings and IDs once
        self.embs = np.load(features_path).astype("float32")  # shape (N, D)
        with open(ids_path, "r") as f:
            self.ids = json.load(f)                             # length N

    @abc.abstractmethod
    def retrieve(self, query_emb: np.ndarray, K: int = 5) -> Tuple[List[str], List[float]]:
        """
        Given a query embedding (1, D), return top-K IDs and their distances/similarities.
        """
        pass


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
        fdb_path: str = None,
        name: str = None
    ):
        """
        If fdb_path is given, use it. Otherwise:
          - if `name` is provided, use FEATURES_PATH/name
          - else derive from the features_path stem: FEATURES_PATH/{stem}_link_graph.pkl

        Args:
            features_path (str): Path to the feature npy file.
            ids_path      (str): Path to the IDs JSON file.
            link_threshold(float): Cosine-sim threshold for links.
            max_links    (int): Max links per node.
            fdb_path     (str): Explicit path to load/save the graph.
            name         (str): Filename under FEATURES_PATH to load/save.
        """
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
        candidate_multiplier: int = 10
    ) -> Tuple[List[str], List[float]]:
        """
        Sublinear DLS retrieval via greedy graph-walk.

        Args:
            query_emb          (np.ndarray): Query embedding (1, D)
            K                       (int): Number of results
            seed_size               (int): Initial random seeds
            max_steps               (int): Max graph-walk iterations
            candidate_multiplier    (int): How many candidates to keep each round
        """
        # flatten and normalize
        q = query_emb.astype("float32").reshape(-1)
        N = self.embs.shape[0]

        # sanity-check graph length
        if len(self.link_graph) != N:
            raise RuntimeError(
                f"Link graph size {len(self.link_graph)} != embeddings {N}. "
                "Rebuild or delete your pickle."
            )

        # Sample seeds
        seeds = np.random.choice(N, size=seed_size, replace=False).tolist()
        visited = set(seeds)

        # Score seeds
        scored = []
        for idx in seeds:
            sim = float(self.embs[idx] @ q / (np.linalg.norm(self.embs[idx]) * np.linalg.norm(q) + 1e-6))
            scored.append((sim, idx))
        scored.sort(reverse=True)

        # Greedily traverse the graph
        steps = 0
        while steps < max_steps:
            best_sim, best_idx = scored[0]
            improved = False

            for nbr in self.link_graph[best_idx]:
                # skip invalid or seen
                if nbr < 0 or nbr >= N or nbr in visited:
                    continue
                visited.add(nbr)
                nbr_sim = float(self.embs[nbr] @ q / (np.linalg.norm(self.embs[nbr]) * np.linalg.norm(q) + 1e-6))
                scored.append((nbr_sim, nbr))
                improved = True

            if not improved:
                break

            # prune to top candidates
            scored.sort(reverse=True)
            scored = scored[: max(candidate_multiplier * K, seed_size)]
            steps += 1
            
        topk  = scored[:K]
        ids   = [ self.ids[idx] for _, idx in topk ]
        scores= [ sim      for sim, _ in topk ]
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
