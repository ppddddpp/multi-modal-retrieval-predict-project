import abc
import numpy as np
import json
from typing import Tuple, List
from sklearn.metrics.pairwise import cosine_similarity
import os
from pathlib import Path
import pickle

BASE_DIR = Path(__file__).resolve().parent.parent
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
FEATURES_PATH = BASE_DIR / "featureDBs"

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
    - On query, does a direct cos‑sim scan (or you could do graph traversal here).
    """

    def __init__(
        self,
        features_path: str,
        ids_path: str,
        link_threshold: float = 0.5,
        max_links: int = 10,
        fdb_path: str = None
    ):
        """
        If fdb_path is given, load the precomputed link graph from there.
        Otherwise, compute it using _build_link_graph and save it to the default path.

        Args:
            features_path (str): Path to the feature npy file.
            ids_path (str): Path to the IDs JSON file.
            link_threshold (float, optional): Cosine similarity threshold for links. Defaults to 0.5.
            max_links (int, optional): Maximum number of links per node. Defaults to 10.
            fdb_path (str, optional): Path to the precomputed link graph. Defaults to None.
        """
        super().__init__(features_path, ids_path)
        self.fdb_path = fdb_path if fdb_path else str(FEATURES_PATH / "dls_link_graph.pkl")
        if self.fdb_path and os.path.exists(self.fdb_path):
            self.link_graph = self._build_link_graph(link_threshold, max_links)
            with open(self.fdb_path, "rb") as f:
                self.link_graph = pickle.load(f)
        else:
            self.link_graph = self._build_link_graph(link_threshold, max_links)
            with open(FEATURES_PATH / "dls_link_graph.pkl", "wb") as f:
                pickle.dump(self.link_graph, f)

    def _build_link_graph(self, threshold: float, max_links: int) -> List[List[int]]:
        """
        Precompute a cosine-similarity neighbor graph above a threshold.
        Each node's neighbors are the top-K most similar nodes (by cosine similarity)
        above the given threshold.

        Args:
            threshold (float): minimum cosine similarity to consider a link
            max_links (int): maximum number of links per node

        Returns:
            List[List[int]]: adjacency list representation of the link graph
        """
        sim = cosine_similarity(self.embs)        # (N, N)
        np.fill_diagonal(sim, -1)                 # ignore self

        graph = []
        for i in range(sim.shape[0]):
            # sort by descending similarity
            nbrs = np.argsort(sim[i])[::-1]
            # keep those above threshold
            selected = [int(j) for j in nbrs if sim[i,j] >= threshold][:max_links]
            graph.append(selected)
        return graph

    def retrieve(self, query_emb: np.ndarray, K: int = 5) -> Tuple[List[str], List[float]]:
        """
        Given a query embedding (1, D), return top-K IDs and their distances/similarities
        computed using a direct cosine-similarity scan.

        Args:
            query_emb (np.ndarray): Query embedding (1, D)
            K (int, optional): top-K to return. Defaults to 5.

        Returns:
            Tuple[List[str], List[float]]: top-K IDs and their distances/similarities
        """
        # direct cosine-sim scan
        q = query_emb.astype("float32")[0]
        sims = self.embs @ q / (np.linalg.norm(self.embs, axis=1) * np.linalg.norm(q) + 1e-6)
        topk = np.argsort(sims)[::-1][:K]
        ids   = [self.ids[i] for i in topk]
        scores= [float(sims[i]) for i in topk]
        return ids, scores

def make_retrieval_engine(
    features_path: str,
    ids_path: str,
    method: str = "dls",
    **kwargs
) -> RetrievalEngine:
    method = method.lower()
    if method == "dls":
        return DLSRetrievalEngine(features_path, ids_path,
                                  link_threshold=kwargs.get("link_threshold", 0.5),
                                  max_links=kwargs.get("max_links", 10))
    else:
        raise ValueError(f"Unknown retrieval method: {method}")