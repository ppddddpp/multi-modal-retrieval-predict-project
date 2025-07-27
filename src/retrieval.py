import numpy as np
import json
from typing import Tuple, List, Literal

class RetrievalEngine:
    def __init__(
        self,
        features_path: str,
        ids_path: str,
        method: Literal["dls"] = "dls",
        dls_link_thresh: float = 0.5,
        dls_max_links: int = 10
    ):
        """
        Unified interface for image-text retrieval.

        Parameters:
        - features_path: path to .npy file of database embeddings
        - ids_path: path to .json file of corresponding sample IDs
        - method: one of "faiss" or "dls"
        - dls_link_thresh: cosine similarity threshold for DenseLinkSearch
        - dls_max_links: max number of linked neighbors to expand
        """
        self.features_path = features_path
        self.ids_path = ids_path
        self.method = method.lower()

        self.embs = np.load(self.features_path).astype("float32")
        with open(self.ids_path, "r") as f:
            self.ids = json.load(f)

        if self.method == "dls":
            self.dls_links = self._build_dls_links(
                threshold=dls_link_thresh, max_links=dls_max_links
            )
        else:
            raise ValueError(f"Unknown retrieval method: {self.method}")

    def _build_dls_links(self, threshold: float, max_links: int):
        """Construct a simple DenseLinkSearch neighbor graph using cosine similarity."""
        from sklearn.metrics.pairwise import cosine_similarity

        sim_matrix = cosine_similarity(self.embs)
        np.fill_diagonal(sim_matrix, -1)  # exclude self-links

        link_graph = []
        for i in range(sim_matrix.shape[0]):
            neighbors = np.argsort(sim_matrix[i])[::-1]
            selected = [
                int(n) for n in neighbors
                if sim_matrix[i, n] >= threshold
            ][:max_links]
            link_graph.append(selected)
        return link_graph

    def retrieve(self, query_emb: np.ndarray, K: int = 5) -> Tuple[List[str], List[float]]:
        """
        Retrieve top-K closest items.

        Args:
            query_emb: shape (1, D)
            K: number of top items to return

        Returns:
            List of sample IDs, List of distances/similarities
        """
        query_emb = query_emb.astype("float32")
        if self.method == "dls":
            # Flatten brute force for demo
            query = query_emb[0]
            sims = self.embs @ query / (
                np.linalg.norm(self.embs, axis=1) * np.linalg.norm(query) + 1e-6
            )
            topk_idx = np.argsort(sims)[::-1][:K]
            idxs = topk_idx.tolist()
            dists = [float(sims[i]) for i in idxs]
        else:
            raise ValueError(f"Unknown retrieval method: {self.method}")

        return [self.ids[i] for i in idxs], dists
