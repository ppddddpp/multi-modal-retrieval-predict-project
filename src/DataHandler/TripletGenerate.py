import random
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

class PseudoTripletDataset(Dataset):
    """
    Build weakly-supervised (query, positive, negative) triplets
    from a labels DataFrame.

    Each __getitem__ returns (qid, pid, nid) as strings.
    """
    def __init__(self, labels_df, min_overlap=0.5, seed=None):
        """
        labels_df: pandas DataFrame with index=report_id and columns=labels (0/1)
        min_overlap: Jaccard threshold for positives
        seed: optional random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
        self.labels_df = labels_df
        self.report_ids = list(labels_df.index.astype(str))
        self.min_overlap = min_overlap

        self.triplets = self._build_triplets()

    def _get_labels(self, rid):
        row = self.labels_df.loc[rid]
        return set(row.index[row == 1])

    def _build_triplets(self):
        triplets = []
        for qid in tqdm(self.report_ids, desc="Building triplets"):
            q_labels = self._get_labels(qid)
            if not q_labels:  # skip empty label sets
                continue
            pos_ids = []
            neg_ids = []
            for rid in self.report_ids:
                if rid == qid:
                    continue
                r_labels = self._get_labels(rid)
                if not r_labels:
                    continue
                inter = len(q_labels & r_labels)
                uni = len(q_labels | r_labels)
                jaccard = inter / uni if uni > 0 else 0
                if jaccard >= self.min_overlap:
                    pos_ids.append(rid)
                elif inter == 0:
                    neg_ids.append(rid)
            if pos_ids and neg_ids:
                pos_id = random.choice(pos_ids)
                neg_id = random.choice(neg_ids)
                triplets.append((qid, pos_id, neg_id))
        return triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        """
        Returns (qid, pid, nid) as strings.
        """
        return self.triplets[idx]

    def get_triplets(self):
        """
        Return the full list of triplets [(qid, pid, nid), ...].
        """
        return self.triplets

class LabelEmbeddingLookup:
    """
    Given a labels_df and a label_embedding_dict,
    returns stacked label embeddings for each report.
    """
    def __init__(self, labels_df, label_emb_dict, device="cpu"):
        self.labels_df = labels_df
        self.label_emb_dict = label_emb_dict  # dict: label -> np.array / torch.Tensor
        self.device = device

    def get_label_embs(self, report_id):
        """
        Returns tensor [n_labels, d] of label embeddings for this report.
        If report has no known labels, returns zeros [1,d].
        """
        labels = set(self.labels_df.loc[report_id].index[self.labels_df.loc[report_id] == 1])
        vecs = []
        for lab in labels:
            if lab in self.label_emb_dict:
                v = self.label_emb_dict[lab]
                if not torch.is_tensor(v):
                    v = torch.tensor(v, dtype=torch.float32)
                vecs.append(v)
        if not vecs:
            d = next(iter(self.label_emb_dict.values())).shape[0]
            return torch.zeros((1, d), dtype=torch.float32, device=self.device)
        return torch.stack(vecs).to(self.device)