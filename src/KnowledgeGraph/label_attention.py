import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelAttention(nn.Module):
    """
    Compute a weighted sum of label embeddings for a report.
    - input: tensor [batch, n_labels, d] of label embeddings
    - output: tensor [batch, d] of report embeddings
    """
    def __init__(self, d_emb, hidden=64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_emb, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)  # scalar score per label
        )

    def forward(self, label_embs, mask=None):
        # label_embs: [batch, n_labels, d]
        scores = self.attn(label_embs).squeeze(-1)  # [batch, n_labels]
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        weights = F.softmax(scores, dim=1)  # [batch, n_labels]
        # weighted sum of label embeddings
        report_emb = torch.bmm(weights.unsqueeze(1), label_embs).squeeze(1)  # [batch, d]
        return report_emb, weights  # optionally return weights to inspect