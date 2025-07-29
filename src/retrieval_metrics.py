import numpy as np
from typing import List, Set, Tuple

def precision_at_k(retrieved_ids, relevant_ids, k=5):
    """
    Precision@k = (# relevant in top-k) / k
    """
    retrieved_topk = retrieved_ids[:k]
    rel_set = set(relevant_ids)
    num_relevant = sum([1 for r in retrieved_topk if r in rel_set])
    return num_relevant / k

def recall_at_k(retrieved_ids, relevant_ids, k=5):
    """
    Recall@k = (# relevant in top-k) / total relevant
    """
    rel_set = set(relevant_ids)
    if len(rel_set) == 0:
        return 0.0
    retrieved_topk = retrieved_ids[:k]
    num_relevant = sum([1 for r in retrieved_topk if r in rel_set])
    return num_relevant / len(rel_set)

def average_precision(retrieved: List[str], relevant: Set[str], k: int = None) -> float:
    """
    AP = sum_{i=1..K} [Precision@i * rel(i)] / (# relevant)
    where rel(i)=1 if retrieved[i] is relevant else 0.
    If k is None, we use all retrieved results.
    """
    if k is None:
        k = len(retrieved)
    hits = 0
    score = 0.0
    for i, r in enumerate(retrieved[:k], start=1):
        if r in relevant:
            hits += 1
            score += hits / i
    return score / len(relevant) if relevant else 0.0

def mean_average_precision(
    all_retrieved: List[List[str]],
    all_relevant:  List[Set[str]],
    k: int = None
) -> float:
    """
    mAP over a set of queries.
    all_retrieved[i] is the retrieved list for query i
    all_relevant[i]  is the set of relevant IDs for query i
    """
    APs = [
        average_precision(ret, rel, k)
        for ret, rel in zip(all_retrieved, all_relevant)
    ]
    return float(np.mean(APs))

def mean_reciprocal_rank(
    all_retrieved: List[List[str]],
    all_relevant:  List[Set[str]]
) -> float:
    """
    MRR = mean( 1 / rank_i ), where rank_i is the first position of a relevant item.
    If none relevant found, reciprocal rank is 0 for that query.
    """
    rr_list = []
    for retrieved, relevant in zip(all_retrieved, all_relevant):
        rr = 0.0
        for i, r in enumerate(retrieved, start=1):
            if r in relevant:
                rr = 1.0 / i
                break
        rr_list.append(rr)
    return float(np.mean(rr_list))