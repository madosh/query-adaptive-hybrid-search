"""nDCG@k and related evaluation metrics (Equations 2-4)."""

import numpy as np
from typing import List, Optional


def compute_dcg(relevance_scores: np.ndarray, k: int = 10) -> float:
    """Compute Discounted Cumulative Gain at rank k.

    DCG@k = Σ_{i=1}^{k} rel_i / log2(i + 1)

    Args:
        relevance_scores: Array of relevance scores in ranked order.
        k: Cutoff rank.

    Returns:
        DCG@k value.
    """
    relevance_scores = relevance_scores[:k]
    if len(relevance_scores) == 0:
        return 0.0
    positions = np.arange(1, len(relevance_scores) + 1)
    discounts = np.log2(positions + 1)
    return float(np.sum(relevance_scores / discounts))


def compute_idcg(relevance_scores: np.ndarray, k: int = 10) -> float:
    """Compute Ideal DCG at rank k (DCG of perfect ranking).

    Args:
        relevance_scores: Array of all relevance scores (will be sorted).
        k: Cutoff rank.

    Returns:
        IDCG@k value.
    """
    ideal_order = np.sort(relevance_scores)[::-1]
    return compute_dcg(ideal_order, k)


def compute_ndcg_at_k(relevance_scores: np.ndarray, k: int = 10) -> float:
    """Compute normalized Discounted Cumulative Gain at rank k.

    nDCG@k = DCG@k / IDCG@k

    Args:
        relevance_scores: Array of relevance scores in ranked order.
            For pre-ranked results, pass scores in the current ranking order.
        k: Cutoff rank (default 10).

    Returns:
        nDCG@k value in [0, 1]. Returns 0 if IDCG is 0.
    """
    if len(relevance_scores) == 0:
        return 0.0

    dcg = compute_dcg(relevance_scores, k)
    idcg = compute_idcg(relevance_scores, k)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def compute_ndcg_from_ranking(
    ranked_doc_ids: List[str],
    relevance: dict,
    k: int = 10,
) -> float:
    """Compute nDCG@k from a ranked list of document IDs and a relevance mapping.

    Args:
        ranked_doc_ids: Document IDs in ranked order.
        relevance: Dict mapping doc_id to relevance score.
        k: Cutoff rank.

    Returns:
        nDCG@k value.
    """
    scores = np.array([relevance.get(doc_id, 0) for doc_id in ranked_doc_ids[:k]])
    all_relevant = np.array(list(relevance.values()))
    
    dcg = compute_dcg(scores, k)
    idcg = compute_idcg(all_relevant, k)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def mean_ndcg(per_query_ndcg: List[float]) -> float:
    """Compute mean nDCG across queries."""
    if not per_query_ndcg:
        return 0.0
    return float(np.mean(per_query_ndcg))
