"""Score normalization utilities for hybrid retrieval fusion."""

import numpy as np
from typing import Dict, List, Tuple


def minmax_normalize(scores: np.ndarray) -> np.ndarray:
    """Min-max normalization within a candidate pool (Equation 7).

    Normalizes scores to [0, 1] range based on the min and max
    within the top-K retrieved candidates for a single query.
    """
    s_min = scores.min()
    s_max = scores.max()
    if s_max == s_min:
        return np.zeros_like(scores)
    return (scores - s_min) / (s_max - s_min)


def rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion score (Equation 6).

    Args:
        rank: 1-based rank of the document.
        k: Smoothing constant (default 60 per paper).
    """
    return 1.0 / (k + rank)


def rrf_normalize(ranks: np.ndarray, k: int = 60) -> np.ndarray:
    """Compute RRF scores for an array of 1-based ranks."""
    return 1.0 / (k + ranks)


def hybrid_score(
    dense_scores: np.ndarray,
    sparse_scores: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Compute hybrid scores with linear interpolation (Equation 8).

    s_hybrid = α * s_dense_normalized + (1 - α) * s_sparse_normalized
    """
    return alpha * dense_scores + (1.0 - alpha) * sparse_scores


def fuse_results(
    dense_results: Dict[str, float],
    sparse_results: Dict[str, float],
    alpha: float,
    method: str = "minmax",
    rrf_k: int = 60,
) -> List[Tuple[str, float]]:
    """Fuse dense and sparse retrieval results for a single query.

    Args:
        dense_results: Mapping from doc_id to dense score.
        sparse_results: Mapping from doc_id to sparse score.
        alpha: Mixing coefficient (1.0 = pure dense, 0.0 = pure sparse).
        method: "minmax" or "rrf".
        rrf_k: RRF smoothing constant.

    Returns:
        List of (doc_id, hybrid_score) tuples sorted descending by score.
    """
    all_doc_ids = set(dense_results.keys()) | set(sparse_results.keys())

    if method == "minmax":
        dense_ids = list(dense_results.keys())
        sparse_ids = list(sparse_results.keys())

        dense_scores_arr = np.array([dense_results[d] for d in dense_ids])
        sparse_scores_arr = np.array([sparse_results[d] for d in sparse_ids])

        dense_norm = minmax_normalize(dense_scores_arr)
        sparse_norm = minmax_normalize(sparse_scores_arr)

        dense_normed = dict(zip(dense_ids, dense_norm))
        sparse_normed = dict(zip(sparse_ids, sparse_norm))

        fused = {}
        for doc_id in all_doc_ids:
            d_score = dense_normed.get(doc_id, 0.0)
            s_score = sparse_normed.get(doc_id, 0.0)
            fused[doc_id] = alpha * d_score + (1.0 - alpha) * s_score

    elif method == "rrf":
        dense_ranked = sorted(dense_results.keys(), key=lambda d: dense_results[d], reverse=True)
        sparse_ranked = sorted(sparse_results.keys(), key=lambda d: sparse_results[d], reverse=True)

        dense_rrf = {doc_id: rrf_score(rank + 1, rrf_k) for rank, doc_id in enumerate(dense_ranked)}
        sparse_rrf = {doc_id: rrf_score(rank + 1, rrf_k) for rank, doc_id in enumerate(sparse_ranked)}

        fused = {}
        for doc_id in all_doc_ids:
            d_score = dense_rrf.get(doc_id, 0.0)
            s_score = sparse_rrf.get(doc_id, 0.0)
            fused[doc_id] = alpha * d_score + (1.0 - alpha) * s_score
    else:
        raise ValueError(f"Unknown fusion method: {method}")

    return sorted(fused.items(), key=lambda x: x[1], reverse=True)
