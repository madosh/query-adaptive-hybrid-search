"""Oracle and optimal-α upper bound computation."""

import numpy as np
from typing import Dict, List, Tuple

from ..utils.score_normalization import minmax_normalize
from .metrics import compute_ndcg_at_k


def compute_oracle_alpha(
    dense_scores: np.ndarray,
    sparse_scores: np.ndarray,
    relevance_labels: np.ndarray,
    num_bins: int = 101,
    k: int = 10,
) -> Tuple[float, float]:
    """Compute the oracle (per-query optimal) alpha and its nDCG@10.

    Sweeps α from 0 to 1 (step 0.01), picks the α that maximizes
    nDCG@10 for this specific query. This is the theoretical upper bound.

    Args:
        dense_scores: Dense retrieval scores for candidates.
        sparse_scores: Sparse retrieval scores for candidates.
        relevance_labels: Relevance labels for candidates.
        num_bins: Number of alpha values to try.
        k: nDCG cutoff.

    Returns:
        Tuple of (best_alpha, best_ndcg).
    """
    dense_norm = minmax_normalize(dense_scores)
    sparse_norm = minmax_normalize(sparse_scores)

    best_alpha = 0.0
    best_ndcg = 0.0

    for alpha_i in np.linspace(0, 1, num_bins):
        hybrid = alpha_i * dense_norm + (1.0 - alpha_i) * sparse_norm
        ranking = np.argsort(-hybrid)
        ranked_labels = relevance_labels[ranking]
        ndcg = compute_ndcg_at_k(ranked_labels, k=k)
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_alpha = alpha_i

    return best_alpha, best_ndcg


def compute_optimal_alpha(
    all_dense_scores: List[np.ndarray],
    all_sparse_scores: List[np.ndarray],
    all_relevance_labels: List[np.ndarray],
    num_bins: int = 101,
    k: int = 10,
) -> Tuple[float, float]:
    """Find the single α that maximizes average nDCG@10 over all queries.

    This is the "Optimal" baseline — one global α for the entire dataset.

    Args:
        all_dense_scores: List of per-query dense score arrays.
        all_sparse_scores: List of per-query sparse score arrays.
        all_relevance_labels: List of per-query relevance label arrays.
        num_bins: Number of alpha values to try.
        k: nDCG cutoff.

    Returns:
        Tuple of (optimal_alpha, mean_ndcg_at_that_alpha).
    """
    num_queries = len(all_dense_scores)
    alphas = np.linspace(0, 1, num_bins)
    mean_ndcgs = np.zeros(num_bins)

    for i, alpha_i in enumerate(alphas):
        ndcgs = []
        for q_idx in range(num_queries):
            dense_norm = minmax_normalize(all_dense_scores[q_idx])
            sparse_norm = minmax_normalize(all_sparse_scores[q_idx])
            hybrid = alpha_i * dense_norm + (1.0 - alpha_i) * sparse_norm
            ranking = np.argsort(-hybrid)
            ranked_labels = all_relevance_labels[q_idx][ranking]
            ndcgs.append(compute_ndcg_at_k(ranked_labels, k=k))
        mean_ndcgs[i] = np.mean(ndcgs)

    best_idx = np.argmax(mean_ndcgs)
    return float(alphas[best_idx]), float(mean_ndcgs[best_idx])


def compute_oracle_results(
    all_dense_scores: List[np.ndarray],
    all_sparse_scores: List[np.ndarray],
    all_relevance_labels: List[np.ndarray],
    num_bins: int = 101,
    k: int = 10,
) -> Dict[str, float]:
    """Compute full oracle evaluation (per-query best α).

    Returns:
        Dict with "oracle_mean_ndcg", "oracle_alpha_mean", "oracle_alpha_std".
    """
    oracle_ndcgs = []
    oracle_alphas = []

    for q_idx in range(len(all_dense_scores)):
        alpha, ndcg = compute_oracle_alpha(
            all_dense_scores[q_idx],
            all_sparse_scores[q_idx],
            all_relevance_labels[q_idx],
            num_bins=num_bins,
            k=k,
        )
        oracle_ndcgs.append(ndcg)
        oracle_alphas.append(alpha)

    return {
        "oracle_mean_ndcg": float(np.mean(oracle_ndcgs)),
        "oracle_alpha_mean": float(np.mean(oracle_alphas)),
        "oracle_alpha_std": float(np.std(oracle_alphas)),
    }
