"""Full evaluation pipeline for the hybrid retrieval system."""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from tqdm import tqdm

from ..models.hybrid_retriever import HybridRetriever
from ..utils.score_normalization import minmax_normalize
from .metrics import compute_ndcg_at_k, compute_ndcg_from_ranking, mean_ndcg
from .oracle import compute_oracle_alpha, compute_optimal_alpha, compute_oracle_results


def evaluate_hybrid_retriever(
    retriever: HybridRetriever,
    queries: List[str],
    relevance: List[Dict[str, int]],
    top_k: int = 10,
) -> Dict[str, float]:
    """Evaluate the hybrid retriever with QDAP-predicted alpha.

    Args:
        retriever: HybridRetriever instance.
        queries: List of query strings.
        relevance: List of dicts mapping doc_id to relevance grade.
        top_k: Evaluation cutoff.

    Returns:
        Dict with evaluation metrics.
    """
    ndcgs = []
    alphas = []

    for query, rel in tqdm(zip(queries, relevance), total=len(queries), desc="Evaluating"):
        results = retriever.search(query, top_k=top_k)
        ranked_doc_ids = [doc_id for doc_id, _ in results]
        ndcg = compute_ndcg_from_ranking(ranked_doc_ids, rel, k=top_k)
        ndcgs.append(ndcg)
        alphas.append(retriever.predict_alpha(query))

    return {
        "ndcg@10": float(np.mean(ndcgs)),
        "ndcg@10_std": float(np.std(ndcgs)),
        "alpha_mean": float(np.mean(alphas)),
        "alpha_std": float(np.std(alphas)),
        "num_queries": len(queries),
    }


def evaluate_static_alpha(
    dense_scores: List[np.ndarray],
    sparse_scores: List[np.ndarray],
    relevance_labels: List[np.ndarray],
    alpha: float,
    k: int = 10,
) -> float:
    """Evaluate with a fixed (static) alpha value.

    Args:
        dense_scores: Per-query dense score arrays.
        sparse_scores: Per-query sparse score arrays.
        relevance_labels: Per-query relevance label arrays.
        alpha: Static mixing coefficient.
        k: nDCG cutoff.

    Returns:
        Mean nDCG@k.
    """
    ndcgs = []
    for q_idx in range(len(dense_scores)):
        d_norm = minmax_normalize(dense_scores[q_idx])
        s_norm = minmax_normalize(sparse_scores[q_idx])
        hybrid = alpha * d_norm + (1.0 - alpha) * s_norm
        ranking = np.argsort(-hybrid)
        ranked_labels = relevance_labels[q_idx][ranking]
        ndcgs.append(compute_ndcg_at_k(ranked_labels, k=k))
    return float(np.mean(ndcgs))


def full_evaluation(
    dense_scores: List[np.ndarray],
    sparse_scores: List[np.ndarray],
    relevance_labels: List[np.ndarray],
    predicted_alphas: List[float],
    k: int = 10,
    num_bins: int = 101,
    output_dir: Optional[str] = None,
) -> Dict[str, float]:
    """Run the full evaluation suite: QDAP, BM25-only, Dense-only, Optimal, Oracle.

    Args:
        dense_scores: Per-query dense score arrays.
        sparse_scores: Per-query sparse score arrays.
        relevance_labels: Per-query relevance label arrays.
        predicted_alphas: QDAP-predicted alphas per query.
        k: nDCG cutoff.
        num_bins: Number of bins for oracle/optimal search.
        output_dir: Optional directory to save results.

    Returns:
        Dict with all evaluation results.
    """
    qdap_ndcgs = []
    for q_idx in range(len(dense_scores)):
        d_norm = minmax_normalize(dense_scores[q_idx])
        s_norm = minmax_normalize(sparse_scores[q_idx])
        alpha = predicted_alphas[q_idx]
        hybrid = alpha * d_norm + (1.0 - alpha) * s_norm
        ranking = np.argsort(-hybrid)
        ranked_labels = relevance_labels[q_idx][ranking]
        qdap_ndcgs.append(compute_ndcg_at_k(ranked_labels, k=k))

    bm25_only = evaluate_static_alpha(dense_scores, sparse_scores, relevance_labels, alpha=0.0, k=k)
    dense_only = evaluate_static_alpha(dense_scores, sparse_scores, relevance_labels, alpha=1.0, k=k)
    optimal_alpha, optimal_ndcg = compute_optimal_alpha(
        dense_scores, sparse_scores, relevance_labels, num_bins=num_bins, k=k
    )
    oracle = compute_oracle_results(dense_scores, sparse_scores, relevance_labels, num_bins=num_bins, k=k)

    results = {
        "qdap_ndcg@10": float(np.mean(qdap_ndcgs)),
        "qdap_ndcg@10_std": float(np.std(qdap_ndcgs)),
        "bm25_only_ndcg@10": bm25_only,
        "dense_only_ndcg@10": dense_only,
        "optimal_alpha": optimal_alpha,
        "optimal_ndcg@10": optimal_ndcg,
        "oracle_ndcg@10": oracle["oracle_mean_ndcg"],
        "oracle_alpha_mean": oracle["oracle_alpha_mean"],
        "oracle_alpha_std": oracle["oracle_alpha_std"],
        "predicted_alpha_mean": float(np.mean(predicted_alphas)),
        "predicted_alpha_std": float(np.std(predicted_alphas)),
        "num_queries": len(dense_scores),
    }

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / "results.json", "w") as f:
            json.dump(results, f, indent=2)

        per_query = []
        for q_idx in range(len(dense_scores)):
            per_query.append({
                "query_idx": q_idx,
                "qdap_ndcg": qdap_ndcgs[q_idx],
                "predicted_alpha": predicted_alphas[q_idx],
            })
        with open(output_path / "per_query_results.json", "w") as f:
            json.dump(per_query, f, indent=2)

    return results
