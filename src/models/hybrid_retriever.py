"""End-to-end hybrid retrieval pipeline combining dense, sparse, and QDAP."""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union

from ..utils.score_normalization import minmax_normalize, fuse_results


class HybridRetriever:
    """Adaptive hybrid retriever that fuses BM25 and dense signals using QDAP-predicted alpha.

    The pipeline:
    1. Retrieve top-K candidates from both dense and sparse retrievers.
    2. Union the candidate sets.
    3. Normalize scores (min-max within the candidate pool).
    4. Predict alpha from the query embedding via QDAP.
    5. Compute hybrid scores: α * dense + (1 - α) * sparse.
    6. Return top-k results by hybrid score.
    """

    def __init__(
        self,
        dense_encoder,
        bm25_index,
        qdap_model,
        fusion: str = "minmax",
        rrf_k: int = 60,
        pool_size: int = 100,
        device: Optional[torch.device] = None,
    ):
        self.dense = dense_encoder
        self.bm25 = bm25_index
        self.qdap = qdap_model
        self.fusion = fusion
        self.rrf_k = rrf_k
        self.pool_size = pool_size
        self.device = device or torch.device("cpu")

        if hasattr(self.dense, "eval"):
            self.dense.eval()
        if hasattr(self.qdap, "eval"):
            self.qdap.eval()

    @torch.no_grad()
    def predict_alpha(self, query: str) -> float:
        """Predict the mixing coefficient alpha for a given query."""
        from .qdap_l import QDAP_L
        from .qdap_s import QDAP_S

        if isinstance(self.qdap, QDAP_L):
            encoded = self.qdap.tokenizer(
                query, padding=True, truncation=True,
                max_length=512, return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            alpha = self.qdap.predict_alpha(encoded["input_ids"], encoded["attention_mask"])
        elif isinstance(self.qdap, QDAP_S):
            q_emb = self.dense.encode_texts(query)
            q_emb_tensor = torch.tensor(q_emb, device=self.device)
            alpha = self.qdap.predict_alpha(q_emb_tensor)
        else:
            raise ValueError(f"Unknown QDAP model type: {type(self.qdap)}")

        return alpha.item()

    def search(
        self,
        query: str,
        top_k: int = 10,
        alpha: Optional[float] = None,
    ) -> List[Tuple[str, float]]:
        """Perform hybrid search for a query.

        Args:
            query: Query text.
            top_k: Number of results to return.
            alpha: If provided, override QDAP prediction with this value.

        Returns:
            List of (doc_id, score) tuples sorted by hybrid score.
        """
        dense_results = self._get_dense_results(query)
        sparse_results = self._get_sparse_results(query)

        if alpha is None:
            alpha = self.predict_alpha(query)

        fused = fuse_results(
            dense_results=dense_results,
            sparse_results=sparse_results,
            alpha=alpha,
            method=self.fusion,
            rrf_k=self.rrf_k,
        )

        return fused[:top_k]

    def search_batch(
        self,
        queries: List[str],
        top_k: int = 10,
    ) -> List[List[Tuple[str, float]]]:
        """Batch hybrid search for multiple queries."""
        results = []
        for query in queries:
            results.append(self.search(query, top_k=top_k))
        return results

    def _get_dense_results(self, query: str) -> Dict[str, float]:
        """Retrieve top-K candidates from the dense retriever."""
        q_emb = self.dense.encode_texts(query)
        scores, doc_ids = self.bm25.dense_index_search(q_emb, self.pool_size)
        return dict(zip(doc_ids, scores.tolist()))

    def _get_sparse_results(self, query: str) -> Dict[str, float]:
        """Retrieve top-K candidates from the BM25 retriever."""
        results = self.bm25.search(query, top_k=self.pool_size)
        return {doc_id: score for doc_id, score in results}

    def evaluate_single(
        self,
        query: str,
        relevance: Dict[str, int],
        top_k: int = 10,
        alpha: Optional[float] = None,
    ) -> Dict[str, float]:
        """Evaluate a single query and return metrics.

        Args:
            query: Query text.
            relevance: Mapping from doc_id to relevance grade.
            top_k: Cutoff for evaluation.
            alpha: Optional override for QDAP alpha.

        Returns:
            Dictionary with metrics (e.g., ndcg@10).
        """
        from ..evaluation.metrics import compute_ndcg_at_k

        results = self.search(query, top_k=top_k, alpha=alpha)
        ranked_doc_ids = [doc_id for doc_id, _ in results]
        relevance_scores = [relevance.get(doc_id, 0) for doc_id in ranked_doc_ids]

        ndcg = compute_ndcg_at_k(np.array(relevance_scores), k=top_k)
        predicted_alpha = alpha if alpha is not None else self.predict_alpha(query)

        return {"ndcg@10": ndcg, "alpha": predicted_alpha}
