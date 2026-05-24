"""Tests for score normalization utilities."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.score_normalization import (
    minmax_normalize,
    rrf_score,
    rrf_normalize,
    hybrid_score,
    fuse_results,
)


class TestMinMaxNormalize:
    def test_basic(self):
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normed = minmax_normalize(scores)
        assert normed.min() == 0.0
        assert normed.max() == 1.0
        np.testing.assert_allclose(normed, [0.0, 0.25, 0.5, 0.75, 1.0])

    def test_all_same(self):
        """When all scores are the same, should return zeros."""
        scores = np.array([3.0, 3.0, 3.0])
        normed = minmax_normalize(scores)
        np.testing.assert_array_equal(normed, [0.0, 0.0, 0.0])

    def test_single_element(self):
        scores = np.array([5.0])
        normed = minmax_normalize(scores)
        assert normed[0] == 0.0

    def test_negative_scores(self):
        scores = np.array([-5.0, -2.0, 0.0, 3.0])
        normed = minmax_normalize(scores)
        assert normed.min() == 0.0
        assert normed.max() == 1.0
        assert normed[0] == 0.0
        assert normed[-1] == 1.0

    def test_preserves_order(self):
        scores = np.array([10.0, 5.0, 8.0, 1.0, 3.0])
        normed = minmax_normalize(scores)
        original_order = np.argsort(-scores)
        normalized_order = np.argsort(-normed)
        np.testing.assert_array_equal(original_order, normalized_order)


class TestRRF:
    def test_basic(self):
        assert rrf_score(1, k=60) == 1.0 / 61
        assert rrf_score(2, k=60) == 1.0 / 62

    def test_monotonically_decreasing(self):
        scores = [rrf_score(r) for r in range(1, 101)]
        for i in range(len(scores) - 1):
            assert scores[i] > scores[i + 1]

    def test_custom_k(self):
        assert rrf_score(1, k=0) == 1.0
        assert rrf_score(1, k=100) == 1.0 / 101

    def test_array_version(self):
        ranks = np.array([1, 2, 3, 4, 5])
        scores = rrf_normalize(ranks, k=60)
        expected = 1.0 / (60 + ranks)
        np.testing.assert_allclose(scores, expected)


class TestHybridScore:
    def test_pure_dense(self):
        dense = np.array([0.8, 0.6, 0.4])
        sparse = np.array([0.2, 0.9, 0.1])
        result = hybrid_score(dense, sparse, alpha=1.0)
        np.testing.assert_allclose(result, dense)

    def test_pure_sparse(self):
        dense = np.array([0.8, 0.6, 0.4])
        sparse = np.array([0.2, 0.9, 0.1])
        result = hybrid_score(dense, sparse, alpha=0.0)
        np.testing.assert_allclose(result, sparse)

    def test_equal_weight(self):
        dense = np.array([1.0, 0.0])
        sparse = np.array([0.0, 1.0])
        result = hybrid_score(dense, sparse, alpha=0.5)
        np.testing.assert_allclose(result, [0.5, 0.5])

    def test_range(self):
        """Hybrid of normalized scores should stay in [0, 1]."""
        dense = np.random.rand(100)
        sparse = np.random.rand(100)
        for alpha in np.linspace(0, 1, 11):
            result = hybrid_score(dense, sparse, alpha)
            assert result.min() >= 0.0
            assert result.max() <= 1.0


class TestFuseResults:
    def test_minmax_fusion(self):
        dense = {"doc1": 0.9, "doc2": 0.7, "doc3": 0.3}
        sparse = {"doc1": 0.2, "doc2": 0.8, "doc4": 0.6}
        results = fuse_results(dense, sparse, alpha=0.5, method="minmax")
        assert len(results) == 4
        assert results[0][1] >= results[1][1]

    def test_rrf_fusion(self):
        dense = {"doc1": 0.9, "doc2": 0.7, "doc3": 0.3}
        sparse = {"doc1": 0.2, "doc2": 0.8, "doc4": 0.6}
        results = fuse_results(dense, sparse, alpha=0.5, method="rrf")
        assert len(results) == 4

    def test_alpha_one_favors_dense(self):
        dense = {"doc1": 1.0, "doc2": 0.0}
        sparse = {"doc1": 0.0, "doc2": 1.0}
        results = fuse_results(dense, sparse, alpha=1.0, method="minmax")
        assert results[0][0] == "doc1"

    def test_alpha_zero_favors_sparse(self):
        dense = {"doc1": 1.0, "doc2": 0.0}
        sparse = {"doc1": 0.0, "doc2": 1.0}
        results = fuse_results(dense, sparse, alpha=0.0, method="minmax")
        assert results[0][0] == "doc2"

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            fuse_results({}, {}, alpha=0.5, method="invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
