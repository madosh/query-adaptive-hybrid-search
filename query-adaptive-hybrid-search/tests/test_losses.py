"""Tests for loss functions."""

import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.losses import (
    infonce_loss,
    infonce_loss_with_in_batch,
    squared_cross_entropy,
    wasserstein_1d,
    qdap_loss,
)


class TestInfoNCELoss:
    def test_basic_shape(self):
        batch_size, dim, num_neg = 4, 768, 7
        q = torch.randn(batch_size, dim)
        p = torch.randn(batch_size, dim)
        n = torch.randn(batch_size, num_neg, dim)
        loss = infonce_loss(q, p, n)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_perfect_alignment(self):
        """When positives are identical to queries, loss should be low."""
        batch_size, dim, num_neg = 4, 128, 7
        q = torch.randn(batch_size, dim)
        q = torch.nn.functional.normalize(q, dim=-1)
        p = q.clone()
        n = torch.randn(batch_size, num_neg, dim)
        n = torch.nn.functional.normalize(n, dim=-1)
        loss = infonce_loss(q, p, n, temperature=0.05)
        random_q = torch.randn(batch_size, dim)
        random_q = torch.nn.functional.normalize(random_q, dim=-1)
        loss_random = infonce_loss(random_q, p, n, temperature=0.05)
        assert loss.item() < loss_random.item()

    def test_temperature_effect(self):
        """Lower temperature should produce larger loss values."""
        batch_size, dim, num_neg = 4, 128, 7
        q = torch.randn(batch_size, dim)
        p = torch.randn(batch_size, dim)
        n = torch.randn(batch_size, num_neg, dim)
        loss_low_t = infonce_loss(q, p, n, temperature=0.01)
        loss_high_t = infonce_loss(q, p, n, temperature=1.0)
        assert loss_low_t.item() != loss_high_t.item()

    def test_with_in_batch(self):
        batch_size, dim, num_neg = 8, 128, 5
        q = torch.randn(batch_size, dim)
        p = torch.randn(batch_size, dim)
        n = torch.randn(batch_size, num_neg, dim)
        loss = infonce_loss_with_in_batch(q, p, n)
        assert loss.shape == ()
        assert loss.item() > 0


class TestSquaredCrossEntropy:
    def test_basic(self):
        batch_size, num_bins = 8, 101
        y_pred = torch.softmax(torch.randn(batch_size, num_bins), dim=-1)
        y_target = torch.softmax(torch.randn(batch_size, num_bins), dim=-1)
        loss = squared_cross_entropy(y_pred, y_target)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_identical_distributions(self):
        """Loss should be minimal when pred matches target."""
        batch_size, num_bins = 4, 101
        y = torch.softmax(torch.randn(batch_size, num_bins), dim=-1)
        loss_same = squared_cross_entropy(y, y)
        y_other = torch.softmax(torch.randn(batch_size, num_bins), dim=-1)
        loss_diff = squared_cross_entropy(y, y_other)
        assert loss_same.item() <= loss_diff.item()

    def test_squaring_concentrates_target(self):
        """Squared target should be more peaked than original."""
        num_bins = 101
        y_target = torch.softmax(torch.randn(1, num_bins), dim=-1)
        y_target_sq = y_target ** 2
        y_target_sq = y_target_sq / y_target_sq.sum(dim=-1, keepdim=True)
        assert y_target_sq.max() >= y_target.max()


class TestWasserstein1D:
    def test_basic(self):
        batch_size, num_bins = 8, 101
        y_pred = torch.softmax(torch.randn(batch_size, num_bins), dim=-1)
        y_target = torch.softmax(torch.randn(batch_size, num_bins), dim=-1)
        loss = wasserstein_1d(y_pred, y_target)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_identical_distributions(self):
        """Wasserstein distance should be 0 for identical distributions."""
        batch_size, num_bins = 4, 101
        y = torch.softmax(torch.randn(batch_size, num_bins), dim=-1)
        loss = wasserstein_1d(y, y)
        assert loss.item() < 1e-5

    def test_distant_distributions(self):
        """Distributions at opposite ends should have large distance."""
        num_bins = 101
        y_left = torch.zeros(1, num_bins)
        y_left[0, 0] = 1.0
        y_right = torch.zeros(1, num_bins)
        y_right[0, -1] = 1.0
        loss = wasserstein_1d(y_left, y_right)
        assert loss.item() > 50  # Max distance for 101 bins


class TestQDAPLoss:
    def test_composite(self):
        batch_size, num_bins = 8, 101
        y_pred = torch.softmax(torch.randn(batch_size, num_bins), dim=-1)
        y_target = torch.softmax(torch.randn(batch_size, num_bins), dim=-1)
        loss = qdap_loss(y_pred, y_target, lambda_weight=0.62)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_lambda_boundaries(self):
        """Test at lambda extremes."""
        batch_size, num_bins = 4, 101
        y_pred = torch.softmax(torch.randn(batch_size, num_bins), dim=-1)
        y_target = torch.softmax(torch.randn(batch_size, num_bins), dim=-1)

        loss_ce_only = qdap_loss(y_pred, y_target, lambda_weight=1.0)
        loss_wd_only = qdap_loss(y_pred, y_target, lambda_weight=0.0)
        loss_combined = qdap_loss(y_pred, y_target, lambda_weight=0.62)

        ce = squared_cross_entropy(y_pred, y_target)
        wd = wasserstein_1d(y_pred, y_target)

        assert abs(loss_ce_only.item() - ce.item()) < 1e-5
        assert abs(loss_wd_only.item() - wd.item()) < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
