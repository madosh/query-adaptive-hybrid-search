"""Tests for QDAP models."""

import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.qdap_s import QDAP_S
from src.models.qdap_l import QDAP_L


class TestQDAP_S:
    def setup_method(self):
        self.model = QDAP_S(embedding_dim=768, num_bins=101, conv_kernel=7)

    def test_forward_shape(self):
        batch_size = 4
        x = torch.randn(batch_size, 768)
        out = self.model(x)
        assert out.shape == (batch_size, 101)

    def test_output_is_distribution(self):
        """Output should sum to 1 (valid probability distribution)."""
        x = torch.randn(8, 768)
        out = self.model(x)
        sums = out.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_output_non_negative(self):
        """All probabilities should be non-negative."""
        x = torch.randn(8, 768)
        out = self.model(x)
        assert (out >= 0).all()

    def test_predict_alpha_range(self):
        """Predicted alpha should be in [0, 1]."""
        x = torch.randn(16, 768)
        alphas = self.model.predict_alpha(x)
        assert alphas.shape == (16,)
        assert (alphas >= 0).all()
        assert (alphas <= 1).all()

    def test_predict_alpha_argmax(self):
        """Argmax alpha should also be in [0, 1]."""
        x = torch.randn(8, 768)
        alphas = self.model.predict_alpha_argmax(x)
        assert (alphas >= 0).all()
        assert (alphas <= 1).all()

    def test_gradient_flow(self):
        """Gradients should flow through the model."""
        x = torch.randn(4, 768, requires_grad=True)
        out = self.model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_detached_input(self):
        """Model should work with detached (no-grad) input."""
        x = torch.randn(4, 768).detach()
        out = self.model(x)
        assert out.shape == (4, 101)

    def test_different_bin_counts(self):
        """Model should work with different bin configurations."""
        for bins in [51, 101, 201]:
            model = QDAP_S(embedding_dim=768, num_bins=bins, conv_kernel=7)
            x = torch.randn(2, 768)
            out = model(x)
            assert out.shape == (2, bins)

    def test_different_kernel_sizes(self):
        """Model should work with different convolution kernels."""
        for kernel in [3, 5, 7, 11]:
            model = QDAP_S(embedding_dim=768, num_bins=101, conv_kernel=kernel)
            x = torch.randn(2, 768)
            out = model(x)
            assert out.shape == (2, 101)


class TestQDAP_L:
    @pytest.fixture(autouse=True)
    def setup(self):
        """Skip if model can't be loaded (requires internet)."""
        try:
            self.model = QDAP_L(
                model_name="Alibaba-NLP/gte-multilingual-base",
                num_bins=101,
                conv_kernel=7,
            )
            self.available = True
        except Exception:
            self.available = False
            pytest.skip("Model not available (requires download)")

    def test_forward_shape(self):
        if not self.available:
            pytest.skip()
        input_ids = torch.randint(0, 1000, (2, 32))
        attention_mask = torch.ones(2, 32, dtype=torch.long)
        out = self.model(input_ids, attention_mask)
        assert out.shape == (2, 101)

    def test_output_is_distribution(self):
        if not self.available:
            pytest.skip()
        input_ids = torch.randint(0, 1000, (2, 32))
        attention_mask = torch.ones(2, 32, dtype=torch.long)
        out = self.model(input_ids, attention_mask)
        sums = out.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_predict_alpha_range(self):
        if not self.available:
            pytest.skip()
        input_ids = torch.randint(0, 1000, (2, 32))
        attention_mask = torch.ones(2, 32, dtype=torch.long)
        alphas = self.model.predict_alpha(input_ids, attention_mask)
        assert (alphas >= 0).all()
        assert (alphas <= 1).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-k", "TestQDAP_S"])
