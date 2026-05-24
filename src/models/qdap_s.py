"""QDAP-S: Lightweight adapter-based alpha predictor.

Operates on frozen dense encoder embeddings (no backprop through encoder).
Uses a linear layer + 1D convolution for smoothed distribution prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QDAP_S(nn.Module):
    """Query-Driven Alpha Prediction — Small variant.

    A minimal predictor that takes a frozen query embedding and produces
    a probability distribution over 101 alpha bins [0.00, 0.01, ..., 1.00].
    """

    def __init__(self, embedding_dim: int = 768, num_bins: int = 101, conv_kernel: int = 7):
        super().__init__()
        self.num_bins = num_bins
        self.linear = nn.Linear(embedding_dim, num_bins)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=conv_kernel, padding=conv_kernel // 2)

    def forward(self, query_embedding: torch.Tensor) -> torch.Tensor:
        """Predict alpha distribution from a frozen query embedding.

        Args:
            query_embedding: Detached embedding of shape (batch, embedding_dim).
                Must NOT have gradient flow back to the dense encoder.

        Returns:
            Probability distribution over alpha bins, shape (batch, num_bins).
        """
        z = self.linear(query_embedding)
        z = z.unsqueeze(1)
        z_smooth = self.conv1d(z).squeeze(1)
        return F.softmax(z_smooth, dim=-1)

    def predict_alpha(self, query_embedding: torch.Tensor) -> torch.Tensor:
        """Predict the scalar alpha value as the expected value of the distribution.

        Args:
            query_embedding: Detached embedding of shape (batch, embedding_dim).

        Returns:
            Predicted alpha values, shape (batch,).
        """
        dist = self.forward(query_embedding)
        alpha_values = torch.linspace(0, 1, self.num_bins, device=dist.device)
        return (dist * alpha_values).sum(dim=-1)

    def predict_alpha_argmax(self, query_embedding: torch.Tensor) -> torch.Tensor:
        """Predict alpha using argmax (discrete mode prediction).

        Args:
            query_embedding: Detached embedding of shape (batch, embedding_dim).

        Returns:
            Predicted alpha values, shape (batch,).
        """
        dist = self.forward(query_embedding)
        indices = dist.argmax(dim=-1)
        return indices.float() / (self.num_bins - 1)
