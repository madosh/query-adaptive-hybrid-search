"""Loss functions for dense encoder and QDAP training.

Implements:
- InfoNCE loss for dense encoder training with antagonist negatives (Eq. 15)
- Squared Cross-Entropy for QDAP target matching (Eq. 10)
- 1D Wasserstein Distance for distribution matching (Eq. 11)
- Composite QDAP loss: λ * L_CE + (1 - λ) * L_WD (Eq. 9)
"""

import torch
import torch.nn.functional as F


def infonce_loss(
    query_emb: torch.Tensor,
    pos_emb: torch.Tensor,
    neg_embs: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    """InfoNCE contrastive loss (Equation 15).

    Args:
        query_emb: Query embeddings of shape (batch, dim).
        pos_emb: Positive document embeddings of shape (batch, dim).
        neg_embs: Negative document embeddings of shape (batch, num_neg, dim).
        temperature: Softmax temperature τ (default 0.05).

    Returns:
        Scalar loss value.
    """
    pos_score = torch.sum(query_emb * pos_emb, dim=-1, keepdim=True) / temperature
    neg_scores = torch.bmm(neg_embs, query_emb.unsqueeze(-1)).squeeze(-1) / temperature
    logits = torch.cat([pos_score, neg_scores], dim=-1)
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


def infonce_loss_with_in_batch(
    query_emb: torch.Tensor,
    pos_emb: torch.Tensor,
    neg_embs: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    """InfoNCE with in-batch negatives combined with explicit antagonist negatives.

    Uses all other positives in the batch as additional negatives.

    Args:
        query_emb: Query embeddings of shape (batch, dim).
        pos_emb: Positive document embeddings of shape (batch, dim).
        neg_embs: Explicit negative embeddings of shape (batch, num_neg, dim).
        temperature: Softmax temperature τ.

    Returns:
        Scalar loss value.
    """
    batch_size = query_emb.size(0)

    pos_score = torch.sum(query_emb * pos_emb, dim=-1, keepdim=True) / temperature
    explicit_neg_scores = torch.bmm(neg_embs, query_emb.unsqueeze(-1)).squeeze(-1) / temperature

    in_batch_scores = torch.mm(query_emb, pos_emb.t()) / temperature
    mask = torch.eye(batch_size, device=query_emb.device).bool()
    in_batch_neg_scores = in_batch_scores.masked_fill(mask, float("-inf"))

    logits = torch.cat([pos_score, explicit_neg_scores, in_batch_neg_scores], dim=-1)
    labels = torch.zeros(batch_size, dtype=torch.long, device=query_emb.device)
    return F.cross_entropy(logits, labels)


def squared_cross_entropy(
    y_pred: torch.Tensor,
    y_target: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy with squared targets (Equation 10).

    Squares the target distribution to emphasize high-probability bins,
    then re-normalizes before computing cross-entropy.

    Args:
        y_pred: Predicted distribution of shape (batch, num_bins), after softmax.
        y_target: Target nDCG distribution of shape (batch, num_bins), after softmax.

    Returns:
        Scalar loss value.
    """
    y_target_sq = y_target ** 2
    y_target_sq = y_target_sq / y_target_sq.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return -(y_target_sq * torch.log(y_pred + 1e-8)).sum(dim=-1).mean()


def wasserstein_1d(
    y_pred: torch.Tensor,
    y_target: torch.Tensor,
) -> torch.Tensor:
    """1D Wasserstein distance via CDF difference (Equation 11).

    Closed-form computation for 1D distributions using the difference
    of cumulative distribution functions.

    Args:
        y_pred: Predicted distribution of shape (batch, num_bins).
        y_target: Target distribution of shape (batch, num_bins).

    Returns:
        Scalar loss value.
    """
    cdf_pred = torch.cumsum(y_pred, dim=-1)
    cdf_target = torch.cumsum(y_target, dim=-1)
    return torch.abs(cdf_target - cdf_pred).sum(dim=-1).mean()


def qdap_loss(
    y_pred: torch.Tensor,
    y_target: torch.Tensor,
    lambda_weight: float = 0.62,
) -> torch.Tensor:
    """Composite QDAP loss (Equation 9): L = λ * L_CE + (1 - λ) * L_WD.

    Args:
        y_pred: Predicted distribution of shape (batch, num_bins).
        y_target: Target distribution of shape (batch, num_bins).
        lambda_weight: Weight balancing CE and WD (default 0.62 per paper).

    Returns:
        Scalar loss value.
    """
    l_ce = squared_cross_entropy(y_pred, y_target)
    l_wd = wasserstein_1d(y_pred, y_target)
    return lambda_weight * l_ce + (1.0 - lambda_weight) * l_wd
