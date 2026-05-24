"""QDAP predictor training (Algorithm A1).

Trains the QDAP model to predict per-query alpha distributions by:
1. Computing nDCG@10 for each alpha in [0.00, 0.01, ..., 1.00]
2. Converting to a softmax-normalized target distribution
3. Training with composite loss: λ * L_CE + (1 - λ) * L_WD
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from typing import Dict, List, Optional, Union
from pathlib import Path
from tqdm import tqdm

from ..models.qdap_s import QDAP_S
from ..models.qdap_l import QDAP_L
from ..utils.score_normalization import minmax_normalize
from ..evaluation.metrics import compute_ndcg_at_k
from .losses import qdap_loss


def compute_alpha_target(
    dense_scores: np.ndarray,
    sparse_scores: np.ndarray,
    relevance_labels: np.ndarray,
    num_bins: int = 101,
    k: int = 10,
) -> np.ndarray:
    """Compute the target alpha distribution for a single query (Algorithm A1).

    For each alpha in [0.00, 0.01, ..., 1.00], compute nDCG@10 of the
    hybrid ranking. Return a softmax-normalized distribution over bins.

    Args:
        dense_scores: Dense retrieval scores for candidates.
        sparse_scores: Sparse (BM25) retrieval scores for candidates.
        relevance_labels: Relevance labels for candidates.
        num_bins: Number of alpha bins (default 101).
        k: Cutoff for nDCG computation.

    Returns:
        Target distribution of shape (num_bins,).
    """
    dense_norm = minmax_normalize(dense_scores)
    sparse_norm = minmax_normalize(sparse_scores)

    ndcg_values = np.zeros(num_bins)
    alphas = np.linspace(0, 1, num_bins)

    for i, alpha_i in enumerate(alphas):
        hybrid = alpha_i * dense_norm + (1.0 - alpha_i) * sparse_norm
        ranking = np.argsort(-hybrid)
        ranked_labels = relevance_labels[ranking]
        ndcg_values[i] = compute_ndcg_at_k(ranked_labels, k=k)

    ndcg_tensor = torch.tensor(ndcg_values, dtype=torch.float32)
    target_dist = F.softmax(ndcg_tensor, dim=0)
    return target_dist.numpy()


class QDAPDataset(Dataset):
    """Dataset for QDAP training with precomputed target distributions."""

    def __init__(
        self,
        queries: List[str],
        target_distributions: List[np.ndarray],
        tokenizer=None,
        query_embeddings: Optional[np.ndarray] = None,
        max_length: int = 128,
        qdap_type: str = "L",
    ):
        """
        Args:
            queries: List of query strings.
            target_distributions: List of target alpha distributions (num_bins,).
            tokenizer: Tokenizer for QDAP-L (not needed for QDAP-S).
            query_embeddings: Precomputed embeddings for QDAP-S.
            max_length: Max query token length.
            qdap_type: "S" or "L".
        """
        self.queries = queries
        self.targets = target_distributions
        self.tokenizer = tokenizer
        self.query_embeddings = query_embeddings
        self.max_length = max_length
        self.qdap_type = qdap_type

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        target = torch.tensor(self.targets[idx], dtype=torch.float32)

        if self.qdap_type == "S":
            embedding = torch.tensor(self.query_embeddings[idx], dtype=torch.float32)
            return {"embedding": embedding, "target": target}
        else:
            encoded = self.tokenizer(
                self.queries[idx], padding="max_length", truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )
            return {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "target": target,
            }


def precompute_targets(
    queries: List[str],
    dense_encoder,
    bm25_index,
    candidate_doc_ids: List[List[str]],
    relevance_labels: List[np.ndarray],
    num_bins: int = 101,
    device: Optional[torch.device] = None,
) -> List[np.ndarray]:
    """Precompute target alpha distributions for all training queries.

    Args:
        queries: List of query strings.
        dense_encoder: Trained dense encoder.
        bm25_index: BM25 index.
        candidate_doc_ids: Per-query list of candidate document IDs.
        relevance_labels: Per-query relevance labels for candidates.
        num_bins: Number of alpha bins.
        device: Device for dense encoder inference.

    Returns:
        List of target distributions, each of shape (num_bins,).
    """
    targets = []

    for i, query in enumerate(tqdm(queries, desc="Computing QDAP targets")):
        doc_ids = candidate_doc_ids[i]
        labels = relevance_labels[i]

        bm25_scores = np.array([
            bm25_index.score_single(query, doc_id) for doc_id in doc_ids
        ])

        q_emb = dense_encoder.encode_texts(query)
        doc_embs = dense_encoder.encode_texts(doc_ids)
        dense_scores = np.dot(q_emb, doc_embs.T).flatten()

        target = compute_alpha_target(dense_scores, bm25_scores, labels, num_bins=num_bins)
        targets.append(target)

    return targets


def train_qdap(
    qdap_model: Union[QDAP_S, QDAP_L],
    train_queries: List[str],
    train_targets: List[np.ndarray],
    val_queries: Optional[List[str]] = None,
    val_targets: Optional[List[np.ndarray]] = None,
    query_embeddings: Optional[np.ndarray] = None,
    qdap_type: str = "L",
    epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    warmup_ratio: float = 0.05,
    weight_decay: float = 0.01,
    lambda_weight: float = 0.62,
    gradient_accumulation_steps: int = 2,
    fp16: bool = True,
    output_dir: str = "./checkpoints/qdap",
    log_every: int = 50,
    device: Optional[torch.device] = None,
    use_wandb: bool = False,
):
    """Train the QDAP predictor.

    Args:
        qdap_model: QDAP-S or QDAP-L model instance.
        train_queries: Training query strings.
        train_targets: Precomputed target distributions.
        val_queries: Optional validation queries.
        val_targets: Optional validation targets.
        query_embeddings: Precomputed embeddings for QDAP-S training.
        qdap_type: "S" or "L".
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Peak learning rate.
        warmup_ratio: Warmup fraction.
        weight_decay: AdamW weight decay.
        lambda_weight: Weight for CE vs WD in composite loss (0.62 per paper).
        gradient_accumulation_steps: Gradient accumulation.
        fp16: Mixed precision training.
        output_dir: Checkpoint output directory.
        log_every: Logging frequency.
        device: Training device.
        use_wandb: Whether to log to W&B.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    qdap_model = qdap_model.to(device)
    qdap_model.train()

    tokenizer = getattr(qdap_model, "tokenizer", None)
    train_dataset = QDAPDataset(
        train_queries, train_targets,
        tokenizer=tokenizer,
        query_embeddings=query_embeddings,
        qdap_type=qdap_type,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )

    optimizer = AdamW(qdap_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs // gradient_accumulation_steps
    scheduler = OneCycleLR(
        optimizer, max_lr=learning_rate, total_steps=total_steps,
        pct_start=warmup_ratio, anneal_strategy="cos",
    )

    scaler = torch.amp.GradScaler("cuda") if fp16 and device.type == "cuda" else None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if use_wandb:
        import wandb
        wandb.init(project="query-adaptive-hybrid-search", name=f"qdap_{qdap_type}_training")

    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        progress = tqdm(train_loader, desc=f"QDAP Epoch {epoch + 1}/{epochs}")
        for batch_idx, batch in enumerate(progress):
            target = batch["target"].to(device)

            with torch.amp.autocast("cuda", enabled=fp16 and device.type == "cuda"):
                if qdap_type == "S":
                    embedding = batch["embedding"].to(device)
                    pred_dist = qdap_model(embedding)
                else:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    pred_dist = qdap_model(input_ids, attention_mask)

                loss = qdap_loss(pred_dist, target, lambda_weight=lambda_weight)
                loss = loss / gradient_accumulation_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += loss.item() * gradient_accumulation_steps
            num_batches += 1
            progress.set_postfix(loss=epoch_loss / num_batches)

            if global_step % log_every == 0 and global_step > 0 and use_wandb:
                import wandb
                wandb.log({
                    "qdap/train_loss": epoch_loss / num_batches,
                    "qdap/lr": scheduler.get_last_lr()[0],
                }, step=global_step)

        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        print(f"QDAP Epoch {epoch + 1} — Avg Loss: {avg_epoch_loss:.4f}")

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": qdap_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "qdap_type": qdap_type,
            "loss": avg_epoch_loss,
        }
        torch.save(checkpoint, output_path / f"checkpoint_epoch_{epoch + 1}.pt")

        if avg_epoch_loss < best_val_loss:
            best_val_loss = avg_epoch_loss
            torch.save(checkpoint, output_path / "best.pt")

    print(f"QDAP training complete. Best model saved to {output_path / 'best.pt'}")
