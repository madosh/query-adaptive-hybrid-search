"""Dense encoder training with InfoNCE loss and antagonist negatives."""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from typing import Dict, List, Optional
from pathlib import Path
from tqdm import tqdm

from ..models.dense_encoder import DenseEncoder
from .losses import infonce_loss, infonce_loss_with_in_batch


class AntagonistDataset(Dataset):
    """Dataset for dense encoder training with antagonist negatives.

    Each item contains a query, a positive document, and pre-mined
    antagonist negative documents.
    """

    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_query_length: int = 128,
        max_doc_length: int = 512,
        num_negatives: int = 7,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length
        self.num_negatives = num_negatives

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        query = item["query"]

        pos_docs = item["positive_docs"]
        if isinstance(pos_docs[0], dict):
            pos_text = pos_docs[0].get("text", pos_docs[0].get("content", ""))
        else:
            pos_text = str(pos_docs[0])

        neg_texts = []
        antagonist_negs = item.get("antagonist_negatives", [])
        regular_negs = item.get("negative_docs", [])

        for neg in antagonist_negs[:self.num_negatives]:
            if isinstance(neg, dict):
                neg_texts.append(neg.get("text", neg.get("content", "")))
            else:
                neg_texts.append(str(neg))

        remaining = self.num_negatives - len(neg_texts)
        if remaining > 0:
            for neg in regular_negs[:remaining]:
                if isinstance(neg, dict):
                    neg_texts.append(neg.get("text", neg.get("content", "")))
                else:
                    neg_texts.append(str(neg))

        while len(neg_texts) < self.num_negatives:
            neg_texts.append("")

        query_enc = self.tokenizer(
            query, padding="max_length", truncation=True,
            max_length=self.max_query_length, return_tensors="pt",
        )
        pos_enc = self.tokenizer(
            pos_text, padding="max_length", truncation=True,
            max_length=self.max_doc_length, return_tensors="pt",
        )
        neg_encs = self.tokenizer(
            neg_texts, padding="max_length", truncation=True,
            max_length=self.max_doc_length, return_tensors="pt",
        )

        return {
            "query_input_ids": query_enc["input_ids"].squeeze(0),
            "query_attention_mask": query_enc["attention_mask"].squeeze(0),
            "pos_input_ids": pos_enc["input_ids"].squeeze(0),
            "pos_attention_mask": pos_enc["attention_mask"].squeeze(0),
            "neg_input_ids": neg_encs["input_ids"],
            "neg_attention_mask": neg_encs["attention_mask"],
        }


def train_dense_encoder(
    model: DenseEncoder,
    train_data: List[Dict],
    val_data: Optional[List[Dict]] = None,
    epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    temperature: float = 0.05,
    num_negatives: int = 7,
    use_in_batch_negatives: bool = True,
    gradient_accumulation_steps: int = 4,
    fp16: bool = True,
    output_dir: str = "./checkpoints/dense_encoder",
    log_every: int = 100,
    device: Optional[torch.device] = None,
    use_wandb: bool = False,
):
    """Train the dense encoder with InfoNCE loss and antagonist negatives.

    Args:
        model: DenseEncoder instance.
        train_data: Training data with antagonist negatives.
        val_data: Optional validation data.
        epochs: Number of training epochs (default 3 per paper).
        batch_size: Training batch size.
        learning_rate: Peak learning rate.
        warmup_ratio: Fraction of steps for warmup.
        weight_decay: AdamW weight decay.
        temperature: InfoNCE temperature τ.
        num_negatives: Number of negatives per query.
        use_in_batch_negatives: Whether to use in-batch negatives.
        gradient_accumulation_steps: Gradient accumulation.
        fp16: Whether to use mixed precision.
        output_dir: Directory to save checkpoints.
        log_every: Log every N steps.
        device: Training device.
        use_wandb: Whether to log to W&B.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.train()

    dataset = AntagonistDataset(
        train_data, model.tokenizer,
        num_negatives=num_negatives,
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )

    optimizer = AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay,
    )
    total_steps = len(dataloader) * epochs // gradient_accumulation_steps
    scheduler = OneCycleLR(
        optimizer, max_lr=learning_rate, total_steps=total_steps,
        pct_start=warmup_ratio, anneal_strategy="cos",
    )

    scaler = torch.amp.GradScaler("cuda") if fp16 and device.type == "cuda" else None
    loss_fn = infonce_loss_with_in_batch if use_in_batch_negatives else infonce_loss

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if use_wandb:
        import wandb
        wandb.init(project="query-adaptive-hybrid-search", name="dense_encoder_training")

    global_step = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch_idx, batch in enumerate(progress):
            query_ids = batch["query_input_ids"].to(device)
            query_mask = batch["query_attention_mask"].to(device)
            pos_ids = batch["pos_input_ids"].to(device)
            pos_mask = batch["pos_attention_mask"].to(device)
            neg_ids = batch["neg_input_ids"].to(device)
            neg_mask = batch["neg_attention_mask"].to(device)

            with torch.amp.autocast("cuda", enabled=fp16 and device.type == "cuda"):
                q_emb = model.encode(query_ids, query_mask)
                p_emb = model.encode(pos_ids, pos_mask)

                bsz, n_neg, seq_len = neg_ids.shape
                neg_ids_flat = neg_ids.view(bsz * n_neg, seq_len)
                neg_mask_flat = neg_mask.view(bsz * n_neg, seq_len)
                n_emb_flat = model.encode(neg_ids_flat, neg_mask_flat)
                n_emb = n_emb_flat.view(bsz, n_neg, -1)

                loss = loss_fn(q_emb, p_emb, n_emb, temperature=temperature)
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

            if global_step % log_every == 0 and global_step > 0:
                avg_loss = epoch_loss / num_batches
                if use_wandb:
                    import wandb
                    wandb.log({"train/loss": avg_loss, "train/lr": scheduler.get_last_lr()[0]}, step=global_step)

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_name": model.tokenizer.name_or_path,
            "loss": epoch_loss / max(num_batches, 1),
        }
        torch.save(checkpoint, output_path / f"checkpoint_epoch_{epoch + 1}.pt")
        print(f"Epoch {epoch + 1} — Avg Loss: {epoch_loss / max(num_batches, 1):.4f}")

    torch.save(checkpoint, output_path / "best.pt")
    print(f"Training complete. Best model saved to {output_path / 'best.pt'}")
