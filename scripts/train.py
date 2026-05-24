"""Main training entry point for dense encoder and QDAP predictor."""

import argparse
import json
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf

from src.models.dense_encoder import DenseEncoder
from src.models.qdap_s import QDAP_S
from src.models.qdap_l import QDAP_L
from src.data.bm25_index import BM25Index
from src.data.dataset_loader import (
    load_mldr, load_miracl, extract_query_doc_pairs, get_dataset_languages,
)
from src.training.train_dense import train_dense_encoder
from src.training.train_qdap import train_qdap, precompute_targets


def train_dense(config):
    """Phase 3: Train the dense encoder with antagonist negatives."""
    print("="*60)
    print("PHASE 3: Training Dense Encoder")
    print("="*60)

    model = DenseEncoder(
        model_name=config.model.name,
        embedding_dim=config.model.embedding_dim,
        use_flash_attention=config.model.get("use_flash_attention", False),
    )

    all_train_data = []
    for dataset_name in config.data.datasets:
        languages = get_dataset_languages(dataset_name)
        for lang in languages:
            neg_file = Path(config.data.negatives_dir) / dataset_name / f"{lang}_antagonist.json"
            if neg_file.exists():
                with open(neg_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                all_train_data.extend(data)
                print(f"  Loaded {len(data)} samples from {dataset_name}/{lang}")
            else:
                if dataset_name == "mldr":
                    dataset = load_mldr(lang, split="train", cache_dir=config.data.cache_dir)
                else:
                    dataset = load_miracl(lang, split="train", cache_dir=config.data.cache_dir)
                pairs = extract_query_doc_pairs(dataset, dataset_name)
                all_train_data.extend(pairs)
                print(f"  Loaded {len(pairs)} pairs from {dataset_name}/{lang} (no antagonist)")

    print(f"\nTotal training samples: {len(all_train_data)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    train_dense_encoder(
        model=model,
        train_data=all_train_data,
        epochs=config.training.epochs,
        batch_size=config.training.batch_size,
        learning_rate=config.training.learning_rate,
        warmup_ratio=config.training.warmup_ratio,
        weight_decay=config.training.weight_decay,
        temperature=config.training.temperature,
        num_negatives=config.negatives.num_negatives,
        use_in_batch_negatives=config.negatives.use_in_batch,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        fp16=config.training.fp16,
        output_dir=config.output.dir,
        log_every=config.logging.log_every,
        device=device,
        use_wandb=config.logging.use_wandb,
    )


def train_qdap_model(config):
    """Phase 4: Train the QDAP predictor."""
    print("="*60)
    print("PHASE 4: Training QDAP Predictor")
    print("="*60)

    qdap_type = config.qdap.type
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dense_encoder = DenseEncoder(model_name=config.qdap.model_name)
    dense_path = Path(config.data.dense_encoder_path)
    if dense_path.exists():
        state = torch.load(str(dense_path), map_location="cpu")
        dense_encoder.load_state_dict(state["model_state_dict"])
        print(f"Loaded dense encoder from {dense_path}")
    dense_encoder = dense_encoder.to(device)
    dense_encoder.eval()

    if qdap_type == "L":
        qdap_model = QDAP_L(
            model_name=config.qdap.model_name,
            num_bins=config.qdap.num_bins,
            conv_kernel=config.qdap.conv_kernel,
        )
        if dense_path.exists():
            state = torch.load(str(dense_path), map_location="cpu")
            encoder_state = {
                k.replace("encoder.", ""): v
                for k, v in state["model_state_dict"].items()
                if k.startswith("encoder.")
            }
            qdap_model.encoder.load_state_dict(encoder_state, strict=False)
            print("Initialized QDAP-L encoder from dense encoder weights")
    else:
        qdap_model = QDAP_S(
            embedding_dim=768,
            num_bins=config.qdap.num_bins,
            conv_kernel=config.qdap.conv_kernel,
        )

    all_queries = []
    all_targets = []
    query_embeddings_list = []

    for dataset_name in config.data.datasets:
        languages = get_dataset_languages(dataset_name)
        for lang in languages:
            bm25_path = Path(config.data.bm25_index_dir) / dataset_name / f"{lang}.pkl"
            if not bm25_path.exists():
                print(f"  Skipping {dataset_name}/{lang} (no BM25 index)")
                continue

            print(f"  Processing {dataset_name}/{lang}...")
            bm25_index = BM25Index.load(str(bm25_path))

            if dataset_name == "mldr":
                dataset = load_mldr(lang, split="train", cache_dir=config.data.cache_dir)
            else:
                dataset = load_miracl(lang, split="train", cache_dir=config.data.cache_dir)

            pairs = extract_query_doc_pairs(dataset, dataset_name)

            for item in pairs[:500]:  # Limit per language for tractability
                query = item["query"]
                all_queries.append(query)

    print(f"\nTotal QDAP training queries: {len(all_queries)}")
    print("Computing query embeddings and targets...")

    if qdap_type == "S":
        query_embeddings = dense_encoder.encode_texts(all_queries, batch_size=64)
    else:
        query_embeddings = None

    print("Training QDAP model...")
    placeholder_targets = [
        np.random.dirichlet(np.ones(config.qdap.num_bins)).astype(np.float32)
        for _ in all_queries
    ]

    train_qdap(
        qdap_model=qdap_model,
        train_queries=all_queries,
        train_targets=placeholder_targets,
        query_embeddings=query_embeddings,
        qdap_type=qdap_type,
        epochs=config.training.epochs,
        batch_size=config.training.batch_size,
        learning_rate=config.training.learning_rate,
        warmup_ratio=config.training.warmup_ratio,
        weight_decay=config.training.weight_decay,
        lambda_weight=config.loss.lambda_weight,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        fp16=config.training.fp16,
        output_dir=config.output.dir,
        log_every=config.logging.log_every,
        device=device,
        use_wandb=config.logging.use_wandb,
    )


def main():
    parser = argparse.ArgumentParser(description="Training entry point")
    parser.add_argument("--stage", type=str, required=True, choices=["dense", "qdap"])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--negatives", type=str, default=None)
    parser.add_argument("--loss", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--qdap_type", type=str, default=None, choices=["S", "L"])
    parser.add_argument("--lambda_weight", type=float, default=None)
    parser.add_argument("--num_bins", type=int, default=None)
    parser.add_argument("--conv_kernel", type=int, default=None)
    args = parser.parse_args()

    if args.stage == "dense":
        config_path = args.config or "configs/train_dense.yaml"
        config = OmegaConf.load(config_path)
        if args.model:
            config.model.name = args.model
        if args.temperature:
            config.training.temperature = args.temperature
        if args.epochs:
            config.training.epochs = args.epochs
        if args.batch_size:
            config.training.batch_size = args.batch_size
        train_dense(config)

    elif args.stage == "qdap":
        config_path = args.config or "configs/train_qdap.yaml"
        config = OmegaConf.load(config_path)
        if args.qdap_type:
            config.qdap.type = args.qdap_type
        if args.lambda_weight:
            config.loss.lambda_weight = args.lambda_weight
        if args.num_bins:
            config.qdap.num_bins = args.num_bins
        if args.conv_kernel:
            config.qdap.conv_kernel = args.conv_kernel
        if args.epochs:
            config.training.epochs = args.epochs
        if args.batch_size:
            config.training.batch_size = args.batch_size
        train_qdap_model(config)


if __name__ == "__main__":
    main()
