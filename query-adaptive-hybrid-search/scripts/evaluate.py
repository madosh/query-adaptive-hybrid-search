"""Main evaluation entry point."""

import argparse
import json
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf

from src.models.dense_encoder import DenseEncoder
from src.models.qdap_s import QDAP_S
from src.models.qdap_l import QDAP_L
from src.data.bm25_index import BM25Index
from src.data.dataset_loader import (
    load_mldr, load_miracl, load_mldr_corpus, load_miracl_corpus,
    extract_query_doc_pairs, get_dataset_languages,
)
from src.utils.score_normalization import minmax_normalize
from src.evaluation.evaluate import full_evaluation
from src.evaluation.metrics import compute_ndcg_at_k


def evaluate_dataset(
    dataset_name: str,
    language: str,
    dense_encoder: DenseEncoder,
    qdap_model,
    bm25_index: BM25Index,
    config,
    device: torch.device,
) -> dict:
    """Evaluate on a single dataset/language combination."""
    if dataset_name == "mldr":
        dataset = load_mldr(language, split="test", cache_dir="./data/cache")
    else:
        dataset = load_miracl(language, split="dev", cache_dir="./data/cache")

    pairs = extract_query_doc_pairs(dataset, dataset_name)
    if not pairs:
        return {}

    all_dense_scores = []
    all_sparse_scores = []
    all_relevance_labels = []
    predicted_alphas = []

    pool_size = config.fusion.pool_size

    for item in tqdm(pairs, desc=f"  Eval {dataset_name}/{language}", leave=False):
        query = item["query"]

        bm25_results = bm25_index.search(query, top_k=pool_size)
        if not bm25_results:
            continue

        bm25_doc_ids = [doc_id for doc_id, _ in bm25_results]
        bm25_scores = np.array([score for _, score in bm25_results])

        q_emb = dense_encoder.encode_texts(query)

        positive_doc_ids = set()
        for doc in item["positive_docs"]:
            if isinstance(doc, dict):
                positive_doc_ids.add(doc.get("docid", doc.get("id", "")))
            else:
                positive_doc_ids.add(str(doc))

        relevance = np.array([
            1.0 if doc_id in positive_doc_ids else 0.0
            for doc_id in bm25_doc_ids
        ])

        dense_scores_dict = bm25_index.batch_score(query, bm25_doc_ids)
        dense_scores = np.array([dense_scores_dict.get(d, 0.0) for d in bm25_doc_ids])

        all_dense_scores.append(dense_scores)
        all_sparse_scores.append(bm25_scores)
        all_relevance_labels.append(relevance)

        q_emb_tensor = torch.tensor(q_emb, device=device)
        if isinstance(qdap_model, QDAP_L):
            alpha = qdap_model.predict_alpha_from_text(query, device=device).item()
        else:
            alpha = qdap_model.predict_alpha(q_emb_tensor).item()
        predicted_alphas.append(alpha)

    if not all_dense_scores:
        return {}

    results = full_evaluation(
        dense_scores=all_dense_scores,
        sparse_scores=all_sparse_scores,
        relevance_labels=all_relevance_labels,
        predicted_alphas=predicted_alphas,
        k=10,
        num_bins=101,
    )

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluation entry point")
    parser.add_argument("--config", type=str, default="configs/eval.yaml")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--languages", nargs="+", default=None)
    parser.add_argument("--metrics", nargs="+", default=None)
    parser.add_argument("--compute_oracle", type=bool, default=None)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    if args.datasets:
        config.evaluation.datasets = args.datasets

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}")

    print("Loading dense encoder...")
    dense_encoder = DenseEncoder()
    dense_path = Path(config.models.dense_encoder_path)
    if dense_path.exists():
        state = torch.load(str(dense_path), map_location="cpu")
        dense_encoder.load_state_dict(state["model_state_dict"])
    dense_encoder = dense_encoder.to(device)
    dense_encoder.eval()

    print("Loading QDAP model...")
    qdap_type = config.models.qdap_type
    if qdap_type == "L":
        qdap_model = QDAP_L()
    else:
        qdap_model = QDAP_S()
    qdap_path = Path(config.models.qdap_path)
    if qdap_path.exists():
        state = torch.load(str(qdap_path), map_location="cpu")
        qdap_model.load_state_dict(state["model_state_dict"])
    qdap_model = qdap_model.to(device)
    qdap_model.eval()

    all_results = {}

    for dataset_name in config.evaluation.datasets:
        languages = get_dataset_languages(dataset_name)
        if args.languages and "all" not in args.languages:
            languages = [l for l in languages if l in args.languages]

        print(f"\n{'='*60}")
        print(f"Evaluating {dataset_name.upper()}")
        print(f"{'='*60}")

        dataset_results = {}
        for lang in languages:
            bm25_path = Path(config.bm25.index_dir) / dataset_name / f"{lang}.pkl"
            if not bm25_path.exists():
                print(f"  [{lang}] BM25 index not found, skipping")
                continue

            print(f"  [{lang}] Evaluating...")
            bm25_index = BM25Index.load(str(bm25_path))

            results = evaluate_dataset(
                dataset_name, lang, dense_encoder, qdap_model,
                bm25_index, config, device,
            )

            if results:
                dataset_results[lang] = results
                print(f"  [{lang}] nDCG@10: {results.get('qdap_ndcg@10', 0):.4f}")

        if dataset_results:
            avg_ndcg = np.mean([r["qdap_ndcg@10"] for r in dataset_results.values()])
            print(f"\n  {dataset_name.upper()} Average nDCG@10: {avg_ndcg:.4f}")
            all_results[dataset_name] = dataset_results

    output_dir = Path(config.output.dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_dir / 'evaluation_results.json'}")


if __name__ == "__main__":
    main()
