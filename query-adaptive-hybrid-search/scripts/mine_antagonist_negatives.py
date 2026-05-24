"""Run antagonist negative mining (Algorithm A2).

Pre-computes hard negatives where both BM25 and dense retriever fail,
caching results for dense encoder training.
"""

import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.dense_encoder import DenseEncoder
from src.data.bm25_index import BM25Index
from src.data.antagonist_sampler import AntagonistSampler
from src.data.dataset_loader import (
    load_mldr, load_miracl,
    extract_query_doc_pairs, get_dataset_languages,
)


def main():
    parser = argparse.ArgumentParser(description="Mine antagonist negatives")
    parser.add_argument("--base_model", type=str, default="Alibaba-NLP/gte-multilingual-base")
    parser.add_argument("--threshold_sigma", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--max_negatives", type=int, default=15)
    parser.add_argument("--datasets", nargs="+", default=["mldr", "miracl"])
    parser.add_argument("--languages", nargs="+", default=["all"])
    parser.add_argument("--bm25_index_dir", type=str, default="./data/bm25_indices")
    parser.add_argument("--output_dir", type=str, default="./data/negatives")
    parser.add_argument("--cache_dir", type=str, default="./data/cache")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    print("Loading dense encoder...")
    dense_encoder = DenseEncoder(model_name=args.base_model)
    dense_encoder.eval()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for dataset_name in args.datasets:
        languages = get_dataset_languages(dataset_name)
        if "all" not in args.languages:
            languages = [l for l in languages if l in args.languages]

        print(f"\n{'='*60}")
        print(f"Mining antagonist negatives for {dataset_name.upper()}")
        print(f"Threshold σ = {args.threshold_sigma}")
        print(f"{'='*60}")

        for lang in languages:
            print(f"\n[{lang}] Loading data...")
            bm25_path = Path(args.bm25_index_dir) / dataset_name / f"{lang}.pkl"
            if not bm25_path.exists():
                print(f"  BM25 index not found at {bm25_path}, skipping.")
                continue

            bm25_index = BM25Index.load(str(bm25_path))

            if dataset_name == "mldr":
                dataset = load_mldr(lang, split="train", cache_dir=args.cache_dir)
            else:
                dataset = load_miracl(lang, split="train", cache_dir=args.cache_dir)

            pairs = extract_query_doc_pairs(dataset, dataset_name)
            print(f"  {len(pairs)} query-document pairs loaded")

            sampler = AntagonistSampler(
                bm25_index=bm25_index,
                dense_encoder=dense_encoder,
                threshold_sigma=args.threshold_sigma,
                top_k=args.top_k,
            )

            results = sampler.mine_negatives_for_dataset(
                pairs, max_negatives_per_query=args.max_negatives,
            )

            out_file = output_path / dataset_name / f"{lang}_antagonist.json"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            print(f"  Mined {len(results)} queries with antagonist negatives")
            print(f"  Saved to {out_file}")

    print("\nAntagonist negative mining complete!")


if __name__ == "__main__":
    main()
