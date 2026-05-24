"""Pre-build BM25 indices per language for both MLDR and MIRACL."""

import argparse
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.bm25_index import BM25Index
from src.data.dataset_loader import (
    load_mldr_corpus, load_miracl_corpus,
    get_dataset_languages,
)

BM25_PARAMS = {
    "miracl": {"k1": 0.9, "b": 0.4},
    "mldr": {"k1": 1.2, "b": 0.75},
}


def build_index_for_language(dataset_name: str, language: str, output_dir: str, cache_dir: str):
    """Build and save a BM25 index for a specific dataset and language."""
    params = BM25_PARAMS[dataset_name]
    index = BM25Index(
        language=language,
        k1=params["k1"],
        b=params["b"],
        dataset_type=dataset_name,
    )

    print(f"  Loading corpus for {dataset_name}/{language}...")
    if dataset_name == "mldr":
        corpus = load_mldr_corpus(language, cache_dir=cache_dir)
    else:
        corpus = load_miracl_corpus(language, cache_dir=cache_dir)

    documents = []
    for item in tqdm(corpus, desc=f"  Preparing docs ({language})", leave=False):
        doc_id = item.get("docid", item.get("_id", str(len(documents))))
        text = item.get("text", item.get("content", ""))
        title = item.get("title", "")
        full_text = f"{title} {text}".strip() if title else text
        documents.append((doc_id, full_text))

    print(f"  Building index ({len(documents)} documents)...")
    index.build_index(documents)

    output_path = Path(output_dir) / dataset_name / f"{language}.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    index.save(str(output_path))
    print(f"  Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Build BM25 indices")
    parser.add_argument("--datasets", nargs="+", default=["mldr", "miracl"],
                       choices=["mldr", "miracl"])
    parser.add_argument("--languages", nargs="+", default=["all"])
    parser.add_argument("--output_dir", type=str, default="./data/bm25_indices")
    parser.add_argument("--cache_dir", type=str, default="./data/cache")
    args = parser.parse_args()

    for dataset_name in args.datasets:
        languages = get_dataset_languages(dataset_name)
        if "all" not in args.languages:
            languages = [l for l in languages if l in args.languages]

        print(f"\n{'='*60}")
        print(f"Building BM25 indices for {dataset_name.upper()}")
        print(f"Parameters: k1={BM25_PARAMS[dataset_name]['k1']}, b={BM25_PARAMS[dataset_name]['b']}")
        print(f"{'='*60}")

        for lang in languages:
            print(f"\n[{lang}]")
            try:
                build_index_for_language(dataset_name, lang, args.output_dir, args.cache_dir)
            except Exception as e:
                print(f"  ERROR: {e}")

    print("\nAll BM25 indices built!")


if __name__ == "__main__":
    main()
