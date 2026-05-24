"""Download MLDR and MIRACL datasets from HuggingFace."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset_loader import (
    load_mldr, load_mldr_corpus,
    load_miracl, load_miracl_corpus,
    get_dataset_languages,
)


def main():
    parser = argparse.ArgumentParser(description="Download MLDR and MIRACL datasets")
    parser.add_argument("--datasets", nargs="+", default=["mldr", "miracl"],
                       choices=["mldr", "miracl"])
    parser.add_argument("--languages", nargs="+", default=["all"],
                       help="Language codes or 'all'")
    parser.add_argument("--cache_dir", type=str, default="./data/cache")
    args = parser.parse_args()

    cache_dir = args.cache_dir
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    for dataset_name in args.datasets:
        languages = get_dataset_languages(dataset_name)
        if "all" not in args.languages:
            languages = [l for l in languages if l in args.languages]

        print(f"\n{'='*60}")
        print(f"Downloading {dataset_name.upper()} ({len(languages)} languages)")
        print(f"{'='*60}")

        for lang in languages:
            print(f"\n  [{lang}] Downloading...")
            try:
                if dataset_name == "mldr":
                    load_mldr(lang, split="train", cache_dir=cache_dir)
                    load_mldr(lang, split="test", cache_dir=cache_dir)
                    load_mldr_corpus(lang, cache_dir=cache_dir)
                    print(f"  [{lang}] Done (train + test + corpus)")
                elif dataset_name == "miracl":
                    load_miracl(lang, split="train", cache_dir=cache_dir)
                    load_miracl(lang, split="dev", cache_dir=cache_dir)
                    load_miracl_corpus(lang, cache_dir=cache_dir)
                    print(f"  [{lang}] Done (train + dev + corpus)")
            except Exception as e:
                print(f"  [{lang}] ERROR: {e}")

    print("\nDownload complete!")


if __name__ == "__main__":
    main()
