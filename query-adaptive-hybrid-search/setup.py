from setuptools import setup, find_packages

setup(
    name="query-adaptive-hybrid-search",
    version="0.1.0",
    description="Query-Adaptive Hybrid Search (Posokhov et al., 2026)",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "transformers>=4.36",
        "datasets>=2.14",
        "faiss-cpu>=1.7.4",
        "rank_bm25>=0.2.2",
        "numpy>=1.24",
        "scipy>=1.10",
        "scikit-learn>=1.3",
        "tqdm>=4.65",
        "hydra-core>=1.3",
        "omegaconf>=2.3",
    ],
    extras_require={
        "gpu": ["faiss-gpu>=1.7.4"],
        "logging": ["wandb"],
    },
)
