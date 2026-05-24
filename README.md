<div align="center">

# Query-Adaptive Hybrid Search

**Dynamically balancing sparse and dense retrieval on a per-query basis**

[![Paper](https://img.shields.io/badge/Paper-DOI%2010.3390%2Fmake8040091-blue)](https://doi.org/10.3390/make8040091)
[![Venue](https://img.shields.io/badge/Venue-MAKE%202026-green)](https://doi.org/10.3390/make8040091)
[![Python](https://img.shields.io/badge/Python-3.9%2B-yellow)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-Research-lightgrey)](#license)

<br>

**Topics**

[![information-retrieval](https://img.shields.io/badge/topic-information--retrieval-2563eb?style=flat-square&logo=github&logoColor=white)](https://github.com/topics/information-retrieval)
[![hybrid-search](https://img.shields.io/badge/topic-hybrid--search-2563eb?style=flat-square&logo=github&logoColor=white)](https://github.com/topics/hybrid-search)
[![semantic-search](https://img.shields.io/badge/topic-semantic--search-2563eb?style=flat-square&logo=github&logoColor=white)](https://github.com/topics/semantic-search)
[![bm25](https://img.shields.io/badge/topic-bm25-2563eb?style=flat-square&logo=github&logoColor=white)](https://github.com/topics/bm25)
[![dense-retrieval](https://img.shields.io/badge/topic-dense--retrieval-2563eb?style=flat-square&logo=github&logoColor=white)](https://github.com/topics/dense-retrieval)
[![multilingual-nlp](https://img.shields.io/badge/topic-multilingual--nlp-2563eb?style=flat-square&logo=github&logoColor=white)](https://github.com/topics/multilingual-nlp)
[![transformers](https://img.shields.io/badge/topic-transformers-2563eb?style=flat-square&logo=github&logoColor=white)](https://github.com/topics/transformers)
[![contrastive-learning](https://img.shields.io/badge/topic-contrastive--learning-2563eb?style=flat-square&logo=github&logoColor=white)](https://github.com/topics/contrastive-learning)
[![miracl](https://img.shields.io/badge/topic-miracl-2563eb?style=flat-square&logo=github&logoColor=white)](https://github.com/topics/miracl)
[![mteb](https://img.shields.io/badge/topic-mteb-2563eb?style=flat-square&logo=github&logoColor=white)](https://github.com/topics/mteb)
[![rag](https://img.shields.io/badge/topic-retrieval--augmented--generation-2563eb?style=flat-square&logo=github&logoColor=white)](https://github.com/topics/retrieval-augmented-generation)
[![research-code](https://img.shields.io/badge/topic-research--code-2563eb?style=flat-square&logo=github&logoColor=white)](https://github.com/topics/research-code)

<br>

*Implementation of "Query-Adaptive Hybrid Search" (Posokhov et al., 2026)*
*Machine Learning and Knowledge Extraction*

</div>

---

## GitHub About (for discoverability)

Paste this into your repo **About** sidebar (Settings → General, or the ⚙️ on the repo homepage):

| Field | Value |
|:---|:---|
| **Description** | Adaptive hybrid retrieval: per-query BM25 + dense fusion via QDAP. PyTorch implementation with antagonist negative sampling on MLDR & MIRACL (29 languages). |
| **Website** | [Paper (DOI)](https://doi.org/10.3390/make8040091) |
| **Topics** | See [`.github/TOPICS.txt`](.github/TOPICS.txt) |

**One-command setup** (requires [GitHub CLI](https://cli.github.com/) after `gh auth login`):

```powershell
# Windows
.\.github\setup-github-about.ps1
```

```bash
# macOS / Linux
bash .github/setup-github-about.sh
```

<details>
<summary><b>All 20 GitHub topics (click to copy)</b></summary>

```
information-retrieval, hybrid-search, semantic-search, bm25, dense-retrieval,
neural-information-retrieval, multilingual-nlp, pytorch, transformers,
contrastive-learning, question-answering, natural-language-processing,
machine-learning, deep-learning, miracl, mteb, retrieval-augmented-generation,
search-engine, research-code, make-2026
```

</details>

---

## Overview

Traditional hybrid search uses a **fixed mixing weight** to combine BM25 and dense retrieval scores. This fails because different queries benefit from different balances — keyword-heavy queries favor BM25, while semantic queries favor neural retrieval.

**This work introduces two key innovations:**

| Contribution | Description |
|:---|:---|
| **QDAP** (Query-Driven Alpha Prediction) | A neural module that predicts the optimal mixing coefficient α from the query alone |
| **Antagonist Negative Sampling** | A training strategy mining hard negatives where *both* retrievers fail, forcing complementary learning |

---

## Architecture

### System Overview

```mermaid
graph TB
    subgraph Input
        Q[/"Query: q"/]
    end

    subgraph Dense["Dense Retrieval Path"]
        DE[Dense Encoder<br/><i>mGTE ~305M params</i>]
        FI[(FAISS Index)]
        DR[Top-100 Dense Results]
    end

    subgraph Sparse["Sparse Retrieval Path"]
        BM[BM25 Index<br/><i>Language-specific tokenization</i>]
        SR[Top-100 Sparse Results]
    end

    subgraph QDAP["Alpha Prediction"]
        QE[Query Embedding]
        AP[QDAP Predictor<br/><i>101-bin distribution</i>]
        AL((α))
    end

    subgraph Fusion["Score Fusion"]
        UN[Union Candidates]
        MN[Min-Max Normalize]
        HS["s_hybrid = α·s_dense + (1-α)·s_sparse"]
        RK[Re-rank → Top-K]
    end

    Q --> DE
    Q --> BM
    DE --> FI --> DR
    BM --> SR
    
    Q --> QE --> AP --> AL

    DR --> UN
    SR --> UN
    UN --> MN
    MN --> HS
    AL --> HS
    HS --> RK

    style AL fill:#ff6b6b,stroke:#333,color:#fff
    style HS fill:#4ecdc4,stroke:#333,color:#fff
    style DE fill:#45b7d1,stroke:#333,color:#fff
    style BM fill:#96ceb4,stroke:#333,color:#fff
```

### QDAP Architecture (Two Variants)

```mermaid
graph LR
    subgraph QDAP_S["QDAP-S (Lightweight)"]
        direction TB
        E1[Frozen Query Embedding<br/><i>768-dim</i>]
        L1[Linear Layer<br/><i>768 → 101</i>]
        C1[1D Conv<br/><i>kernel=7, smooth</i>]
        S1[Softmax]
        D1[α Distribution<br/><i>101 bins</i>]
        E1 --> L1 --> C1 --> S1 --> D1
    end

    subgraph QDAP_L["QDAP-L (Full Encoder)"]
        direction TB
        T2[Query Tokens]
        E2[Transformer Encoder<br/><i>~305M params</i>]
        N2[L2 Normalize]
        L2[Linear Layer<br/><i>768 → 101</i>]
        C2[1D Conv<br/><i>kernel=7, smooth</i>]
        S2[Softmax]
        D2[α Distribution<br/><i>101 bins</i>]
        T2 --> E2 --> N2 --> L2 --> C2 --> S2 --> D2
    end

    style D1 fill:#ff6b6b,stroke:#333,color:#fff
    style D2 fill:#ff6b6b,stroke:#333,color:#fff
    style E2 fill:#45b7d1,stroke:#333,color:#fff
```

> **Key Insight:** QDAP predicts a *distribution* over 101 alpha values [0.00, 0.01, ..., 1.00], not a single point estimate. The 1D convolution smooths adjacent bins, and the expected value gives the final α.

---

## Training Pipeline

### End-to-End Training Phases

```mermaid
graph LR
    P1["Phase 1<br/><b>Build BM25 Indices</b><br/><i>Per language + dataset</i>"]
    P2["Phase 2<br/><b>Mine Antagonist Negatives</b><br/><i>Algorithm A2</i>"]
    P3["Phase 3<br/><b>Train Dense Encoder</b><br/><i>InfoNCE + antagonist negs</i>"]
    P4["Phase 4<br/><b>Train QDAP</b><br/><i>CE + Wasserstein loss</i>"]
    P5["Phase 5<br/><b>Evaluate</b><br/><i>nDCG@10 on test sets</i>"]

    P1 --> P2 --> P3 --> P4 --> P5

    style P1 fill:#a8e6cf,stroke:#333
    style P2 fill:#dcedc1,stroke:#333
    style P3 fill:#ffd3b6,stroke:#333
    style P4 fill:#ffaaa5,stroke:#333
    style P5 fill:#ff8b94,stroke:#333
```

### Antagonist Negative Sampling (Algorithm A2)

```mermaid
graph TD
    subgraph Stage1["Stage 1: Filter"]
        D1[("Training Set<br/>(q, d⁺) pairs")]
        F1{"BM25 nDCG@10 < σ?"}
        K1[Keep — BM25 fails here]
        R1[Remove — BM25 sufficient]
        D1 --> F1
        F1 -->|Yes| K1
        F1 -->|No| R1
    end

    subgraph Stage2["Stage 2: Mine Negatives"]
        K1 --> C2[Retrieve Top-100<br/>from BOTH retrievers]
        C2 --> U2[Union candidate pool]
        U2 --> T2{"s_dense(q,c⁻) > s_dense(q,d⁺)<br/>AND<br/>s_sparse(q,c⁻) > s_sparse(q,d⁺)?"}
        T2 -->|Yes| AN[Antagonist Negative ✓]
        T2 -->|No| Skip[Discard]
    end

    style AN fill:#ff6b6b,stroke:#333,color:#fff
    style K1 fill:#ffd93d,stroke:#333
    style F1 fill:#6bcb77,stroke:#333,color:#fff
```

> **Intuition:** Antagonist negatives are documents that *both* retrievers incorrectly rank above the true positive. Training on these forces the dense encoder to learn complementary signals that BM25 misses.

### QDAP Target Construction (Algorithm A1)

```mermaid
graph TD
    Q["For each training query q"]
    S1["Get dense scores for top-100 candidates"]
    S2["Get sparse scores for top-100 candidates"]
    N["Min-max normalize both score sets"]
    
    subgraph Sweep["Sweep α ∈ {0.00, 0.01, ..., 1.00}"]
        H["s_hybrid = α·s_dense + (1-α)·s_sparse"]
        ND["Compute nDCG@10 for this α"]
    end
    
    V["101-dim vector of nDCG values"]
    SM["Softmax → Target distribution"]

    Q --> S1
    Q --> S2
    S1 --> N
    S2 --> N
    N --> Sweep
    H --> ND
    Sweep --> V --> SM

    style SM fill:#ff6b6b,stroke:#333,color:#fff
    style Sweep fill:#f0f0f0,stroke:#333
```

---

## Loss Function

The QDAP predictor is trained with a composite loss combining cross-entropy and Wasserstein distance:

```mermaid
graph LR
    subgraph Loss["Composite Loss (Eq. 9)"]
        direction TB
        CE["L_CE: Squared Cross-Entropy<br/><i>Emphasizes high-nDCG bins</i>"]
        WD["L_WD: 1D Wasserstein Distance<br/><i>Penalizes CDF mismatch</i>"]
        CL["L = 0.62·L_CE + 0.38·L_WD"]
        CE --> CL
        WD --> CL
    end

    style CL fill:#4ecdc4,stroke:#333,color:#fff
    style CE fill:#45b7d1,stroke:#333,color:#fff
    style WD fill:#96ceb4,stroke:#333,color:#fff
```

| Component | Formula | Purpose |
|:---|:---|:---|
| **Squared CE** | `-Σ (ŷ²/‖ŷ²‖₁) · log(p)` | Sharpens target, focuses on optimal α region |
| **Wasserstein** | `Σ \|CDF(p) - CDF(ŷ)\|` | Respects ordinal structure of alpha bins |
| **λ weight** | `0.62` | Empirically selected balance |

---

## Results

### Main Results (nDCG@10)

| Model | MLDR | MIRACL | **Average** |
|:---|:---:|:---:|:---:|
| BM25 baseline | 53.6 | 31.7 | 42.6 |
| BGE-M3 (Dense+Sparse+Multi-vec) | 65.0 | 71.2 | 68.1 |
| mGTE-TRM (Dense+Sparse) | 71.3 | 64.7 | 68.0 |
| **HTR (Ours)** | **74.3** | **67.1** | **70.7** |
| HTR Oracle (upper bound) | 79.1 | 73.8 | 76.4 |

### Performance Comparison

```mermaid
xychart-beta
    title "nDCG@10 Comparison Across Models"
    x-axis ["BM25", "BGE-M3", "mGTE-TRM", "HTR (Ours)", "Oracle"]
    y-axis "nDCG@10 (Average)" 40 --> 80
    bar [42.6, 68.1, 68.0, 70.7, 76.4]
```

### Multilingual Coverage

```mermaid
graph LR
    subgraph MLDR["MLDR (13 languages)"]
        ML[ar, de, en, es, fr, hi,<br/>it, ja, ko, pt, ru, th, zh]
    end
    subgraph MIRACL["MIRACL (16 languages)"]
        MR[ar, bn, en, es, fa, fi, fr,<br/>hi, id, ja, ko, ru, sw, te, th, zh]
    end
    subgraph Combined["Joint Training"]
        JT[Single multilingual model<br/>trained on ALL languages]
    end
    
    MLDR --> Combined
    MIRACL --> Combined

    style Combined fill:#4ecdc4,stroke:#333,color:#fff
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/query-adaptive-hybrid-search.git
cd query-adaptive-hybrid-search

# Install dependencies
pip install -e .

# (Optional) GPU support with FAISS-GPU
pip install faiss-gpu

# (Optional) Experiment tracking
pip install wandb
```

### Requirements

- Python 3.9+
- PyTorch 2.0+
- Transformers 4.36+
- ~16GB GPU memory (for training)
- ~50GB disk space (for datasets)

---

## Usage

### Full Pipeline

```bash
# Phase 1: Download datasets
python scripts/download_data.py --datasets mldr miracl --languages all

# Phase 2: Build BM25 indices (language-specific tokenization)
python scripts/build_bm25_index.py --datasets mldr miracl --languages all

# Phase 3: Mine antagonist negatives (offline, before training)
python scripts/mine_antagonist_negatives.py \
    --base_model "Alibaba-NLP/gte-multilingual-base" \
    --threshold_sigma 0.5 \
    --top_k 100

# Phase 4: Train dense encoder (3 epochs, InfoNCE + antagonist negatives)
python scripts/train.py --stage dense \
    --config configs/train_dense.yaml

# Phase 5: Train QDAP predictor (composite CE + Wasserstein loss)
python scripts/train.py --stage qdap \
    --config configs/train_qdap.yaml

# Phase 6: Evaluate (nDCG@10, oracle, optimal baselines)
python scripts/evaluate.py --config configs/eval.yaml
```

### Inference Example

```python
from src.models import DenseEncoder, QDAP_L, HybridRetriever
from src.data import BM25Index

# Load models
dense = DenseEncoder("Alibaba-NLP/gte-multilingual-base")
qdap = QDAP_L.from_dense_encoder("checkpoints/dense_encoder/best.pt")
bm25 = BM25Index.load("data/bm25_indices/miracl/en.pkl")

# Create hybrid retriever
retriever = HybridRetriever(dense, bm25, qdap, fusion="minmax")

# Search — alpha is automatically predicted per-query
results = retriever.search("What causes northern lights?", top_k=10)
for doc_id, score in results:
    print(f"  {doc_id}: {score:.4f}")
```

---

## Project Structure

```
├── configs/
│   ├── train_dense.yaml          # Dense encoder training config
│   ├── train_qdap.yaml           # QDAP predictor training config
│   └── eval.yaml                 # Evaluation config
├── src/
│   ├── models/
│   │   ├── dense_encoder.py      # mGTE-based dual encoder (768-dim, RoPE)
│   │   ├── qdap_s.py             # QDAP-S: lightweight adapter (~1M params)
│   │   ├── qdap_l.py             # QDAP-L: full encoder (~305M params)
│   │   └── hybrid_retriever.py   # End-to-end hybrid retrieval pipeline
│   ├── data/
│   │   ├── dataset_loader.py     # MLDR + MIRACL from HuggingFace
│   │   ├── antagonist_sampler.py # Algorithm A2: antagonist negative mining
│   │   └── bm25_index.py         # BM25 with trigram tokenization for CJK+
│   ├── training/
│   │   ├── train_dense.py        # InfoNCE training with antagonist negatives
│   │   ├── train_qdap.py         # QDAP training (Algorithm A1)
│   │   └── losses.py             # InfoNCE, Squared CE, Wasserstein, Composite
│   ├── evaluation/
│   │   ├── metrics.py            # nDCG@10 computation
│   │   ├── evaluate.py           # Full evaluation pipeline
│   │   └── oracle.py             # Oracle + optimal-α upper bounds
│   └── utils/
│       ├── score_normalization.py # Min-max, RRF, hybrid fusion
│       └── tokenizers.py         # Language-specific + trigram tokenizers
├── scripts/
│   ├── download_data.py          # Dataset download
│   ├── build_bm25_index.py       # BM25 index construction
│   ├── mine_antagonist_negatives.py  # Hard negative mining
│   ├── train.py                  # Training entry point
│   └── evaluate.py               # Evaluation entry point
├── tests/
│   ├── test_losses.py            # Loss function tests
│   ├── test_qdap.py              # QDAP model tests
│   └── test_normalization.py     # Score normalization tests
├── requirements.txt
├── setup.py
└── README.md
```

---

## Key Design Decisions

```mermaid
mindmap
    root((HTR Design))
        Retrieval
            Dense: mGTE backbone
            Sparse: BM25 per-language
            Fusion: Min-Max normalization
        QDAP
            101 alpha bins
            1D Conv smoothing
            Distribution output
            Expected value → α
        Training
            Antagonist negatives
            InfoNCE τ=0.05
            Joint multilingual
            3 epochs
        Loss
            Squared CE λ=0.62
            Wasserstein 1-λ=0.38
            Ordinal-aware
```

---

## Hyperparameters

| Parameter | Value | Source |
|:---|:---:|:---|
| Embedding dimension | 768 | Section 4.1 |
| Dense encoder params | ~305M | Section 4.1 |
| QDAP bins | 101 | α ∈ {0.00, 0.01, ..., 1.00} |
| QDAP conv kernel | 7 | Section 3.3 |
| Loss λ (CE weight) | 0.62 | Section 3.3 |
| BM25 k₁ / b (MIRACL) | 0.9 / 0.4 | Section 3.2 |
| BM25 k₁ / b (MLDR) | 1.2 / 0.75 | Section 3.2 |
| RRF k | 60 | Section 3.2 |
| Candidate pool size | 100 | Section 3.2 |
| InfoNCE temperature τ | 0.05 | Standard |
| Training epochs | 3 | Figures 6–7 |
| Random seeds | 10 | Section 4 |

---

## Citation

```bibtex
@article{posokhov2026query,
  title={Query-Adaptive Hybrid Search},
  author={Posokhov, Pavel and others},
  journal={Machine Learning and Knowledge Extraction},
  volume={8},
  number={4},
  pages={91},
  year={2026},
  publisher={MDPI},
  doi={10.3390/make8040091}
}
```

---

## License

This implementation is for **research purposes only**. Please cite the original paper if you use this code in your work.

---

<div align="center">
<i>Built with PyTorch and Transformers</i>
</div>
