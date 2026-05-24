# Architecture Deep Dive

Detailed technical diagrams for the Query-Adaptive Hybrid Search system.

---

## 1. Dense Encoder Architecture

```mermaid
graph TD
    subgraph Input
        T["Input Tokens<br/>[CLS] t₁ t₂ ... tₙ [SEP]"]
    end

    subgraph Backbone["Transformer Backbone (~305M params)"]
        EMB[Token + RoPE Embeddings]
        TL1[Transformer Layer 1]
        TL2[Transformer Layer 2]
        TLN[Transformer Layer N<br/><i>12 layers, 768 hidden</i>]
    end

    subgraph Output
        CLS["[CLS] Hidden State"]
        NORM[L2 Normalization]
        VEC["Unit Vector ∈ ℝ⁷⁶⁸"]
    end

    T --> EMB --> TL1 --> TL2 --> TLN --> CLS --> NORM --> VEC

    style VEC fill:#45b7d1,stroke:#333,color:#fff
    style NORM fill:#96ceb4,stroke:#333
```

### Dual-Encoder Scoring

```mermaid
graph LR
    subgraph Query
        QT[Query Tokens] --> QE[Encoder] --> QV["q ∈ ℝ⁷⁶⁸"]
    end

    subgraph Document
        DT[Doc Tokens] --> DE[Encoder<br/><i>shared weights</i>] --> DV["d ∈ ℝ⁷⁶⁸"]
    end

    QV --> DOT["score = q · d<br/><i>(cosine similarity)</i>"]
    DV --> DOT

    style DOT fill:#4ecdc4,stroke:#333,color:#fff
```

---

## 2. BM25 Scoring Formula

```mermaid
graph TD
    subgraph Formula["BM25 Score (Eq. 5)"]
        TF["Term Frequency<br/>tf_j in document c"]
        IDF["Inverse Doc Frequency<br/>ln((N - cf_j + 0.5) / (cf_j + 0.5))"]
        LN["Length Normalization<br/>k₁·((1-b) + b·|c|/avg|c|)"]
        TFC["TF Component<br/>(k₁+1)·tf / (LN + tf)"]
        SCORE["w_j(c,C) = TFC × IDF"]
    end

    TF --> TFC
    LN --> TFC
    TFC --> SCORE
    IDF --> SCORE

    style SCORE fill:#96ceb4,stroke:#333,color:#fff
```

### Language-Specific Tokenization Strategy

```mermaid
graph TD
    subgraph Decision["Tokenization Router"]
        LANG{Language?}
    end

    subgraph Trigram["Trigram Tokenization"]
        TR["Remove whitespace → Character trigrams<br/><i>e.g., '你好世界' → ['你好世', '好世界']</i>"]
        TRL["Used for: hi, zh, ja, ko, bn, te"]
    end

    subgraph Whitespace["Whitespace Tokenization"]
        WS["Split on spaces → Remove stopwords<br/><i>e.g., 'the quick fox' → ['quick', 'fox']</i>"]
        WSL["Used for: en, de, fr, es, it, pt, ru, ar, ..."]
    end

    LANG -->|"CJK + South Asian"| Trigram
    LANG -->|"Space-delimited"| Whitespace

    style LANG fill:#ffd93d,stroke:#333
    style TR fill:#ff6b6b,stroke:#333,color:#fff
    style WS fill:#45b7d1,stroke:#333,color:#fff
```

---

## 3. Score Normalization and Fusion

```mermaid
graph TD
    subgraph Retrieval["Per-Query Retrieval"]
        D100["Dense Top-100<br/>{d₁:0.92, d₂:0.87, ..., d₁₀₀:0.31}"]
        S100["Sparse Top-100<br/>{d₅:14.2, d₁:12.8, ..., d₈₈:1.1}"]
    end

    subgraph Normalize["Min-Max Normalization (per retriever)"]
        DN["Dense Normalized<br/>{d₁:1.00, d₂:0.92, ..., d₁₀₀:0.00}"]
        SN["Sparse Normalized<br/>{d₅:1.00, d₁:0.89, ..., d₈₈:0.00}"]
    end

    subgraph Union["Candidate Union"]
        ALL["All unique docs from both sets<br/>Missing score → 0.0"]
    end

    subgraph Fuse["Hybrid Score"]
        HS["s(d) = α · s_dense(d) + (1-α) · s_sparse(d)"]
    end

    D100 --> DN
    S100 --> SN
    DN --> ALL
    SN --> ALL
    ALL --> HS

    style HS fill:#4ecdc4,stroke:#333,color:#fff
```

---

## 4. QDAP Training Target Distribution

```mermaid
graph LR
    subgraph Example["Example: Query 'quantum entanglement applications'"]
        direction TB
        A0["α=0.0 → nDCG=0.42"]
        A3["α=0.3 → nDCG=0.58"]
        A6["α=0.6 → nDCG=0.71"]
        A7["α=0.7 → nDCG=0.73 ← peak"]
        A8["α=0.8 → nDCG=0.68"]
        A1["α=1.0 → nDCG=0.55"]
    end

    subgraph Target["After Softmax"]
        DIST["Target Distribution<br/>Peak around α=0.7<br/><i>This query benefits<br/>more from dense retrieval</i>"]
    end

    Example --> Target

    style A7 fill:#ff6b6b,stroke:#333,color:#fff
    style DIST fill:#4ecdc4,stroke:#333,color:#fff
```

---

## 5. Composite Loss Visualization

```mermaid
graph TD
    subgraph Predicted["Predicted Distribution P"]
        PP["QDAP output (101 bins)"]
    end

    subgraph Target["Target Distribution Q"]
        QQ["nDCG-based (101 bins)"]
    end

    subgraph CE_Loss["Cross-Entropy Path"]
        SQ["Square target: Q² / ‖Q²‖₁"]
        CEL["-Σ Q²_norm · log(P)"]
    end

    subgraph WD_Loss["Wasserstein Path"]
        CDF_P["CDF(P) = cumsum(P)"]
        CDF_Q["CDF(Q) = cumsum(Q)"]
        WDL["Σ |CDF(Q) - CDF(P)|"]
    end

    subgraph Final["Combined"]
        LOSS["L = 0.62·L_CE + 0.38·L_WD"]
    end

    PP --> CEL
    QQ --> SQ --> CEL
    PP --> CDF_P --> WDL
    QQ --> CDF_Q --> WDL
    CEL --> LOSS
    WDL --> LOSS

    style LOSS fill:#ff6b6b,stroke:#333,color:#fff
    style SQ fill:#ffd93d,stroke:#333
```

---

## 6. Full Training Data Flow

```mermaid
sequenceDiagram
    participant DS as Datasets (MLDR + MIRACL)
    participant BM as BM25 Index
    participant DE as Dense Encoder
    participant AN as Antagonist Sampler
    participant QD as QDAP

    Note over DS,QD: Phase 1-2: Preparation
    DS->>BM: Build language-specific indices
    DS->>AN: Provide (query, doc⁺) pairs
    BM->>AN: BM25 scores for filtering (σ < 0.5)
    DE->>AN: Dense scores for negative selection

    Note over DS,QD: Phase 3: Dense Encoder Training
    AN->>DE: Antagonist negatives + in-batch negatives
    Note right of DE: InfoNCE loss, τ=0.05, 3 epochs

    Note over DS,QD: Phase 4: QDAP Training
    DE->>QD: Dense scores (frozen encoder)
    BM->>QD: Sparse scores
    Note right of QD: Sweep α → nDCG curve → softmax target
    Note right of QD: Composite loss (CE + Wasserstein)

    Note over DS,QD: Phase 5: Inference
    DE-->>QD: Query embedding (parallel)
    QD-->>QD: Predict α
```

---

## 7. Comparison with Baselines

```mermaid
graph TD
    subgraph Static["Static Fusion (Baselines)"]
        SA["Single α for all queries<br/><i>BGE-M3, mGTE-TRM</i>"]
        SA --> SF["Same weight regardless<br/>of query type"]
    end

    subgraph Adaptive["Adaptive Fusion (Ours)"]
        QA["Per-query α prediction<br/><i>HTR with QDAP</i>"]
        QA --> KW["Keyword query → low α<br/><i>favor BM25</i>"]
        QA --> SEM["Semantic query → high α<br/><i>favor dense</i>"]
        QA --> MIX["Mixed query → balanced α<br/><i>use both equally</i>"]
    end

    style SA fill:#ffaaa5,stroke:#333
    style QA fill:#4ecdc4,stroke:#333,color:#fff
    style KW fill:#96ceb4,stroke:#333
    style SEM fill:#45b7d1,stroke:#333,color:#fff
    style MIX fill:#ffd93d,stroke:#333
```
