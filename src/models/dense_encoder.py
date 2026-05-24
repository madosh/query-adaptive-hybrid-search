"""Dense encoder based on mGTE architecture with RoPE positional embeddings."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Union
from transformers import AutoModel, AutoTokenizer


class DenseEncoder(nn.Module):
    """Transformer-based dual-encoder following mGTE architecture.

    Uses a multilingual transformer backbone (e.g., gte-multilingual-base)
    with RoPE positional encodings. Outputs L2-normalized [CLS] embeddings.
    """

    def __init__(
        self,
        model_name: str = "Alibaba-NLP/gte-multilingual-base",
        embedding_dim: int = 768,
        max_seq_length: int = 512,
        use_flash_attention: bool = False,
    ):
        super().__init__()
        attn_impl = "flash_attention_2" if use_flash_attention else "eager"
        self.encoder = AutoModel.from_pretrained(
            model_name,
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode input tokens and return L2-normalized [CLS] embedding.

        Args:
            input_ids: Token IDs of shape (batch, seq_len).
            attention_mask: Attention mask of shape (batch, seq_len).

        Returns:
            Normalized embeddings of shape (batch, embedding_dim).
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return F.normalize(cls_embedding, p=2, dim=-1)

    def score(self, q_emb: torch.Tensor, c_emb: torch.Tensor) -> torch.Tensor:
        """Cosine similarity between query and candidate embeddings.

        Since embeddings are L2-normalized, cosine similarity = dot product.
        """
        return torch.sum(q_emb * c_emb, dim=-1)

    def score_multi(self, q_emb: torch.Tensor, c_embs: torch.Tensor) -> torch.Tensor:
        """Score a query against multiple candidates.

        Args:
            q_emb: Query embedding of shape (dim,) or (1, dim).
            c_embs: Candidate embeddings of shape (num_candidates, dim).

        Returns:
            Scores of shape (num_candidates,).
        """
        if q_emb.dim() == 1:
            q_emb = q_emb.unsqueeze(0)
        return torch.mm(q_emb, c_embs.t()).squeeze(0)

    def tokenize(self, texts: Union[str, List[str]], max_length: Optional[int] = None) -> dict:
        """Tokenize text(s) for the encoder."""
        if max_length is None:
            max_length = self.max_seq_length
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    @torch.no_grad()
    def encode_texts(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """Encode text(s) into embeddings (inference mode).

        Args:
            texts: Single text or list of texts.
            batch_size: Batch size for encoding.

        Returns:
            Numpy array of embeddings, shape (num_texts, embedding_dim).
        """
        if isinstance(texts, str):
            texts = [texts]

        self.eval()
        all_embeddings = []
        device = next(self.parameters()).device

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded = self.tokenize(batch_texts)
            encoded = {k: v.to(device) for k, v in encoded.items()}
            embeddings = self.encode(encoded["input_ids"], encoded["attention_mask"])
            all_embeddings.append(embeddings.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)
