"""QDAP-L: Full encoder-scale alpha predictor.

Initialized from trained dense encoder weights. Runs in parallel with the
dense retriever at inference (processes query only, not documents).
~300M parameters, much faster than 7B+ LLM approaches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Optional


class QDAP_L(nn.Module):
    """Query-Driven Alpha Prediction — Large variant.

    A full transformer encoder (initialized from dense retriever weights)
    with a classification head that predicts a distribution over 101 alpha bins.
    """

    def __init__(
        self,
        model_name: str = "Alibaba-NLP/gte-multilingual-base",
        num_bins: int = 101,
        conv_kernel: int = 7,
        max_seq_length: int = 512,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.max_seq_length = max_seq_length
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        hidden_size = self.encoder.config.hidden_size
        self.linear = nn.Linear(hidden_size, num_bins)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=conv_kernel, padding=conv_kernel // 2)

    @classmethod
    def from_dense_encoder(cls, dense_encoder_path: str, num_bins: int = 101, conv_kernel: int = 7):
        """Initialize QDAP-L from a trained dense encoder checkpoint.

        Copies encoder weights from the trained dense model, then adds
        the QDAP classification head (randomly initialized).
        """
        from .dense_encoder import DenseEncoder

        dense = DenseEncoder.__new__(DenseEncoder)
        state = torch.load(dense_encoder_path, map_location="cpu")

        model_name = state.get("model_name", "Alibaba-NLP/gte-multilingual-base")
        instance = cls(model_name=model_name, num_bins=num_bins, conv_kernel=conv_kernel)

        encoder_state = {
            k.replace("encoder.", ""): v
            for k, v in state["model_state_dict"].items()
            if k.startswith("encoder.")
        }
        instance.encoder.load_state_dict(encoder_state)
        return instance

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Predict alpha distribution from raw query tokens.

        Args:
            input_ids: Token IDs of shape (batch, seq_len).
            attention_mask: Attention mask of shape (batch, seq_len).

        Returns:
            Probability distribution over alpha bins, shape (batch, num_bins).
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state[:, 0, :]
        hidden = F.normalize(hidden, p=2, dim=-1)
        z = self.linear(hidden).unsqueeze(1)
        z_smooth = self.conv1d(z).squeeze(1)
        return F.softmax(z_smooth, dim=-1)

    def predict_alpha(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Predict scalar alpha as expected value of the distribution.

        Args:
            input_ids: Token IDs of shape (batch, seq_len).
            attention_mask: Attention mask of shape (batch, seq_len).

        Returns:
            Predicted alpha values, shape (batch,).
        """
        dist = self.forward(input_ids, attention_mask)
        alpha_values = torch.linspace(0, 1, self.num_bins, device=dist.device)
        return (dist * alpha_values).sum(dim=-1)

    def predict_alpha_from_text(self, texts, device: Optional[torch.device] = None) -> torch.Tensor:
        """Predict alpha directly from text strings.

        Args:
            texts: Single text or list of texts.
            device: Device to run inference on.

        Returns:
            Predicted alpha values, shape (batch,).
        """
        if isinstance(texts, str):
            texts = [texts]
        encoded = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_seq_length, return_tensors="pt",
        )
        if device is not None:
            encoded = {k: v.to(device) for k, v in encoded.items()}
        return self.predict_alpha(encoded["input_ids"], encoded["attention_mask"])
