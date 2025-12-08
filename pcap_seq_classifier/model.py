"""Modelo secuencial PyTorch (LSTM/GRU) para clasificación binaria por sesión.

Clase: SeqClassifier
"""
from typing import Optional
import torch
import torch.nn as nn


class SeqClassifier(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
        rnn_type: str = "lstm",
        vocab_size: Optional[int] = None,
        use_embedding_layer: bool = True,
    ) -> None:
        """Inicializa el modelo.

        - Si use_embedding_layer=True, el input serán índices (p.ej. hashed fingerprints modulados)
        - Alternativamente, el input puede ser vectores ya embebidos.
        """
        super().__init__()
        raise NotImplementedError

    def forward(self, packed_sequence: torch.nn.utils.rnn.PackedSequence) -> torch.Tensor:
        """Recibe PackedSequence y retorna logits shape (batch, 1) o (batch,).

        Debe soportar inferencia paso a paso (ver inference.py)."""
        raise NotImplementedError