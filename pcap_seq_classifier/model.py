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
        """
        Args:
            embedding_dim: tamaño del embedding o del vector de entrada directo.
            hidden_size: tamaño del estado oculto de la RNN.
            num_layers: número de capas en LSTM/GRU.
            bidirectional: True → RNN bidireccional.
            dropout: dropout entre capas (si num_layers > 1).
            rnn_type: "lstm" o "gru".
            vocab_size: requerido si use_embedding_layer=True.
            use_embedding_layer: si True, espera índices; si False, espera vectores.
        """
        super().__init__()

        self.use_embedding = use_embedding_layer

        # 1) Capa de embedding opcional
        if use_embedding_layer:
            if vocab_size is None:
                raise ValueError("Debes proporcionar vocab_size cuando use_embedding_layer=True.")
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            rnn_input_dim = embedding_dim
        else:
            # El usuario entrega directamente vectores embebidos
            self.embedding = None
            rnn_input_dim = embedding_dim

        # 2) RNN (LSTM o GRU)
        rnn_cls = nn.LSTM if rnn_type.lower() == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=rnn_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 3) Clasificador final
        # Si bidireccional, concatenamos ambos lados
        fc_in = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(fc_in, 1)

    # ----------------------------------------------------------------------

    def forward(self, packed_sequence: torch.nn.utils.rnn.PackedSequence) -> torch.Tensor:
        """
        Args:
            packed_sequence: PackedSequence a procesar por la RNN.

        Returns:
            logits: Tensor (batch, 1)
        """
        # Asegurar PackedSequence correcto
        if not isinstance(packed_sequence, torch.nn.utils.rnn.PackedSequence):
            raise TypeError("forward() espera un PackedSequence.")

        # Si hay embedding, mapear datos antes de repackear
        if self.use_embedding:
            data, batch_sizes, sorted_idx, unsorted_idx = (
                packed_sequence.data,
                packed_sequence.batch_sizes,
                packed_sequence.sorted_indices,
                packed_sequence.unsorted_indices,
            )
            emb = self.embedding(data)

            packed_emb = torch.nn.utils.rnn.PackedSequence(
                data=emb,
                batch_sizes=batch_sizes,
                sorted_indices=sorted_idx,
                unsorted_indices=unsorted_idx,
            )
            rnn_out, hidden = self.rnn(packed_emb)

        else:
            # Caso: viene ya embebido
            rnn_out, hidden = self.rnn(packed_sequence)

        # hidden:
        # LSTM → (h_n, c_n)
        if isinstance(hidden, tuple):
            hidden = hidden[0]  # tomar h_n

        # hidden shape: (num_layers * num_directions, batch, hidden_size)
        # Tomamos la última capa
        last_layer = hidden[-1] if not self.rnn.bidirectional else torch.cat(
            (hidden[-2], hidden[-1]), dim=1
        )

        logits = self.fc(last_layer)
        return logits
