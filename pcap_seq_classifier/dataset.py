"""Dataset y collate function para entrenar modelos secuenciales con PyTorch.

Clases / funciones:
- SessionDataset(sessions: List[Tuple[session_key, List[fps]]], labels: Dict[session_key, int])
- collate_fn(batch)

Nota: las secuencias tienen longitud variable; collate_fn debe retornar tensores padded y lengths.
"""
from typing import List, Tuple, Dict, Any
import torch
from torch.utils.data import Dataset


class SessionDataset(Dataset):
    def __init__(self, sessions: List[Tuple[tuple, List[int]]], labels: Dict[tuple, int], transform=None):
        """sessions: lista de (session_key, fingerprint_list)
        labels: mapping session_key -> 0/1
        transform: opcional, p.ej. normalización o embedding hashing
        """
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx: int):
        raise NotImplementedError


def collate_fn(batch: List[Any]):
    """Transforma batch de sesiones a (padded_tensor, lengths, labels).

    Debe retornar tensores de torch adecuados para el modelo y las longitudes originales.
    """
    raise NotImplementedError