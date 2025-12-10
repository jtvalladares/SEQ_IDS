# dataset.py
"""
Dataset para cargar payloads, fingerprints y embeddings desde disco o memoria.
Diseñado para integrarse con PyTorch DataLoader.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Optional, Dict, Any, Tuple


class PayloadDataset(Dataset):
    """
    Dataset que contiene payloads brutos, fingerprints SimHash y/o embeddings.

    Se puede inicializar de tres formas:

    1. Solo payloads -> para aplicar procesamiento en el DataLoader
    2. Payloads + fingerprints
    3. Payloads + embeddings + labels (clasificación)
    """

    def __init__(
        self,
        payloads: List[bytes],
        fingerprints: Optional[List[int]] = None,
        embeddings: Optional[np.ndarray] = None,
        labels: Optional[List[int]] = None,
    ):
        self.payloads = payloads
        self.fingerprints = fingerprints
        self.embeddings = embeddings
        self.labels = labels

        n = len(payloads)

        if fingerprints is not None:
            assert len(fingerprints) == n, "fingerprints y payloads deben tener igual longitud"
        if embeddings is not None:
            assert embeddings.shape[0] == n, "embeddings y payloads deben tener igual longitud"
        if labels is not None:
            assert len(labels) == n, "labels y payloads deben tener igual longitud"

    def __len__(self):
        return len(self.payloads)

    def __getitem__(self, idx):
        item = {"payload": self.payloads[idx]}

        if self.fingerprints is not None:
            # convertir fingerprint a tensor de 64 bits
            fp = self.fingerprints[idx]
            item["fingerprint"] = torch.tensor(fp, dtype=torch.long)

        if self.embeddings is not None:
            emb = self.embeddings[idx]
            item["embedding"] = torch.tensor(emb, dtype=torch.float32)

        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


# Funciones auxiliares -------------------------------------------------------

def load_payloads(path: str) -> List[bytes]:
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)

def load_fingerprints(path: str) -> List[int]:
    import numpy as np
    return np.load(path).tolist()

def load_embeddings(path: str) -> np.ndarray:
    import numpy as np
    return np.load(path)

class SessionDataset(Dataset):
    def __init__(self, sessions: List[Tuple[tuple, List[int]]], labels: Dict[tuple, int], transform=None):
        self.sessions = sessions
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx: int):
        session_key, fp_list = self.sessions[idx]
        if self.transform:
            fp_list = self.transform(fp_list)
        label = self.labels.get(session_key, 0)
        return torch.tensor(fp_list, dtype=torch.long), label


def collate_fn(batch: List[Any]):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded, lengths, labels
