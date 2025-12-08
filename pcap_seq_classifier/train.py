"""Loop de entrenamiento y evaluación.

Funciones expuestas:
- train_loop(...)
- evaluate(...)
- save_checkpoint(...)
"""
from typing import Any, Dict
import torch


def train_loop(train_loader, val_loader, model, optimizer, scheduler, cfg: Dict[str, Any]):
    """Implementa el loop de entrenamiento con checkpoints y early stopping."""
    raise NotImplementedError


def evaluate(loader, model) -> Dict[str, float]:
    """Evalúa el modelo y retorna un diccionario con métricas (accuracy, precision, recall, f1, auc)."""
    raise NotImplementedError


def save_checkpoint(state: Dict[str, Any], path: str) -> None:
    """Guarda estado de entrenamiento (model_state, optimizer_state, epoch, metadata)."""
    raise NotImplementedError