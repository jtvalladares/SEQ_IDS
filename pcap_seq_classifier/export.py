"""Funciones para exportar el modelo y metadatos (TorchScript, ONNX, checkpoints).
"""
from typing import Any
import torch


def export_torchscript(model: torch.nn.Module, example_input: Any, out_path: str) -> None:
    """Exporta modelo a TorchScript."""
    raise NotImplementedError


def export_onnx(model: torch.nn.Module, example_input: Any, out_path: str) -> None:
    """Exporta modelo a ONNX."""
    raise NotImplementedError


def save_metadata(metadata: dict, out_path: str) -> None:
    """Guarda metadata JSON (simhash params, model params, seed, date, etc.)."""
    raise NotImplementedError