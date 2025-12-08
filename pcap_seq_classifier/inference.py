"""Inferencia incremental por sesión: procesa fingerprints paquete a paquete y registra decisiones.

Funciones expuestas:
- incremental_inference_for_session(model, fingerprints_iterable, threshold, device)
- predict_session_early(model, session_fps, threshold)
"""
from typing import Iterable, Dict, Any, List
import torch


def incremental_inference_for_session(model, fingerprints_iterable: Iterable[int], threshold: float = 0.9, device: str = "cpu") -> Dict[str, Any]:
    """Crea una instancia de inferencia que recibe fingerprints uno a uno.

    Debe retornar un registro con: first_decision_label, packets_used_first, changes_list, final_label, packet_count.
    """
    raise NotImplementedError


def predict_session_early(model, session_fps: List[int], threshold: float = 0.9, device: str = "cpu") -> Dict[str, Any]:
    """Función batch que simula la inferencia incremental y devuelve el mismo esquema de resultados.

    Útil para testing offline.
    """
    raise NotImplementedError