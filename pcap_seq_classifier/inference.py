"""
Inferencia incremental por sesión: procesa fingerprints paquete a paquete y registra decisiones.

Funciones expuestas:
- incremental_inference_for_session(model, fingerprints_iterable, threshold, device)
- predict_session_early(model, session_fps, threshold)
"""
from typing import Iterable, Dict, Any, List
import torch


def _predict_single(model, fp: int, device: str):
    """Ejecuta predicción del modelo para un solo fingerprint (tensor de 1x1 o 1xd)."""
    model.eval()
    with torch.no_grad():
        # asumimos que fp es un entero hash; se convierte a tensor (batch=1)
        x = torch.tensor([fp], dtype=torch.long, device=device)
        logits = model(x)                       # salida de tu modelo
        probs = torch.softmax(logits, dim=1)    # normaliza
        score, label = torch.max(probs, dim=1)  # prob máxima y label asociado
        return float(score.item()), int(label.item())


def incremental_inference_for_session(
    model,
    fingerprints_iterable: Iterable[int],
    threshold: float = 0.9,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Procesa fingerprints uno a uno y aplica "early stopping" si se supera el threshold.

    Retorna:
      - first_decision_label
      - packets_used_first
      - changes_list: [(packet_idx, old_label, new_label)]
      - final_label
      - packet_count
    """
    first_decision_label = None
    packets_used_first = None
    changes_list = []
    prev_label = None

    packet_count = 0

    for fp in fingerprints_iterable:
        score, label = _predict_single(model, fp, device=device)
        packet_count += 1

        # Registrar cambios en predicción
        if prev_label is not None and label != prev_label:
            changes_list.append((packet_count, prev_label, label))
        prev_label = label

        # Registrar primera decisión fuerte
        if first_decision_label is None and score >= threshold:
            first_decision_label = label
            packets_used_first = packet_count

    # Si nunca se alcanzó el threshold, la decisión final es el último label
    final_label = prev_label

    return {
        "first_decision_label": first_decision_label,
        "packets_used_first": packets_used_first,
        "changes_list": changes_list,
        "final_label": final_label,
        "packet_count": packet_count,
    }


def predict_session_early(
    model,
    session_fps: List[int],
    threshold: float = 0.9,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Simula inferencia incremental en modo batch (offline).
    Es equivalente a incremental_inference_for_session(session_fps).
    """
    return incremental_inference_for_session(
        model,
        fingerprints_iterable=session_fps,
        threshold=threshold,
        device=device
    )
