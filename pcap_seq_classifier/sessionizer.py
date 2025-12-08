"""Funciones para agrupar paquetes en sesiones.

Funciones expuestas:
- packet_to_session_key(pkt) -> tuple
- group_packets_by_session(packets) -> Dict[session_key, List[Packet]]
- sort_session_packets(sessions)
"""
from typing import List, Dict, Tuple, Any


def packet_to_session_key(pkt: dict) -> tuple:
    """Convierte un paquete (dict) en una clave de sesión (5-tuple o similar).

    Debe normalizar dirección/puertos para que el orden (A→B vs B→A) sea consistente si corresponde.
    """
    raise NotImplementedError


def group_packets_by_session(packets: List[dict]) -> Dict[tuple, List[dict]]:
    """Agrupa la lista de paquetes por clave de sesión."""
    raise NotImplementedError


def sort_session_packets(sessions: Dict[tuple, List[dict]]) -> Dict[tuple, List[dict]]:
    """Ordena en-place o retorna sesiones con sus paquetes ordenados por timestamp."""
    raise NotImplementedError