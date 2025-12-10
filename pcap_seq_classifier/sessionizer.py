"""
Funciones para agrupar paquetes en sesiones.

Funciones expuestas:
- packet_to_session_key(pkt) -> tuple
- group_packets_by_session(packets) -> Dict[session_key, List[Packet]]
- sort_session_packets(sessions)
"""
from typing import List, Dict, Tuple, Any


def packet_to_session_key(pkt: Any) -> tuple:
    """
    Convierte un paquete en una clave de sesión normalizada.
    Funciona con:
        - diccionarios {'src', 'dst', 'sport', 'dport', 'proto'}
        - Paquetes de scapy (IP/TCP/UDP)

    Normalización:
        La clave es ordenada para que:
            (A:port1 → B:port2) == (B:port2 → A:port1)
        Esto permite agrupar ambos sentidos en la misma sesión.

    Formato de clave:
        (proto, min_ip, min_port, max_ip, max_port)
    """

    # --- Compatibilidad con Scapy ---
    if hasattr(pkt, "haslayer"):
        # obtener capa IPv4 o IPv6
        if pkt.haslayer("IP"):
            src = pkt["IP"].src
            dst = pkt["IP"].dst
            proto = pkt["IP"].proto
        elif pkt.haslayer("IPv6"):
            src = pkt["IPv6"].src
            dst = pkt["IPv6"].dst
            proto = pkt["IPv6"].nh
        else:
            # no tiene capa IP → no se considera sesión
            return ("NOIP", id(pkt))

        # capa de transporte
        sport = pkt.sport if hasattr(pkt, "sport") else 0
        dport = pkt.dport if hasattr(pkt, "dport") else 0

    # --- Diccionario genérico ---
    elif isinstance(pkt, dict):
        src = pkt.get("src", "0.0.0.0")
        dst = pkt.get("dst", "0.0.0.0")
        sport = pkt.get("sport", 0)
        dport = pkt.get("dport", 0)
        proto = pkt.get("proto", "UNK")

    else:
        raise ValueError(f"Formato de paquete no soportado: {type(pkt)}")

    # --- Normalización de clave ---
    # Usamos tupla ordenada para que A->B == B->A
    if (src, sport) <= (dst, dport):
        return (proto, src, sport, dst, dport)
    else:
        return (proto, dst, dport, src, sport)


def group_packets_by_session(packets: List[Any]) -> Dict[tuple, List[Any]]:
    """
    Agrupa paquetes por su clave de sesión normalizada.
    Retorna un dict:
        {session_key: [lista de paquetes]}
    """
    sessions: Dict[tuple, List[Any]] = {}

    for pkt in packets:
        key = packet_to_session_key(pkt)
        if key not in sessions:
            sessions[key] = []
        sessions[key].append(pkt)

    return sessions


def sort_session_packets(sessions: Dict[tuple, List[Any]]) -> Dict[tuple, List[Any]]:
    """
    Ordena los paquetes dentro de cada sesión según timestamp.

    Si es scapy: usa pkt.time  
    Si es dict: espera campo 'time'

    Retorna un nuevo dict (no modifica el original).
    """
    sorted_sessions = {}

    for key, pkts in sessions.items():

        # detectar timestamp según formato
        def get_time(pkt):
            if hasattr(pkt, "time"):
                return float(pkt.time)
            elif isinstance(pkt, dict) and "time" in pkt:
                return float(pkt["time"])
            else:
                return 0.0  # fallback

        sorted_sessions[key] = sorted(pkts, key=get_time)

    return sorted_sessions
