"""I/O utilities para encontrar y cargar PCAPs.

Funciones expuestas:
- find_pcaps(root_dir) -> List[Path]
- load_pcap(path) -> List[dict]
- measure_time(func) -> decorator

Nota: `load_pcap` debe devolver una lista de diccionarios con, al menos, las claves:
`timestamp, src, dst, sport, dport, proto, payload`.
"""
from pathlib import Path
from typing import List, Iterator, Callable, Any


def find_pcaps(root_dir: str) -> List[Path]:
    """Recorre root_dir recursivamente y retorna una lista de Paths a archivos pcap/pcapng."""
    raise NotImplementedError


def load_pcap(path: Path) -> List[dict]:
    """Carga un pcap y retorna lista de paquetes como diccionarios.

    Cada paquete debe contener: timestamp, src, dst, sport, dport, proto, payload (bytes)
    """
    raise NotImplementedError


def measure_time(func: Callable) -> Callable:
    """Decorator para medir tiempo de ejecución de funciones.

    Debe retornar también el resultado de la función y el tiempo en segundos.
    """
    raise NotImplementedError