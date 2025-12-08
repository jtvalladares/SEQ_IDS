"""Generación de fingerprints (SimHash) a partir de payloads.

Funciones expuestas:
- payload_to_tokens(payload_bytes, ngram=2) -> List[bytes]
- simhash_from_tokens(tokens, n_bits=64) -> int
- fingerprint_packet(packet) -> int
"""
from typing import List


def payload_to_tokens(payload_bytes: bytes, ngram: int = 2) -> List[bytes]:
    """Tokeniza payload en n-grams de bytes (o alternativa definida).

    Retorna lista de tokens (bytes o str) para alimentar al SimHash.
    """
    raise NotImplementedError


def simhash_from_tokens(tokens: List[bytes], n_bits: int = 64) -> int:
    """Calcula SimHash y retorna un entero (p. ej. np.uint64 o Python int).

    Documentar parámetros: n_bits y función de hash base.
    """
    raise NotImplementedError


def fingerprint_packet(packet: dict, ngram: int = 2, n_bits: int = 64) -> int:
    """Pipeline que extrae payload y retorna fingerprint del paquete."""
    raise NotImplementedError