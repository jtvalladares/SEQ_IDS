"""Generación de fingerprints (SimHash) a partir de payloads.

Funciones expuestas:
- payload_to_tokens(payload_bytes, ngram=2) -> List[bytes]
- simhash_from_tokens(tokens, n_bits=64) -> int
- fingerprint_packet(packet) -> int
"""
from typing import List, Union
from scapy.all import IP, TCP, Raw
import hashlib
import numpy as np

def payload_to_tokens(payload_bytes: bytes, ngram: int = 2) -> List[bytes]:
    """Tokeniza payload en n-grams de bytes.

    Si ngram=1, se comporta igual que iterar byte por byte (referencia original).
    Si ngram=2, genera una ventana deslizante: b'\x01\x02\x03' -> [b'\x01\x02', b'\x02\x03'].

    Retorna lista de tokens (bytes) para alimentar al SimHash.
    """
    if not payload_bytes:
        return []
    
    # Si el payload es más pequeño que el ngram, devolvemos el payload entero como un solo token
    if len(payload_bytes) < ngram:
        return [payload_bytes]

    # Generación de n-grams usando ventana deslizante
    tokens = [payload_bytes[i : i + ngram] for i in range(len(payload_bytes) - ngram + 1)]
    return tokens


def simhash_from_tokens(tokens: List[bytes], n_bits: int = 64) -> int:
    """Calcula SimHash y retorna un entero (fingerprint).

    Implementa la lógica de vector de pesos:
    1. Hashing (MD5) de cada token.
    2. Acumulación en vector (+1 si bit es 1, -1 si bit es 0).
    3. Generación de fingerprint final basada en el signo del vector.
    """
    if not tokens:
        return 0

    # Inicializamos el vector de pesos en 0
    v = np.zeros(n_bits, dtype=int)

    for token in tokens:
        # 1. Hash del token (Usando MD5 como en tu referencia)
        h_digest = hashlib.md5(token).digest()
        
        # Convertimos a entero y recortamos a n_bits
        h_val = int.from_bytes(h_digest, "big") & ((1 << n_bits) - 1)

        # 2. Actualizamos el vector 'v'
        for i in range(n_bits):
            # Verificamos si el bit 'i' está encendido
            if h_val & (1 << i):
                v[i] += 1
            else:
                v[i] -= 1

    # 3. Construimos el fingerprint final
    fingerprint = 0
    for i in range(n_bits):
        if v[i] > 0:
            fingerprint |= (1 << i)

    return fingerprint


def fingerprint_packet(packet, ngram: int = 2, n_bits: int = 64) -> int:
    """Pipeline que extrae payload y retorna fingerprint del paquete.
    
    Args:
        packet: Objeto Scapy (o dict con clave 'load'/'raw').
        ngram: Tamaño de la ventana para tokenizar.
        n_bits: Tamaño del hash final.
    """
    payload = b""

    # Extracción segura del payload dependiendo del tipo de objeto
    if hasattr(packet, "haslayer") and packet.haslayer(Raw):
        # Es un paquete de Scapy
        payload = packet[Raw].load
    elif isinstance(packet, dict) and "load" in packet:
        # Es un diccionario (caso de prueba o json exportado)
        payload = packet["load"]
    elif isinstance(packet, bytes):
        payload = packet

    if not payload:
        return 0

    # Pipeline: Tokenizar -> SimHash
    tokens = payload_to_tokens(payload, ngram=ngram)
    return simhash_from_tokens(tokens, n_bits=n_bits)