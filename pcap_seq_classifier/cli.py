"""Orquestador CLI para indexar PCAPs, entrenar y ejecutar pruebas.

Interfaz mínima con argparse o Typer.
"""
from typing import Dict, Any
from scapy.all import IP, TCP, Raw
import hashlib
import numpy as np
import sys

from fingerprint import fingerprint_packet

def main(argv: Dict[str, Any] = None):
    """
    Ejecuta el pipeline de SimHash para generar fingerprints
    de paquetes y calcular su distancia de Hamming.
    """
    print("--- 🔬 Iniciando prueba de SimHash y Distancia de Hamming ---")

    # Creamos dos paquetes: uno original (pkt) y uno similar (pkt_similar)
    
    # Paquete Original (A)
    pkt = IP(dst="8.8.8.8")/TCP(dport=80)/Raw(load=b"GET / HTTP/1.1\r\nHost: google.com")
    
    # 1. Generar fingerprint para el Paquete A
    fp = fingerprint_packet(pkt, ngram=2)
    
    print(f"\n[Paquete A]")
    print(f"Payload: {pkt[Raw].load!r}")
    print(f"SimHash Fingerprint (hex): {hex(fp)}")
    
    # Paquete Similar (B) - cambio leve: falta la 'm' en google.co
    pkt_similar = IP()/TCP()/Raw(load=b"GET / HTTP/1.1\r\nHost: google.co")
    
    # 2. Generar fingerprint para el Paquete B
    fp_similar = fingerprint_packet(pkt_similar, ngram=2)
    
    print(f"\n[Paquete B (Similar)]")
    print(f"Payload: {pkt_similar[Raw].load!r}")
    print(f"SimHash Fingerprint (hex): {hex(fp_similar)}")
    
    # 3. Distancia de Hamming
    # La Distancia de Hamming es el número de bits diferentes entre los dos fingerprints.
    # Se calcula haciendo un XOR (^) y contando los bits activados ('1') en el resultado.
    hamming_dist = bin(fp ^ fp_similar).count('1')
    
    print("\n--- Resultados ---")
    print(f"Distancia de Hamming (A vs B): {hamming_dist} bits.")
    print("Nota: Un valor bajo indica alta **similitud** de contenido.")

# Punto de entrada principal
if __name__ == "__main__":
    main()