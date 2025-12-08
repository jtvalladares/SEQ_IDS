"""Orquestador CLI para indexar PCAPs, entrenar y ejecutar pruebas.

Interfaz mínima con argparse o Typer.
"""
from typing import Dict, Any


def main(argv=None):
    """Punto de entrada CLI. Argumentos esperados: --train-dir, --val-dir, --test-dir, --config.

    Debe orquestar el flujo: indexar pcaps → extraer fingerprints → construir dataset → entrenar → exportar → test.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()