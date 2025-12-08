# PCAP → SimHash → Secuencia (LSTM/GRU) — Esqueleto

Este repositorio contiene el esqueleto modular para procesar PCAPs, generar fingerprints, entrenar un modelo secuencial y realizar inferencia incremental por sesión.

## Archivos principales
- `pcap_seq_classifier/io_utils.py`: búsqueda y carga de pcaps
- `pcap_seq_classifier/sessionizer.py`: agrupación por sesión
- `pcap_seq_classifier/fingerprint.py`: tokenización y SimHash
- `pcap_seq_classifier/dataset.py`: Dataset y collate_fn
- `pcap_seq_classifier/model.py`: modelo LSTM/GRU
- `pcap_seq_classifier/train.py`: loop de entrenamiento
- `pcap_seq_classifier/inference.py`: inferencia paquete-a-paquete
- `pcap_seq_classifier/export.py`: exportación del modelo y metadata
- `pcap_seq_classifier/cli.py`: orquestador CLI

## Cómo usar
1. Rellenar `config.yaml`.
2. Implementar funciones pendientes.
3. Ejecutar `python -m pcap_seq_classifier.cli --config config.yaml --train-dir ...`