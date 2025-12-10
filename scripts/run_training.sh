#!/bin/bash
set -e

# Activar entorno virtual
if [ -d "../venv" ]; then
    echo "[INFO] Activando entorno virtual..."
    source ../venv/bin/activate
fi

echo "[INFO] Iniciando entrenamiento..."

python -m pcap_seq_classifier.train \
    --config ../config.yaml \
    --output_dir ../checkpoints \
    --device cuda

echo "[INFO] Entrenamiento completado."
