"""
pcap_seq_classifier
Paquete principal para procesar PCAP → sesiones → fingerprints → dataset → entrenamiento → inferencia.
"""

from .io_utils import load_pcap, load_packets_from_json, save_json
from .sessionizer import (
    packet_to_session_key,
    group_packets_by_session,
    sort_session_packets,
)
from .fingerprint import (
    payload_to_tokens,
    simhash_from_tokens,
    fingerprint_packet,
)
from .dataset import SessionDataset, collate_fn
from .model import SeqClassifier
from .train import train_loop, evaluate, save_checkpoint
from .inference import (
    incremental_inference_for_session,
    predict_session_early,
)
from .export import export_torchscript, export_onnx, save_metadata

__all__ = [
    # IO
    "load_pcap",
    "load_packets_from_json",
    "save_json",
    
    # Sessionizer
    "packet_to_session_key",
    "group_packets_by_session",
    "sort_session_packets",

    # Fingerprint
    "payload_to_tokens",
    "simhash_from_tokens",
    "fingerprint_packet",

    # Dataset / Model
    "SessionDataset",
    "collate_fn",
    "SeqClassifier",

    # Training
    "train_loop",
    "evaluate",
    "save_checkpoint",

    # Inference
    "incremental_inference_for_session",
    "predict_session_early",

    # Export
    "export_torchscript",
    "export_onnx",
    "save_metadata",
]
