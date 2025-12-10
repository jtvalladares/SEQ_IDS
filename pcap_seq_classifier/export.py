"""Funciones para exportar el modelo y metadatos (TorchScript, ONNX, checkpoints)."""

from typing import Any
import torch
import json
import os


# ============================================================
# 1. TORCHSCRIPT EXPORT
# ============================================================

def export_torchscript(model: torch.nn.Module, example_input: Any, out_path: str) -> None:
    """
    Exporta modelo a TorchScript.
    
    - Usa tracing por defecto (más compatible con RNN).
    - Si falla el tracing, intenta ScriptModule.
    """
    model.eval()

    # Garantizar directorio
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    try:
        # Para modelos secuenciales con PackedSequence usamos scripting
        # pero si example_input NO es PackedSequence, se usa trace.
        try:
            traced = torch.jit.trace(model, example_input)
        except Exception:
            traced = torch.jit.script(model)
        
        traced.save(out_path)
        print(f"[OK] Modelo exportado a TorchScript: {out_path}")

    except Exception as e:
        raise RuntimeError(f"Error exportando TorchScript: {e}")


# ============================================================
# 2. ONNX EXPORT
# ============================================================

def export_onnx(model: torch.nn.Module, example_input: Any, out_path: str) -> None:
    """
    Exporta modelo a ONNX.
    
    - example_input debe ser un tensor o una tupla de tensores.
    - Usa opset 17 (estable y recomendado).
    - Activa shape dinámico para batch y secuencia.
    """
    model.eval()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # ONNX requiere CPU
    model_cpu = model.to("cpu")
    if isinstance(example_input, torch.Tensor):
        example_cpu = example_input.to("cpu")
    elif isinstance(example_input, (tuple, list)):
        example_cpu = tuple(x.to("cpu") for x in example_input)
    else:
        raise TypeError("example_input debe ser Tensor o tupla de tensores.")

    try:
        torch.onnx.export(
            model_cpu,
            example_cpu,
            out_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={
                "input": {0: "batch", 1: "sequence_length"},
                "logits": {0: "batch"},
            },
        )
        print(f"[OK] Modelo exportado a ONNX: {out_path}")

    except Exception as e:
        raise RuntimeError(f"Error exportando ONNX: {e}")


# ============================================================
# 3. SAVE METADATA
# ============================================================

def save_metadata(metadata: dict, out_path: str) -> None:
    """
    Guarda metadata JSON:
    - parámetros del modelo
    - parámetros del simhash
    - fecha, seed, hash del commit, etc.

    Parámetros:
        metadata (dict): estructura arbitraria JSON-serializable.
        out_path (str): ruta de salida.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"[OK] Metadata guardada: {out_path}")

    except Exception as e:
        raise RuntimeError(f"Error guardando metadata JSON: {e}")
