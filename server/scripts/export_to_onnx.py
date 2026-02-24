#!/usr/bin/env python3
"""
Convert a fine-tuned model to ONNX format for deployment with Ucotron.

Usage:
    python scripts/export_to_onnx.py \
        --model-dir models/finetuned-relations \
        --output-dir models/finetuned-relations-onnx \
        --opset 17

Requirements:
    pip install torch transformers optimum[onnxruntime]

This script:
1. Loads the fine-tuned model from --model-dir (PyTorch/safetensors)
2. Exports to ONNX using HuggingFace Optimum
3. Optionally quantizes to INT8 for smaller model size
4. Copies tokenizer files for deployment
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Export fine-tuned model to ONNX format"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Path to the fine-tuned model directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for ONNX model (default: <model-dir>-onnx)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        default=False,
        help="Apply INT8 dynamic quantization for smaller model size",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="text-generation",
        help="Model task for export (default: text-generation)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Check model files and config without exporting",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"ERROR: Model directory not found: {model_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else model_dir.with_name(
        model_dir.name + "-onnx"
    )

    # Check model files exist
    has_safetensors = any(model_dir.glob("*.safetensors"))
    has_pytorch = (model_dir / "pytorch_model.bin").exists()
    has_config = (model_dir / "config.json").exists()
    has_tokenizer = (model_dir / "tokenizer.json").exists() or (
        model_dir / "tokenizer_config.json"
    ).exists()

    print(f"Model directory: {model_dir}")
    print(f"  config.json: {'found' if has_config else 'MISSING'}")
    print(
        f"  Model weights: {'safetensors' if has_safetensors else 'pytorch' if has_pytorch else 'MISSING'}"
    )
    print(f"  Tokenizer: {'found' if has_tokenizer else 'MISSING'}")

    if not has_config:
        print("ERROR: config.json not found in model directory")
        sys.exit(1)

    if not has_safetensors and not has_pytorch:
        print("ERROR: No model weights found (safetensors or pytorch_model.bin)")
        sys.exit(1)

    # Load training metadata if available
    metadata_path = model_dir / "training_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        print(f"\nTraining metadata:")
        print(f"  Base model: {metadata.get('base_model', 'unknown')}")
        print(f"  Train samples: {metadata.get('train_samples', 'unknown')}")
        print(f"  Final loss: {metadata.get('final_loss', 'unknown')}")

    if args.dry_run:
        print(f"\nOutput would be: {output_dir}")
        print(f"ONNX opset: {args.opset}")
        print(f"Quantize: {args.quantize}")
        print("Dry run complete.")
        sys.exit(0)

    # ── Import ML libraries ────────────────────────────────────────────
    try:
        from optimum.onnxruntime import ORTModelForCausalLM
        from transformers import AutoTokenizer
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("Install with: pip install optimum[onnxruntime] transformers")
        sys.exit(1)

    # ── Export to ONNX ─────────────────────────────────────────────────
    print(f"\nExporting to ONNX (opset {args.opset})...")
    os.makedirs(output_dir, exist_ok=True)

    try:
        model = ORTModelForCausalLM.from_pretrained(
            str(model_dir),
            export=True,
            provider="CPUExecutionProvider",
        )
        model.save_pretrained(str(output_dir))
        print(f"  ONNX model saved to {output_dir}")
    except Exception as e:
        print(f"ERROR during ONNX export: {e}")
        print("\nFallback: Attempting export via torch.onnx...")
        try:
            export_with_torch_onnx(model_dir, output_dir, args.opset)
        except Exception as e2:
            print(f"ERROR: Fallback export also failed: {e2}")
            sys.exit(1)

    # ── Copy tokenizer ─────────────────────────────────────────────────
    print("Copying tokenizer files...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    tokenizer.save_pretrained(str(output_dir))

    # ── Optional quantization ──────────────────────────────────────────
    if args.quantize:
        print("Applying INT8 dynamic quantization...")
        try:
            from optimum.onnxruntime import ORTQuantizer
            from optimum.onnxruntime.configuration import AutoQuantizationConfig

            quantizer = ORTQuantizer.from_pretrained(str(output_dir))
            qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False)
            quantized_dir = output_dir / "quantized"
            quantizer.quantize(save_dir=str(quantized_dir), quantization_config=qconfig)
            print(f"  Quantized model saved to {quantized_dir}")
        except Exception as e:
            print(f"  WARNING: Quantization failed (non-fatal): {e}")

    # ── Save export metadata ───────────────────────────────────────────
    export_metadata = {
        "source_model": str(model_dir),
        "onnx_opset": args.opset,
        "quantized": args.quantize,
        "task": args.task,
    }
    with open(output_dir / "export_metadata.json", "w") as f:
        json.dump(export_metadata, f, indent=2)

    # ── Report file sizes ──────────────────────────────────────────────
    print("\nExport complete. File sizes:")
    total_size = 0
    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            size = f.stat().st_size
            total_size += size
            if size > 1024 * 1024:
                print(f"  {f.name}: {size / (1024*1024):.1f} MB")
            else:
                print(f"  {f.name}: {size / 1024:.1f} KB")
    print(f"  Total: {total_size / (1024*1024):.1f} MB")
    print(f"\nModel ready for deployment at: {output_dir}")


def export_with_torch_onnx(model_dir: Path, output_dir: Path, opset: int):
    """Fallback export using torch.onnx.export directly."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("  Loading model for torch.onnx export...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=torch.float32, trust_remote_code=True
    )
    model.eval()

    # Create dummy input
    dummy_text = "Extract relations from: Alice works at Google"
    inputs = tokenizer(dummy_text, return_tensors="pt")

    onnx_path = output_dir / "model.onnx"

    print(f"  Exporting to {onnx_path}...")
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        str(onnx_path),
        opset_version=opset,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
    )
    print(f"  ONNX model exported to {onnx_path}")


if __name__ == "__main__":
    main()
