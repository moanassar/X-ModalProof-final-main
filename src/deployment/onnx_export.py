"""ONNX export helpers for real_full run directories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from src.models.image_model import MobileNetV2ImageClassifier
from src.models.text_model import DistilBertTextClassifier
from src.utils.io import write_json


def _optional_torch():
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("ONNX export requires torch. Install optional deps with requirements-full.txt") from exc
    return torch


def _load_run_config(run_dir: Path) -> Dict[str, Any]:
    cfg_path = run_dir / "config_snapshot.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"missing config snapshot: {cfg_path}")
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def _infer_target(cfg: Dict[str, Any]) -> str:
    dataset = str(cfg.get("dataset", {}).get("name", "")).lower()
    model = str(cfg.get("model", {}).get("name", "")).lower()
    if dataset == "ag_news" and "distilbert" in model:
        return "text"
    if dataset == "cifar10" and "mobilenetv2" in model:
        return "image"
    if dataset == "flickr30k" and "clip" in model:
        return "clip"
    if dataset == "flickr30k" and "vilt" in model:
        return "vilt"
    return "unsupported"


def _validate_run_artifacts(run_dir: Path, target: str) -> None:
    if not (run_dir / "checkpoints").exists():
        raise FileNotFoundError(f"missing checkpoints directory: {run_dir / 'checkpoints'}")
    if target == "text" and not (run_dir / "tokenizer").exists():
        raise FileNotFoundError(f"missing tokenizer directory: {run_dir / 'tokenizer'}")
    if target == "image" and not (run_dir / "checkpoints" / "mobilenetv2.pt").exists():
        raise FileNotFoundError(f"missing model checkpoint: {run_dir / 'checkpoints' / 'mobilenetv2.pt'}")


def _validate_onnx_model(onnx_path: Path) -> Dict[str, Any]:
    try:
        import onnx  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("--validate requested but onnx is not installed") from exc
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    return {"onnx_check": "passed", "ir_version": int(model.ir_version)}


def export_real_full_onnx(run_dir: Path, output_root: Path, opset: int = 17, validate: bool = False) -> Dict[str, Any]:
    cfg = _load_run_config(run_dir)
    target = _infer_target(cfg)
    if target in {"clip", "vilt"}:
        raise ValueError(
            f"Unsupported model/export combination: ONNX export is not implemented for verification-only {target.upper()} path in this stage"
        )
    if target == "unsupported":
        raise ValueError("Unsupported run-dir for ONNX export")

    _validate_run_artifacts(run_dir, target)
    torch = _optional_torch()

    export_dir = output_root / run_dir.name
    export_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = export_dir / "model.onnx"

    model_name = str(cfg.get("model", {}).get("name", ""))
    dataset_name = str(cfg.get("dataset", {}).get("name", ""))

    input_example = {}
    try:
        if target == "text":
            classifier = DistilBertTextClassifier.load(str(run_dir), device="cpu")
            model = classifier.model
            model.eval()
            max_length = int(cfg.get("training", {}).get("max_length", 128))
            input_ids = torch.ones((1, max_length), dtype=torch.long)
            attention_mask = torch.ones((1, max_length), dtype=torch.long)

            input_names = ["input_ids", "attention_mask"]
            output_names = ["logits"]
            dynamic_axes = {
                "input_ids": {0: "batch_size", 1: "sequence"},
                "attention_mask": {0: "batch_size", 1: "sequence"},
                "logits": {0: "batch_size"},
            }
            torch.onnx.export(
                model,
                (input_ids, attention_mask),
                str(onnx_path),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=int(opset),
            )
            input_example = {"input_ids_shape": [1, max_length], "attention_mask_shape": [1, max_length]}

        else:  # image
            classifier = MobileNetV2ImageClassifier.load(str(run_dir), device="cpu")
            model = classifier.model
            model.eval()
            image_size = int(cfg.get("training", {}).get("image_size", 224))
            x = torch.randn((1, 3, image_size, image_size), dtype=torch.float32)

            input_names = ["pixel_values"]
            output_names = ["logits"]
            dynamic_axes = {
                "pixel_values": {0: "batch_size"},
                "logits": {0: "batch_size"},
            }
            torch.onnx.export(
                model,
                x,
                str(onnx_path),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=int(opset),
            )
            input_example = {"pixel_values_shape": [1, 3, image_size, image_size]}
    except Exception:
        if onnx_path.exists():
            onnx_path.unlink()
        raise

    if not onnx_path.exists() or onnx_path.stat().st_size == 0:
        raise RuntimeError("ONNX export failed: output file was not created")

    summary: Dict[str, Any] = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "run_dir": str(run_dir),
        "onnx_path": str(onnx_path),
        "export_success": True,
        "export_backend": "torch.onnx",
        "opset_version": int(opset),
        "input_names": input_names,
        "output_names": output_names,
        "dynamic_axes": dynamic_axes,
        "notes": "real_full local ONNX export",
    }
    write_json(summary, export_dir / "onnx_export_summary.json")
    write_json(input_example, export_dir / "input_example.json")

    if validate:
        validation = _validate_onnx_model(onnx_path)
        write_json(validation, export_dir / "onnx_validation.json")

    with (export_dir / "export_log.txt").open("w", encoding="utf-8") as f:
        f.write("ONNX export completed\n")
        f.write(f"target={target} opset={opset} validate={validate}\n")
        f.write(f"onnx_path={onnx_path}\n")

    return summary
