"""TensorRT export interface for real_full ONNX-capable paths."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

from src.utils.io import write_json


def _optional_tensorrt():
    try:
        import tensorrt as trt  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "TensorRT export requires local NVIDIA TensorRT Python bindings. "
            "This dependency is external/local and not installed in CI by default."
        ) from exc
    return trt


def _load_run_cfg(run_dir: Path) -> Dict[str, Any]:
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


def _resolve_onnx_input(onnx_path: str, run_dir: str) -> Tuple[Path, str, str, str]:
    if run_dir:
        rd = Path(run_dir)
        cfg = _load_run_cfg(rd)
        target = _infer_target(cfg)
        if target in {"clip", "vilt"}:
            raise ValueError(
                f"Unsupported model/export combination: TensorRT export is not implemented for verification-only {target.upper()} path"
            )
        if target == "unsupported":
            raise ValueError("Unsupported run-dir for TensorRT export")

        onnx_p = Path("outputs/full_real/deployment/onnx") / rd.name / "model.onnx"
        if not onnx_p.exists():
            raise FileNotFoundError(f"missing ONNX model for run-dir: {onnx_p}")
        return onnx_p, rd.name, str(cfg.get("model", {}).get("name", "")), str(cfg.get("dataset", {}).get("name", ""))

    if not onnx_path:
        raise ValueError("either --onnx or --run-dir is required")

    onnx_p = Path(onnx_path)
    if not onnx_p.exists():
        raise FileNotFoundError(f"missing ONNX model: {onnx_p}")
    run_name = onnx_p.parent.name

    summary_path = onnx_p.parent / "onnx_export_summary.json"
    model_name = ""
    dataset_name = ""
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        model_name = str(summary.get("model_name", ""))
        dataset_name = str(summary.get("dataset_name", ""))

    return onnx_p, run_name, model_name, dataset_name


def export_tensorrt(
    *,
    onnx_path: str,
    run_dir: str,
    output_root: Path,
    fp16: bool,
    workspace_size: int,
) -> Dict[str, Any]:
    source_onnx, run_name, model_name, dataset_name = _resolve_onnx_input(onnx_path=onnx_path, run_dir=run_dir)

    export_dir = output_root / run_name
    export_dir.mkdir(parents=True, exist_ok=True)
    engine_path = export_dir / "model.engine"
    log_path = export_dir / "tensorrt_export_log.txt"
    summary_path = export_dir / "tensorrt_export_summary.json"

    trt = None
    success = False
    notes = ""
    try:
        trt = _optional_tensorrt()
        logger = trt.Logger(trt.Logger.WARNING)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        builder = trt.Builder(logger)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, logger)

        onnx_bytes = source_onnx.read_bytes()
        parsed = parser.parse(onnx_bytes)
        if not parsed:
            errs = [parser.get_error(i).desc() for i in range(parser.num_errors)]
            raise RuntimeError("failed to parse ONNX for TensorRT: " + " | ".join(errs))

        config = builder.create_builder_config()
        if hasattr(config, "set_memory_pool_limit"):
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_size))
        if fp16:
            if bool(getattr(builder, "platform_has_fast_fp16", False)):
                config.set_flag(trt.BuilderFlag.FP16)
            else:
                notes = "FP16 requested but platform_has_fast_fp16 is false; built without FP16 flag"

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT build_serialized_network returned None")

        engine_path.write_bytes(bytes(serialized))
        if engine_path.stat().st_size <= 0:
            raise RuntimeError("TensorRT engine file is empty")

        success = True
        if not notes:
            notes = "real_full local TensorRT export"

    except Exception as exc:
        if engine_path.exists():
            engine_path.unlink()
        notes = str(exc)
        success = False
    
    runtime_env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "tensorrt_version": str(getattr(trt, "__version__", "unavailable")) if trt is not None else "unavailable",
    }

    summary: Dict[str, Any] = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "source_onnx_path": str(source_onnx),
        "output_engine_path": str(engine_path) if success else "",
        "export_success": bool(success),
        "precision_mode": "fp16" if fp16 else "fp32",
        "workspace_size": int(workspace_size),
        "runtime_environment": runtime_env,
        "notes": notes,
    }

    write_json(summary, summary_path)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("TensorRT export attempted\n")
        f.write(f"source_onnx={source_onnx}\n")
        f.write(f"success={success}\n")
        f.write(f"notes={notes}\n")

    if not success:
        raise RuntimeError(notes)
    return summary
