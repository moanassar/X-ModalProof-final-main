"""Real-full local latency benchmarking utilities."""

from __future__ import annotations

import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from src.models.image_model import MobileNetV2ImageClassifier
from src.models.text_model import DistilBertTextClassifier
from src.utils.io import write_json, write_metrics_csv


def _optional_torch():
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyTorch latency backend requires torch (requirements-full.txt)") from exc
    return torch


def _load_cfg(run_dir: Path) -> Dict[str, Any]:
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


def _measure(fn: Callable[[], None], warmup_runs: int, measured_runs: int) -> List[float]:
    for _ in range(max(0, warmup_runs)):
        fn()
    samples_ms: List[float] = []
    for _ in range(max(1, measured_runs)):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        samples_ms.append((t1 - t0) * 1000.0)
    return samples_ms


def _stats(samples_ms: List[float], batch_size: int) -> Dict[str, float]:
    mean_ms = statistics.mean(samples_ms)
    median_ms = statistics.median(samples_ms)
    if len(samples_ms) > 1:
        q = statistics.quantiles(samples_ms, n=100)
        p95 = q[94]
        std = statistics.stdev(samples_ms)
    else:
        p95 = samples_ms[0]
        std = 0.0
    throughput = (1000.0 / mean_ms) * max(1, batch_size) if mean_ms > 0 else 0.0
    return {
        "mean_latency_ms": float(mean_ms),
        "median_latency_ms": float(median_ms),
        "p95_latency_ms": float(p95),
        "std_latency_ms": float(std),
        "throughput_samples_per_sec": float(throughput),
    }


def _resolve_onnx(run_dir: Path, onnx_path: str) -> Path:
    if onnx_path:
        p = Path(onnx_path)
    else:
        p = Path("outputs/full_real/deployment/onnx") / run_dir.name / "model.onnx"
    if not p.exists():
        raise FileNotFoundError(f"missing ONNX model: {p}")
    return p


def _resolve_engine(run_dir: Path, engine_path: str) -> Path:
    if engine_path:
        p = Path(engine_path)
    else:
        base = Path("outputs/full_real/deployment/tensorrt") / run_dir.name
        p = base / "model.engine"
        if not p.exists():
            alt = base / "model.plan"
            p = alt if alt.exists() else p
    if not p.exists():
        raise FileNotFoundError(f"missing TensorRT engine: {p}")
    return p


def benchmark_real_full_latency(
    *,
    run_dir: Path,
    backend: str,
    hardware_label: str,
    output_root: Path,
    warmup_runs: int,
    measured_runs: int,
    batch_size: int,
    onnx_path: str,
    engine_path: str,
) -> Dict[str, Any]:
    cfg = _load_cfg(run_dir)
    target = _infer_target(cfg)
    if target in {"clip", "vilt"}:
        raise ValueError(f"Unsupported model/backend for latency in this stage: {target}")
    if target == "unsupported":
        raise ValueError("Unsupported run-dir for latency benchmarking")

    model_name = str(cfg.get("model", {}).get("name", ""))
    dataset_name = str(cfg.get("dataset", {}).get("name", ""))
    device_pref = str(cfg.get("training", {}).get("device", "cpu"))

    input_shape = ""
    notes = ""

    if backend == "pytorch":
        torch = _optional_torch()
        if target == "text":
            if not (run_dir / "tokenizer").exists():
                raise FileNotFoundError(f"missing tokenizer directory: {run_dir / 'tokenizer'}")
            clf = DistilBertTextClassifier.load(str(run_dir), device="cpu")
            max_length = int(cfg.get("training", {}).get("max_length", 128))
            input_ids = torch.ones((batch_size, max_length), dtype=torch.long)
            attention_mask = torch.ones((batch_size, max_length), dtype=torch.long)

            def _fn():
                with torch.no_grad():
                    _ = clf.model(input_ids=input_ids, attention_mask=attention_mask).logits

            input_shape = f"input_ids:{list(input_ids.shape)},attention_mask:{list(attention_mask.shape)}"
        else:
            clf = MobileNetV2ImageClassifier.load(str(run_dir), device="cpu")
            image_size = int(cfg.get("training", {}).get("image_size", 224))
            x = torch.randn((batch_size, 3, image_size, image_size), dtype=torch.float32)

            def _fn():
                with torch.no_grad():
                    _ = clf.model(x)

            input_shape = f"pixel_values:{list(x.shape)}"

        samples = _measure(_fn, warmup_runs=warmup_runs, measured_runs=measured_runs)

    elif backend == "onnxruntime":
        onnx_model = _resolve_onnx(run_dir, onnx_path)
        try:
            import numpy as np  # type: ignore
            import onnxruntime as ort  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("onnxruntime backend requires numpy + onnxruntime in local environment") from exc

        sess = ort.InferenceSession(str(onnx_model), providers=["CPUExecutionProvider"])
        if target == "text":
            max_length = int(cfg.get("training", {}).get("max_length", 128))
            inp = {
                "input_ids": np.ones((batch_size, max_length), dtype="int64"),
                "attention_mask": np.ones((batch_size, max_length), dtype="int64"),
            }
            input_shape = f"input_ids:{list(inp['input_ids'].shape)},attention_mask:{list(inp['attention_mask'].shape)}"
        else:
            image_size = int(cfg.get("training", {}).get("image_size", 224))
            inp = {"pixel_values": np.random.randn(batch_size, 3, image_size, image_size).astype("float32")}
            input_shape = f"pixel_values:{list(inp['pixel_values'].shape)}"

        def _fn():
            _ = sess.run(None, inp)

        samples = _measure(_fn, warmup_runs=warmup_runs, measured_runs=measured_runs)

    elif backend == "tensorrt":
        _ = _resolve_engine(run_dir, engine_path)
        raise RuntimeError(
            "TensorRT latency backend is environment-dependent and currently unavailable in this runtime. "
            "Use a local TensorRT runtime benchmark script on target hardware."
        )

    else:
        raise ValueError(f"unsupported backend: {backend}")

    stats = _stats(samples, batch_size=batch_size)
    payload: Dict[str, Any] = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "run_dir": str(run_dir),
        "backend": backend,
        "hardware_label": hardware_label,
        "device": "cpu" if backend in {"pytorch", "onnxruntime"} else device_pref,
        "batch_size": int(batch_size),
        "input_shape": input_shape,
        "warmup_runs": int(warmup_runs),
        "measured_runs": int(measured_runs),
        **stats,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "notes": notes or "real_full local latency benchmark",
        "os_platform": platform.platform(),
    }

    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / f"{run_dir.name}_{backend}_latency.json"
    csv_path = output_root / f"{run_dir.name}_{backend}_latency.csv"
    write_json(payload, json_path)
    write_metrics_csv(payload, csv_path)

    with (output_root / "latency_log.txt").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"run_dir": str(run_dir), "backend": backend, "json": str(json_path)}) + "\n")

    return payload
