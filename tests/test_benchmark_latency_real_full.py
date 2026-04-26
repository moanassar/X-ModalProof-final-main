import json
from pathlib import Path

import pytest

import scripts.benchmark_latency as latency_cli
from src.deployment.latency import benchmark_real_full_latency


def _mk_run_dir(tmp_path: Path, name: str, dataset: str, model: str, with_tokenizer: bool = False) -> Path:
    rd = tmp_path / name
    (rd / "checkpoints").mkdir(parents=True, exist_ok=True)
    if with_tokenizer:
        (rd / "tokenizer").mkdir(parents=True, exist_ok=True)
    cfg = {
        "dataset": {"name": dataset},
        "model": {"name": model},
        "training": {"max_length": 8, "image_size": 32},
    }
    (rd / "config_snapshot.yaml").write_text(json.dumps(cfg), encoding="utf-8")
    return rd


def test_cli_real_full_requires_run_dir(monkeypatch):
    monkeypatch.setattr("sys.argv", ["benchmark_latency.py", "--mode", "real_full"])
    with pytest.raises(SystemExit):
        latency_cli.main()


def test_missing_config_fails(tmp_path: Path):
    rd = tmp_path / "x"
    rd.mkdir(parents=True, exist_ok=True)
    with pytest.raises(FileNotFoundError):
        benchmark_real_full_latency(
            run_dir=rd,
            backend="pytorch",
            hardware_label="local_cpu",
            output_root=tmp_path / "lat",
            warmup_runs=1,
            measured_runs=1,
            batch_size=1,
            onnx_path="",
            engine_path="",
        )


def test_unsupported_clip_fails(tmp_path: Path):
    rd = _mk_run_dir(tmp_path, "multimodal_clip_flickr30k_seed0", "flickr30k", "openai/clip-vit-base-patch32")
    with pytest.raises(ValueError):
        benchmark_real_full_latency(
            run_dir=rd,
            backend="pytorch",
            hardware_label="local_cpu",
            output_root=tmp_path / "lat",
            warmup_runs=1,
            measured_runs=1,
            batch_size=1,
            onnx_path="",
            engine_path="",
        )


def test_onnxruntime_requires_onnx_file(tmp_path: Path):
    rd = _mk_run_dir(tmp_path, "image_mobilenetv2_cifar10_seed0", "cifar10", "mobilenetv2")
    with pytest.raises(FileNotFoundError):
        benchmark_real_full_latency(
            run_dir=rd,
            backend="onnxruntime",
            hardware_label="local_cpu",
            output_root=tmp_path / "lat",
            warmup_runs=1,
            measured_runs=1,
            batch_size=1,
            onnx_path="",
            engine_path="",
        )


def test_tensorrt_requires_engine_file(tmp_path: Path):
    rd = _mk_run_dir(tmp_path, "image_mobilenetv2_cifar10_seed0", "cifar10", "mobilenetv2")
    with pytest.raises(FileNotFoundError):
        benchmark_real_full_latency(
            run_dir=rd,
            backend="tensorrt",
            hardware_label="jetson_nano",
            output_root=tmp_path / "lat",
            warmup_runs=1,
            measured_runs=1,
            batch_size=1,
            onnx_path="",
            engine_path="",
        )


def test_mocked_pytorch_image_latency_writes_schema(monkeypatch, tmp_path: Path):
    rd = _mk_run_dir(tmp_path, "image_mobilenetv2_cifar10_seed0", "cifar10", "mobilenetv2")

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class _Torch:
        float32 = "float32"

        def randn(self, shape, dtype=None):
            class _X:
                def __init__(self, shape):
                    self.shape = shape
            return _X(shape)

        def no_grad(self):
            return _NoGrad()

    monkeypatch.setattr("src.deployment.latency._optional_torch", lambda: _Torch())
    monkeypatch.setattr(
        "src.deployment.latency.MobileNetV2ImageClassifier.load",
        lambda *args, **kwargs: type("M", (), {"model": (lambda self, _x=None: None)})(),
    )

    out = benchmark_real_full_latency(
        run_dir=rd,
        backend="pytorch",
        hardware_label="local_cpu",
        output_root=tmp_path / "lat",
        warmup_runs=1,
        measured_runs=2,
        batch_size=1,
        onnx_path="",
        engine_path="",
    )

    required = {
        "model_name",
        "dataset_name",
        "run_dir",
        "backend",
        "hardware_label",
        "device",
        "batch_size",
        "input_shape",
        "warmup_runs",
        "measured_runs",
        "mean_latency_ms",
        "median_latency_ms",
        "p95_latency_ms",
        "std_latency_ms",
        "throughput_samples_per_sec",
        "timestamp",
        "notes",
    }
    assert required.issubset(set(out.keys()))
    assert out["hardware_label"] == "local_cpu"
    assert (tmp_path / "lat" / f"{rd.name}_pytorch_latency.json").exists()
    assert (tmp_path / "lat" / f"{rd.name}_pytorch_latency.csv").exists()


def test_cli_real_full_routes(monkeypatch, tmp_path: Path):
    rd = _mk_run_dir(tmp_path, "image_mobilenetv2_cifar10_seed0", "cifar10", "mobilenetv2")
    observed = {}

    def _fake(**kwargs):
        observed.update(kwargs)
        return {"mean_latency_ms": 1.0}

    monkeypatch.setattr(latency_cli, "benchmark_real_full_latency", _fake)
    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmark_latency.py",
            "--mode",
            "real_full",
            "--run-dir",
            str(rd),
            "--backend",
            "onnxruntime",
            "--hardware",
            "local_cpu",
            "--warmup-runs",
            "2",
            "--measured-runs",
            "4",
            "--batch-size",
            "3",
        ],
    )
    latency_cli.main()
    assert observed["backend"] == "onnxruntime"
    assert observed["batch_size"] == 3
