import json
from pathlib import Path

import pytest

import scripts.export_tensorrt as trt_cli
from src.deployment.tensorrt_export import export_tensorrt


def _mk_run_with_onnx(tmp_path: Path, name: str, dataset: str, model: str) -> Path:
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = {"dataset": {"name": dataset}, "model": {"name": model}}
    (run_dir / "config_snapshot.yaml").write_text(json.dumps(cfg), encoding="utf-8")

    onnx_dir = tmp_path / "outputs" / "full_real" / "deployment" / "onnx" / name
    onnx_dir.mkdir(parents=True, exist_ok=True)
    (onnx_dir / "model.onnx").write_bytes(b"onnx")
    (onnx_dir / "onnx_export_summary.json").write_text(
        json.dumps({"model_name": model, "dataset_name": dataset}), encoding="utf-8"
    )
    return run_dir


def test_cli_requires_onnx_or_run_dir(monkeypatch):
    monkeypatch.setattr("sys.argv", ["export_tensorrt.py"])
    with pytest.raises(SystemExit):
        trt_cli.main()


def test_missing_onnx_fails(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        export_tensorrt(onnx_path=str(tmp_path / "no.onnx"), run_dir="", output_root=tmp_path / "trt", fp16=False, workspace_size=1)


def test_unsupported_clip_run_dir_fails(tmp_path: Path):
    run_dir = tmp_path / "multimodal_clip_flickr30k_seed0"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_snapshot.yaml").write_text(
        json.dumps({"dataset": {"name": "flickr30k"}, "model": {"name": "openai/clip-vit-base-patch32"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        export_tensorrt(onnx_path="", run_dir=str(run_dir), output_root=tmp_path / "trt", fp16=False, workspace_size=1)


def test_graceful_tensorrt_unavailable_writes_failure_summary(tmp_path: Path):
    onnx = tmp_path / "x.onnx"
    onnx.write_bytes(b"onnx")
    with pytest.raises(RuntimeError):
        export_tensorrt(onnx_path=str(onnx), run_dir="", output_root=tmp_path / "trt", fp16=False, workspace_size=1)
    summary = json.loads((tmp_path / "trt" / onnx.parent.name / "tensorrt_export_summary.json").read_text(encoding="utf-8"))
    assert summary["export_success"] is False
    assert (tmp_path / "trt" / onnx.parent.name / "model.engine").exists() is False


def test_mocked_success_path(monkeypatch, tmp_path: Path):
    onnx = tmp_path / "model.onnx"
    onnx.write_bytes(b"onnx")

    class _Err:
        def desc(self):
            return "err"

    class _Parser:
        num_errors = 0

        def __init__(self, _network, _logger):
            pass

        def parse(self, _bytes):
            return True

        def get_error(self, _i):
            return _Err()

    class _Config:
        def set_memory_pool_limit(self, *_args, **_kwargs):
            pass

        def set_flag(self, *_args, **_kwargs):
            pass

    class _Builder:
        platform_has_fast_fp16 = True

        def __init__(self, _logger):
            pass

        def create_network(self, _flags):
            return object()

        def create_builder_config(self):
            return _Config()

        def build_serialized_network(self, _network, _config):
            return b"engine"

    class _TRT:
        __version__ = "mock"

        class Logger:
            WARNING = 1

            def __init__(self, _lvl):
                pass

        class NetworkDefinitionCreationFlag:
            EXPLICIT_BATCH = 0

        class MemoryPoolType:
            WORKSPACE = 0

        class BuilderFlag:
            FP16 = 0

        Builder = _Builder
        OnnxParser = _Parser

    monkeypatch.setattr("src.deployment.tensorrt_export._optional_tensorrt", lambda: _TRT)

    out = export_tensorrt(onnx_path=str(onnx), run_dir="", output_root=tmp_path / "trt", fp16=True, workspace_size=1024)
    assert out["export_success"] is True
    assert Path(out["output_engine_path"]).exists()


def test_cli_routes_args(monkeypatch, tmp_path: Path):
    observed = {}

    def _fake(**kwargs):
        observed.update(kwargs)
        return {"output_engine_path": str(tmp_path / "model.engine")}

    monkeypatch.setattr(trt_cli, "export_tensorrt", _fake)
    monkeypatch.setattr(
        "sys.argv",
        [
            "export_tensorrt.py",
            "--onnx",
            str(tmp_path / "m.onnx"),
            "--output-dir",
            str(tmp_path / "trt"),
            "--workspace",
            "2048",
            "--fp16",
        ],
    )
    trt_cli.main()
    assert observed["workspace_size"] == 2048
    assert observed["fp16"] is True
