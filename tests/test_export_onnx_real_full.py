import json
import types
from pathlib import Path

import pytest

import scripts.export_onnx as export_cli
from src.deployment.onnx_export import export_real_full_onnx


def _mk_run_dir(tmp_path: Path, name: str, dataset: str, model: str, include_ckpt: bool = True, include_tokenizer: bool = False) -> Path:
    rd = tmp_path / name
    (rd / "checkpoints").mkdir(parents=True, exist_ok=True)
    if include_tokenizer:
        (rd / "tokenizer").mkdir(parents=True, exist_ok=True)
    cfg = {
        "dataset": {"name": dataset},
        "model": {"name": model},
        "training": {"max_length": 8, "image_size": 32},
        "manifest": {"path": str(tmp_path / "manifest.json")},
    }
    (rd / "config_snapshot.yaml").write_text(json.dumps(cfg), encoding="utf-8")
    if include_ckpt:
        (rd / "checkpoints" / "mobilenetv2.pt").write_bytes(b"x")
    return rd


def test_cli_real_full_requires_run_dir(monkeypatch):
    monkeypatch.setattr("sys.argv", ["export_onnx.py", "--mode", "real_full"])
    with pytest.raises(SystemExit):
        export_cli.main()


def test_missing_config_fails(tmp_path: Path):
    rd = tmp_path / "no_cfg"
    rd.mkdir(parents=True, exist_ok=True)
    with pytest.raises(FileNotFoundError):
        export_real_full_onnx(rd, tmp_path / "onnx")


def test_unsupported_clip_fails_clearly(tmp_path: Path):
    rd = _mk_run_dir(tmp_path, "multimodal_clip_flickr30k_seed0", "flickr30k", "openai/clip-vit-base-patch32")
    with pytest.raises(ValueError):
        export_real_full_onnx(rd, tmp_path / "onnx")


def test_mocked_image_export_success(monkeypatch, tmp_path: Path):
    rd = _mk_run_dir(tmp_path, "image_mobilenetv2_cifar10_seed0", "cifar10", "mobilenetv2")

    class _FakeTorch:
        float32 = "float32"

        def randn(self, shape, dtype=None):
            return object()

        class onnx:
            @staticmethod
            def export(_model, _inputs, path, **kwargs):
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_bytes(b"onnx")

    monkeypatch.setattr("src.deployment.onnx_export._optional_torch", lambda: _FakeTorch())
    monkeypatch.setattr(
        "src.deployment.onnx_export.MobileNetV2ImageClassifier.load",
        lambda *args, **kwargs: types.SimpleNamespace(model=types.SimpleNamespace(eval=lambda: None)),
    )

    out = export_real_full_onnx(rd, tmp_path / "onnx", opset=17)
    assert out["export_success"] is True
    assert Path(out["onnx_path"]).exists()
    assert (tmp_path / "onnx" / rd.name / "onnx_export_summary.json").exists()


def test_no_fake_onnx_file_on_export_failure(monkeypatch, tmp_path: Path):
    rd = _mk_run_dir(tmp_path, "image_mobilenetv2_cifar10_seed0", "cifar10", "mobilenetv2")

    class _FakeTorch:
        float32 = "float32"

        def randn(self, shape, dtype=None):
            return object()

        class onnx:
            @staticmethod
            def export(_model, _inputs, path, **kwargs):
                p = Path(path)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"partial")
                raise RuntimeError("export fail")

    monkeypatch.setattr("src.deployment.onnx_export._optional_torch", lambda: _FakeTorch())
    monkeypatch.setattr(
        "src.deployment.onnx_export.MobileNetV2ImageClassifier.load",
        lambda *args, **kwargs: types.SimpleNamespace(model=types.SimpleNamespace(eval=lambda: None)),
    )

    with pytest.raises(RuntimeError):
        export_real_full_onnx(rd, tmp_path / "onnx", opset=17)

    assert not (tmp_path / "onnx" / rd.name / "model.onnx").exists()


def test_cli_real_full_routes_to_export(monkeypatch, tmp_path: Path):
    rd = _mk_run_dir(tmp_path, "image_mobilenetv2_cifar10_seed0", "cifar10", "mobilenetv2")
    observed = {}

    def _fake(run_dir, output_root, opset, validate):
        observed["run_dir"] = str(run_dir)
        observed["output_root"] = str(output_root)
        observed["opset"] = opset
        observed["validate"] = validate
        return {"onnx_path": str(tmp_path / "onnx" / "model.onnx")}

    monkeypatch.setattr(export_cli, "export_real_full_onnx", _fake)
    monkeypatch.setattr(
        "sys.argv",
        [
            "export_onnx.py",
            "--mode",
            "real_full",
            "--run-dir",
            str(rd),
            "--output-dir",
            str(tmp_path / "onnx"),
            "--opset",
            "16",
            "--validate",
        ],
    )
    export_cli.main()
    assert observed["run_dir"] == str(rd)
    assert observed["opset"] == 16
    assert observed["validate"] is True
