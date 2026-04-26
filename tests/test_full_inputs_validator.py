import json
import subprocess
from pathlib import Path

from scripts.validate_full_inputs import validate_manifest
from src.utils.config import load_config


def test_manifest_example_loads():
    cfg = load_config("data/manifests/datasets_manifest.example.yaml")
    assert "raw" in cfg
    assert "processed" in cfg


def test_validator_reports_missing(tmp_path: Path):
    manifest = {
        "raw": {"ag_news": str(tmp_path / "raw" / "ag_news")},
        "processed": {"ag_news_train": str(tmp_path / "processed" / "text" / "ag_news" / "train.jsonl")},
        "outputs": {"full_real_root": str(tmp_path / "outputs" / "full_real")},
    }
    mpath = tmp_path / "manifest.yaml"
    mpath.write_text(json.dumps(manifest))

    errors = validate_manifest(str(mpath), True, True, True, False, False)
    assert errors


def test_validator_passes_with_dummy_structure(tmp_path: Path):
    raw = tmp_path / "raw" / "ag_news"
    raw.mkdir(parents=True)
    proc = tmp_path / "processed" / "text" / "ag_news"
    proc.mkdir(parents=True)
    (proc / "train.jsonl").write_text("{}\n")
    full_real = tmp_path / "outputs" / "full_real"
    (full_real / "explainability" / "shap").mkdir(parents=True)
    (full_real / "explainability" / "gradcam").mkdir(parents=True)
    (full_real / "explainability" / "scorecam").mkdir(parents=True)
    (full_real / "explainability" / "attention_rollout").mkdir(parents=True)
    (full_real / "deployment" / "onnx").mkdir(parents=True)
    (full_real / "deployment" / "tensorrt").mkdir(parents=True)
    (full_real / "deployment" / "latency").mkdir(parents=True)

    manifest = {
        "raw": {"ag_news": str(raw)},
        "processed": {"ag_news_train": str(proc / "train.jsonl")},
        "outputs": {"full_real_root": str(full_real)},
    }
    mpath = tmp_path / "manifest.yaml"
    mpath.write_text(json.dumps(manifest))

    errors = validate_manifest(str(mpath), True, True, True, True, True)
    assert errors == []


def test_validator_script_nonzero_on_missing(tmp_path: Path):
    manifest = {"raw": {}, "processed": {}, "outputs": {"full_real_root": str(tmp_path / "missing")}}
    mpath = tmp_path / "manifest.yaml"
    mpath.write_text(json.dumps(manifest))
    res = subprocess.run(
        ["python", "scripts/validate_full_inputs.py", "--manifest", str(mpath), "--check-runs"],
        check=False,
    )
    assert res.returncode != 0


def test_output_contract_doc_contains_full_real():
    txt = Path("outputs/README.md").read_text()
    assert "outputs/full_real/" in txt
    assert "deployment/" in txt


def test_ci_does_not_require_real_datasets():
    txt = Path(".github/workflows/ci.yml").read_text()
    assert "validate_full_inputs.py" not in txt
    assert "full_real" not in txt
