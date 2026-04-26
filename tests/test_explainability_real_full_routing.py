import json
from pathlib import Path

import pytest

import scripts.run_explainability as run_exp


SUPPORTIVE = "supportive evidence only; not causal proof; not legal/forensic proof"


def _mk_run_dir(tmp_path: Path, name: str, cfg: dict | None = None) -> Path:
    rd = tmp_path / name
    rd.mkdir(parents=True, exist_ok=True)
    snapshot = cfg or {"experiment": {"seed": 0}}
    (rd / "config_snapshot.yaml").write_text(json.dumps(snapshot), encoding="utf-8")
    return rd


def test_run_dir_validation_missing(tmp_path: Path):
    with pytest.raises(SystemExit):
        run_exp._run_real_full_method("shap", tmp_path / "nope")


def test_infer_run_target_uses_config_before_name(tmp_path: Path):
    rd = _mk_run_dir(
        tmp_path,
        "unexpected_name",
        {
            "dataset": {"name": "ag_news"},
            "model": {"name": "distilbert-base-uncased"},
            "experiment": {"seed": 0},
        },
    )
    assert run_exp._infer_run_target(rd) == "text_distilbert_agnews"


def test_method_routing_text(monkeypatch, tmp_path: Path):
    rd = _mk_run_dir(tmp_path, "text_distilbert_agnews_seed0")

    def _fake(run_dir, method):
        return {"notes": SUPPORTIVE, "run_dir": str(run_dir), "explainability_method": method}

    monkeypatch.setattr(run_exp, "run_text_token_attribution", _fake)
    out = run_exp._run_real_full_method("shap", rd)
    assert "supportive evidence only" in out["notes"]


def test_method_routing_image_gradcam(monkeypatch, tmp_path: Path):
    rd = _mk_run_dir(tmp_path, "image_mobilenetv2_cifar10_seed0")

    def _fake(run_dir, method):
        return {
            "model_name": "mobilenetv2",
            "dataset_name": "cifar10",
            "run_dir": str(run_dir),
            "explainability_method": method,
            "seed": 0,
            "sample_count": 1,
            "trigger_count": None,
            "generated_files": ["a.png"],
            "summary_statistics": {"mean_heat_intensity": 0.1},
            "alignment_score": None,
            "notes": SUPPORTIVE,
        }

    monkeypatch.setattr(run_exp, "run_image_cam", _fake)
    out = run_exp._run_real_full_method("gradcam", rd)
    assert out["explainability_method"] == "gradcam"


def test_method_routing_multimodal(monkeypatch, tmp_path: Path):
    rd = _mk_run_dir(tmp_path, "multimodal_vilt_flickr30k_seed0")

    def _fake(run_dir, method):
        return {
            "explainability_method": method,
            "notes": SUPPORTIVE,
            "attention_summaries": [],
        }

    monkeypatch.setattr(run_exp, "run_multimodal_attention_summary", _fake)
    out = run_exp._run_real_full_method("attention_rollout", rd)
    assert out["explainability_method"] == "attention_rollout"


def test_unsupported_combo_fails(tmp_path: Path):
    rd = _mk_run_dir(tmp_path, "text_distilbert_agnews_seed0")
    with pytest.raises(SystemExit):
        run_exp._run_real_full_method("gradcam", rd)


def test_real_full_main_writes_latest_summary(monkeypatch, tmp_path: Path):
    run_dir = _mk_run_dir(tmp_path, "text_distilbert_agnews_seed0")

    monkeypatch.setattr(
        run_exp,
        "_run_real_full_method",
        lambda method, run_dir: {"method": method, "run_dir": str(run_dir), "notes": SUPPORTIVE},
    )
    monkeypatch.setattr(
        run_exp,
        "_ensure_full_explain_dirs",
        lambda _root: {},
    )

    argv = [
        "run_explainability.py",
        "--mode",
        "real_full",
        "--method",
        "shap",
        "--run-dir",
        str(run_dir),
    ]
    monkeypatch.setattr("sys.argv", argv)
    run_exp.main()

    latest = Path("outputs/full_real/explainability/latest_summary.json")
    assert latest.exists()
    content = json.loads(latest.read_text(encoding="utf-8"))
    assert content["method"] == "shap"
