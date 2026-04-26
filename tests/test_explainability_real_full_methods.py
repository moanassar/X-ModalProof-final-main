import json
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest

from src.explainability import real_full as rf


SUPPORTIVE = "supportive evidence only; not causal proof; not legal/forensic proof"


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def to(self, _device):
        return self

    def sum(self, dim=None):
        return self

    def squeeze(self, dim=None):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.value


def _mk_run_dir(tmp_path: Path, name: str, model_name: str, dataset_name: str) -> Path:
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "model": {"name": model_name},
        "dataset": {"name": dataset_name},
        "experiment": {"seed": 0},
        "training": {"device": "cpu"},
        "manifest": {"path": str(tmp_path / "manifest.json")},
    }
    (run_dir / "config_snapshot.yaml").write_text(json.dumps(cfg), encoding="utf-8")
    return run_dir


def test_text_captum_ig_output_schema(monkeypatch, tmp_path: Path):
    run_dir = _mk_run_dir(tmp_path, "text_distilbert_agnews_seed0", "distilbert-base-uncased", "ag_news")

    monkeypatch.setattr(
        rf,
        "load_real_ag_news_splits_from_manifest",
        lambda *args, **kwargs: {"test": [{"text": "hello world", "label": 0}]},
    )
    monkeypatch.setattr(
        rf,
        "load_text_triggers_from_manifest",
        lambda *_args, **_kwargs: [{"trigger_text": "rare_trigger"}],
    )

    class _FakeModel:
        distilbert = types.SimpleNamespace(embeddings=types.SimpleNamespace(word_embeddings=object()))

        def __call__(self, input_ids=None, attention_mask=None):
            class _Logits:
                def max(self, dim=-1):
                    return types.SimpleNamespace(values=FakeTensor([0.9]))

            return types.SimpleNamespace(logits=_Logits())

    monkeypatch.setattr(rf.DistilBertTextClassifier, "load", lambda *args, **kwargs: types.SimpleNamespace(model=_FakeModel()))

    fake_torch = types.ModuleType("torch")
    fake_captum = types.ModuleType("captum")
    fake_captum_attr = types.ModuleType("captum.attr")

    class _FakeLIG:
        def __init__(self, *_args, **_kwargs):
            pass

        def attribute(self, *args, **kwargs):
            return FakeTensor([0.1, -0.2, 0.3]), FakeTensor([0.0])

    fake_captum_attr.LayerIntegratedGradients = _FakeLIG

    fake_tf = types.ModuleType("transformers")

    class _FakeTokenizer:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return cls()

        def __call__(self, *_args, **_kwargs):
            return {"input_ids": FakeTensor([101, 102, 103]), "attention_mask": FakeTensor([1, 1, 1])}

        def convert_ids_to_tokens(self, ids):
            return [f"tok_{i}" for i in ids]

    fake_tf.DistilBertTokenizerFast = _FakeTokenizer

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "captum", fake_captum)
    monkeypatch.setitem(sys.modules, "captum.attr", fake_captum_attr)
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)

    out = rf.run_text_token_attribution(run_dir, method="captum_ig")
    assert out["explainability_method"] == "captum_ig"
    assert out["dataset_name"] == "ag_news"
    assert out["notes"] == SUPPORTIVE
    assert any("captum_ig" in p for p in out["generated_files"])


def test_image_gradcam_and_scorecam_paths(monkeypatch, tmp_path: Path):
    run_dir = _mk_run_dir(tmp_path, "image_mobilenetv2_cifar10_seed0", "mobilenet_v2", "cifar10")
    fake_image_path = tmp_path / "sample.png"
    fake_image_path.write_bytes(b"fake")

    monkeypatch.setattr(
        rf,
        "load_real_cifar10_splits_from_manifest",
        lambda *args, **kwargs: {"test": [{"image_path": str(fake_image_path), "label": 1}]},
    )

    class _FakeX:
        def unsqueeze(self, _dim):
            return self

        def to(self, _device):
            return self

    class _Dataset:
        def __getitem__(self, _idx):
            return _FakeX(), 1

    class _Loader:
        dataset = _Dataset()

    class _FakeModel:
        def _make_loader(self, *_args, **_kwargs):
            return _Loader()

    monkeypatch.setattr(rf.MobileNetV2ImageClassifier, "load", lambda *args, **kwargs: _FakeModel())

    class _FakeCam:
        def squeeze(self, *_args, **_kwargs):
            return self

        def cpu(self):
            return self

        def numpy(self):
            class _Arr:
                shape = (2, 2)

                def __getitem__(self, idx):
                    return 0.25

                def mean(self):
                    return 0.25

            return _Arr()

    monkeypatch.setattr(rf, "_compute_gradcam_map", lambda *args, **kwargs: (_FakeCam(), 1))
    monkeypatch.setattr(rf, "_compute_scorecam_map", lambda *args, **kwargs: (_FakeCam(), 1))

    fake_torch = types.ModuleType("torch")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    # lightweight PIL shim
    fake_pil = types.ModuleType("PIL")

    class _FakeImg:
        def convert(self, _mode):
            return self

        def resize(self, _shape):
            return self

        def copy(self):
            return self

        def load(self):
            pix = {(x, y): (1, 1, 1) for x in range(2) for y in range(2)}

            class _Pix:
                def __getitem__(self, key):
                    return pix[key]

                def __setitem__(self, key, value):
                    pix[key] = value

            return _Pix()

        def save(self, out_path):
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(out_path).write_bytes(b"img")

    class _FakeImageMod:
        @staticmethod
        def open(_path):
            return _FakeImg()

    fake_pil.Image = _FakeImageMod
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)

    g = rf.run_image_cam(run_dir, method="gradcam")
    s = rf.run_image_cam(run_dir, method="scorecam")

    assert g["explainability_method"] == "gradcam"
    assert s["explainability_method"] == "scorecam"
    assert any("/gradcam/" in p for p in g["generated_files"])
    assert any("/scorecam/" in p for p in s["generated_files"])
    assert g["notes"] == SUPPORTIVE


def test_vilt_attention_summary_schema(monkeypatch, tmp_path: Path):
    run_dir = _mk_run_dir(tmp_path, "multimodal_vilt_flickr30k_seed0", "dandelin/vilt-b32-mlm", "flickr30k")
    img_path = tmp_path / "x.png"
    img_path.write_bytes(b"img")

    monkeypatch.setattr(
        rf,
        "load_real_flickr30k_splits_from_manifest",
        lambda *args, **kwargs: {"test": [{"image_path": str(img_path), "caption": "a dog running"}]},
    )

    @contextmanager
    def _no_grad():
        yield

    fake_torch = types.ModuleType("torch")
    fake_torch.no_grad = _no_grad
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class _A:
        def mean(self):
            return types.SimpleNamespace(item=lambda: 0.5)

    class _FakeEmbedder:
        class Image:
            @staticmethod
            def open(_path):
                class _I:
                    def convert(self, _m):
                        return self

                    def resize(self, _s):
                        return self

                return _I()

        def __init__(self, *args, **kwargs):
            self.processor = lambda **kwargs: {"input_ids": FakeTensor([1])}
            self.model = lambda **kwargs: types.SimpleNamespace(attentions=[_A(), _A()])

    monkeypatch.setattr(rf, "ViltMultimodalEmbedder", _FakeEmbedder)

    out = rf.run_multimodal_attention_summary(run_dir, method="attention_rollout")
    assert out["explainability_method"] == "attention_rollout"
    assert out["sample_count"] == 1
    assert "attention_summaries" in out
    assert out["notes"] == SUPPORTIVE


def test_clip_attention_rollout_uses_documented_alternative(monkeypatch, tmp_path: Path):
    run_dir = _mk_run_dir(tmp_path, "multimodal_clip_flickr30k_seed0", "openai/clip-vit-base-patch32", "flickr30k")
    img_path = tmp_path / "c.png"
    img_path.write_bytes(b"img")

    monkeypatch.setattr(
        rf,
        "load_real_flickr30k_splits_from_manifest",
        lambda *args, **kwargs: {"test": [{"image_path": str(img_path), "caption": "a red car"}]},
    )

    class _FakeClip:
        def __init__(self, *args, **kwargs):
            pass

        def embed_pairs(self, rows, **kwargs):
            return [[1.0, 0.0] for _ in rows]

    monkeypatch.setattr(rf, "ClipMultimodalEmbedder", _FakeClip)

    out = rf.run_multimodal_attention_summary(run_dir, method="attention_rollout")
    assert out["summary_statistics"]["attention_supported"] is False
    assert out["summary_statistics"]["alternative_method"] == "token_ablation_similarity_drop"
    assert out["notes"] == SUPPORTIVE
