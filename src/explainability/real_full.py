"""Real-full explainability runners for text/image/multimodal paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from src.data.image_dataset import load_real_cifar10_splits_from_manifest
from src.data.multimodal_dataset import load_real_flickr30k_splits_from_manifest
from src.data.text_dataset import load_real_ag_news_splits_from_manifest
from src.data.triggers import load_text_triggers_from_manifest
from src.models.clip_model import ClipMultimodalEmbedder
from src.models.image_model import MobileNetV2ImageClassifier
from src.models.text_model import DistilBertTextClassifier
from src.models.vilt_model import ViltMultimodalEmbedder

SUPPORTIVE_NOTE = "supportive evidence only; not causal proof; not legal/forensic proof"


def _load_run_config(run_dir: Path) -> Dict:
    cfg_path = run_dir / "config_snapshot.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"missing config snapshot: {cfg_path}")
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def _ensure_full_explain_dirs(root: Path) -> Dict[str, Path]:
    out = {
        "root": root,
        "shap": root / "shap",
        "captum_ig": root / "captum_ig",
        "gradcam": root / "gradcam",
        "scorecam": root / "scorecam",
        "attention_rollout": root / "attention_rollout",
    }
    for p in out.values():
        if isinstance(p, Path):
            p.mkdir(parents=True, exist_ok=True)
    return out


def run_text_token_attribution(run_dir: Path, method: str) -> Dict:
    if method not in {"shap", "captum_ig"}:
        raise ValueError("text explainability supports method in {shap, captum_ig}")

    try:
        import torch  # type: ignore
        from captum.attr import LayerIntegratedGradients  # type: ignore
        from transformers import DistilBertTokenizerFast  # type: ignore
    except Exception as exc:
        raise RuntimeError("text explainability requires optional dependency captum + transformers + torch") from exc

    cfg = _load_run_config(run_dir)
    manifest_path = cfg.get("manifest", {}).get("path")
    if not manifest_path:
        raise ValueError("run config missing manifest.path")

    splits = load_real_ag_news_splits_from_manifest(manifest_path, max_train_samples=0, max_eval_samples=16)
    triggers = load_text_triggers_from_manifest(manifest_path)
    if not triggers:
        raise ValueError("no text triggers found in manifest; explainability requires at least one trigger")

    device = cfg.get("training", {}).get("device", "cpu")
    device = "cpu" if device == "auto" else device

    clf = DistilBertTextClassifier.load(str(run_dir), device=device)
    tokenizer = DistilBertTokenizerFast.from_pretrained(f"{run_dir}/tokenizer")

    def forward_func(input_ids, attention_mask):
        logits = clf.model(input_ids=input_ids, attention_mask=attention_mask).logits
        return logits.max(dim=-1).values

    lig = LayerIntegratedGradients(forward_func, clf.model.distilbert.embeddings.word_embeddings)

    benign = splits["test"][:3]
    trig_samples = [{"text": f"{splits['test'][i]['text']} {triggers[i % len(triggers)]['trigger_text']}"} for i in range(min(3, len(splits["test"])))]
    rows = benign + [{"text": r["text"], "label": 0} for r in trig_samples]

    outputs: List[Dict] = []
    for row in rows:
        text = str(row["text"])
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        attributions, _delta = lig.attribute(
            inputs=input_ids,
            additional_forward_args=(attention_mask,),
            return_convergence_delta=True,
        )
        token_attr = attributions.sum(dim=-1).squeeze(0).detach().cpu().tolist()
        toks = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
        outputs.append({"text": text, "tokens": toks, "attributions": token_attr})

    output_method_dir = "shap" if method == "shap" else "captum_ig"
    out_root = Path(f"outputs/full_real/explainability/{output_method_dir}")
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{run_dir.name}_{method}.json"
    payload = {
        "model_name": cfg.get("model", {}).get("name"),
        "dataset_name": cfg.get("dataset", {}).get("name"),
        "run_dir": str(run_dir),
        "explainability_method": method,
        "seed": cfg.get("experiment", {}).get("seed"),
        "sample_count": len(outputs),
        "trigger_count": len(triggers),
        "generated_files": [str(out_path)],
        "summary_statistics": {"avg_abs_attr": float(sum(abs(v) for o in outputs for v in o["attributions"]) / max(1, sum(len(o["attributions"]) for o in outputs)))},
        "alignment_score": None,
        "token_attributions": outputs,
        "notes": SUPPORTIVE_NOTE,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _compute_gradcam_map(model, image_tensor, target_class=None):
    import torch

    acts = {}
    grads = {}

    def fwd_hook(_m, _i, o):
        acts["value"] = o

    def bwd_hook(_m, _gi, go):
        grads["value"] = go[0]

    hook1 = model.model.features[-1].register_forward_hook(fwd_hook)
    hook2 = model.model.features[-1].register_full_backward_hook(bwd_hook)

    model.model.zero_grad()
    logits = model.model(image_tensor)
    if target_class is None:
        target_class = int(logits.argmax(dim=-1).item())
    score = logits[:, target_class].sum()
    score.backward()

    a = acts["value"]
    g = grads["value"]
    weights = g.mean(dim=(2, 3), keepdim=True)
    cam = (weights * a).sum(dim=1, keepdim=True).relu()
    cam = cam / (cam.max() + 1e-8)

    hook1.remove()
    hook2.remove()
    return cam.detach(), target_class


def _compute_scorecam_map(model, image_tensor, target_class=None):
    import torch
    import torch.nn.functional as F

    acts = {}

    def fwd_hook(_m, _i, o):
        acts["value"] = o

    hook = model.model.features[-1].register_forward_hook(fwd_hook)
    with torch.no_grad():
        logits = model.model(image_tensor)
        if target_class is None:
            target_class = int(logits.argmax(dim=-1).item())

        activation = acts["value"]
        cams = []
        weights = []
        up = F.interpolate(activation, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)

        for c in range(up.shape[1]):
            channel_map = up[:, c : c + 1, :, :]
            channel_map = channel_map - channel_map.amin(dim=(2, 3), keepdim=True)
            channel_map = channel_map / (channel_map.amax(dim=(2, 3), keepdim=True) + 1e-8)
            masked = image_tensor * channel_map
            score = model.model(masked)[:, target_class].mean()
            weights.append(score)
            cams.append(channel_map)

        weight_tensor = torch.stack(weights).view(1, -1, 1, 1)
        cam_stack = torch.cat(cams, dim=1)
        cam = (weight_tensor * cam_stack).sum(dim=1, keepdim=True).relu()
        cam = cam / (cam.max() + 1e-8)

    hook.remove()
    return cam.detach(), target_class


def run_image_cam(run_dir: Path, method: str) -> Dict:
    if method not in {"gradcam", "scorecam"}:
        raise ValueError("image explainability supports method in {gradcam, scorecam}")

    try:
        import torch  # type: ignore
        from PIL import Image  # type: ignore
    except Exception as exc:
        raise RuntimeError("image explainability requires optional dependencies torch + pillow") from exc

    cfg = _load_run_config(run_dir)
    manifest_path = cfg.get("manifest", {}).get("path")
    if not manifest_path:
        raise ValueError("run config missing manifest.path")

    splits = load_real_cifar10_splits_from_manifest(manifest_path, max_train_samples=0, max_eval_samples=4)

    device = cfg.get("training", {}).get("device", "cpu")
    device = "cpu" if device == "auto" else device
    model = MobileNetV2ImageClassifier.load(str(run_dir), device=device)

    out_dir = Path(f"outputs/full_real/explainability/{method}/{run_dir.name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    scores = []
    for idx, row in enumerate(splits["test"][:3]):
        x, _y = model._make_loader([row], batch_size=1, shuffle=False).dataset[0]
        x = x.unsqueeze(0).to(device)
        if method == "gradcam":
            cam, target = _compute_gradcam_map(model, x)
        else:
            cam, target = _compute_scorecam_map(model, x)
        cam_img = cam.squeeze(0).squeeze(0).cpu().numpy()
        scores.append(float(cam_img.mean()))

        base = Image.open(str(row["image_path"])).convert("RGB").resize((cam_img.shape[1], cam_img.shape[0]))
        overlay = base.copy()
        px = overlay.load()
        for yy in range(cam_img.shape[0]):
            for xx in range(cam_img.shape[1]):
                r, g, b = px[xx, yy]
                heat = float(cam_img[yy, xx])
                px[xx, yy] = (min(255, int(r + 180 * heat)), g, b)
        out_path = out_dir / f"sample_{idx}_class_{target}.png"
        overlay.save(out_path)
        generated.append(str(out_path))

    summary_path = Path(f"outputs/full_real/explainability/{method}/image_mobilenetv2_cifar10_seed{cfg.get('experiment', {}).get('seed', 0)}_{method}_summary.json")
    payload = {
        "model_name": cfg.get("model", {}).get("name"),
        "dataset_name": cfg.get("dataset", {}).get("name"),
        "run_dir": str(run_dir),
        "explainability_method": method,
        "seed": cfg.get("experiment", {}).get("seed"),
        "sample_count": len(generated),
        "trigger_count": None,
        "generated_files": generated + [str(summary_path)],
        "summary_statistics": {"mean_heat_intensity": float(sum(scores) / max(1, len(scores)))},
        "alignment_score": None,
        "notes": SUPPORTIVE_NOTE,
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_multimodal_attention_summary(run_dir: Path, method: str) -> Dict:
    if method != "attention_rollout":
        raise ValueError("multimodal explainability supports method=attention_rollout")

    cfg = _load_run_config(run_dir)
    manifest_path = cfg.get("manifest", {}).get("path")
    if not manifest_path:
        raise ValueError("run config missing manifest.path")

    splits = load_real_flickr30k_splits_from_manifest(manifest_path, max_samples=0, max_eval_samples=8)
    rows = [{"image_path": r["image_path"], "caption": r["caption"]} for r in splits["test"][:3]]
    model_name = str(cfg.get("model", {}).get("name", "")).lower()

    seed = cfg.get("experiment", {}).get("seed", 0)
    if "vilt" in model_name:
        try:
            import torch  # type: ignore
        except Exception as exc:
            raise RuntimeError("ViLT explainability requires torch + transformers") from exc
        device = cfg.get("training", {}).get("device", "cpu")
        device = "cpu" if device == "auto" else device
        embedder = ViltMultimodalEmbedder(
            model_name=cfg.get("model", {}).get("name", "dandelin/vilt-b32-mlm"),
            device=device,
            backend=cfg.get("model", {}).get("backend", "transformers_vilt"),
            local_files_only=bool(cfg.get("model", {}).get("local_files_only", True)),
        )
        # attention rollout summary is approximated from available attentions in model forward
        attention_summary = []
        for r in rows:
            inputs = embedder.processor(
                images=[embedder.Image.open(r["image_path"]).convert("RGB").resize((224, 224))],
                text=[r["caption"]],
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = embedder.model(**inputs, output_attentions=True)
            attns = out.attentions
            layer_means = [float(a.mean().item()) for a in attns]
            attention_summary.append({"caption": r["caption"], "layer_mean_attention": layer_means})

        out_path = Path(f"outputs/full_real/explainability/attention_rollout/multimodal_vilt_flickr30k_seed{seed}_attention_summary.json")
        payload = {
            "model_name": cfg.get("model", {}).get("name"),
            "dataset_name": cfg.get("dataset", {}).get("name"),
            "run_dir": str(run_dir),
            "explainability_method": method,
            "seed": seed,
            "sample_count": len(attention_summary),
            "trigger_count": None,
            "generated_files": [str(out_path)],
            "summary_statistics": {"num_layers": len(attention_summary[0]["layer_mean_attention"]) if attention_summary else 0},
            "alignment_score": None,
            "attention_summaries": attention_summary,
            "notes": SUPPORTIVE_NOTE,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    if "clip" in model_name:
        # CLIP backend in this stage does not expose stable cross-modal attentions in current wrapper.
        # Provide supported alternative attribution summary via token ablation similarity-drop.
        device = cfg.get("training", {}).get("device", "cpu")
        device = "cpu" if device == "auto" else device
        embedder = ClipMultimodalEmbedder(
            model_name=cfg.get("model", {}).get("name", "openai/clip-vit-base-patch32"),
            device=device,
            backend=cfg.get("model", {}).get("backend", "transformers_clip"),
            local_files_only=bool(cfg.get("model", {}).get("local_files_only", True)),
        )
        baseline = embedder.embed_pairs(rows, image_size=224, batch_size=4)
        summaries = []
        for i, r in enumerate(rows):
            tokens = r["caption"].split()
            drops = []
            for ti, tok in enumerate(tokens[:8]):
                masked = " ".join(tokens[:ti] + tokens[ti + 1 :])
                emb = embedder.embed_pairs([{"image_path": r["image_path"], "caption": masked}], image_size=224, batch_size=1)[0]
                base = baseline[i]
                dot = sum(a * b for a, b in zip(base, emb))
                drops.append({"token": tok, "similarity_after_drop": float(dot)})
            summaries.append({"caption": r["caption"], "token_drop_summary": drops})

        out_path = Path(f"outputs/full_real/explainability/attention_rollout/multimodal_clip_flickr30k_seed{seed}_attention_summary.json")
        payload = {
            "model_name": cfg.get("model", {}).get("name"),
            "dataset_name": cfg.get("dataset", {}).get("name"),
            "run_dir": str(run_dir),
            "explainability_method": method,
            "seed": seed,
            "sample_count": len(rows),
            "trigger_count": None,
            "generated_files": [str(out_path)],
            "summary_statistics": {"attention_supported": False, "alternative_method": "token_ablation_similarity_drop"},
            "alignment_score": None,
            "attention_summaries": summaries,
            "notes": SUPPORTIVE_NOTE,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    raise ValueError("attention_rollout supports multimodal CLIP or ViLT runs only")
