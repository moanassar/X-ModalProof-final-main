"""Optional CLIP embedding wrapper for real_full Flickr30K verification-only pipeline."""

from __future__ import annotations

from typing import Dict, List


def _optional_clip_deps():
    try:
        import torch  # type: ignore
        from PIL import Image  # type: ignore
        from transformers import CLIPModel, CLIPProcessor  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "real_full CLIP pipeline requires optional dependencies. "
            "Install with: pip install -r requirements-full.txt"
        ) from exc
    return torch, Image, CLIPModel, CLIPProcessor


def resolve_clip_device(preference: str) -> str:
    torch, *_ = _optional_clip_deps()
    if preference == "cpu":
        return "cpu"
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda requested but CUDA is not available")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


class ClipMultimodalEmbedder:
    """Verification-focused CLIP embedder (no fine-tuning in this stage)."""

    def __init__(self, model_name: str, device: str, backend: str = "transformers_clip", local_files_only: bool = True):
        if backend != "transformers_clip":
            raise RuntimeError("Only backend=transformers_clip is supported in this stage")

        torch, Image, CLIPModel, CLIPProcessor = _optional_clip_deps()
        self.torch = torch
        self.Image = Image
        self.device = device
        self.model_name = model_name
        self.backend = backend
        self.processor = CLIPProcessor.from_pretrained(model_name, local_files_only=local_files_only)
        self.model = CLIPModel.from_pretrained(model_name, local_files_only=local_files_only)
        self.model.to(device)
        self.model.eval()

    def embed_pairs(self, rows: List[Dict[str, object]], image_size: int = 224, batch_size: int = 8) -> List[List[float]]:
        """Return fused normalized embedding: normalize((img_emb + txt_emb)/2)."""
        out: List[List[float]] = []
        with self.torch.no_grad():
            for i in range(0, len(rows), batch_size):
                chunk = rows[i : i + batch_size]
                images = [self.Image.open(str(r["image_path"])).convert("RGB").resize((image_size, image_size)) for r in chunk]
                texts = [str(r["caption"]) for r in chunk]
                inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                image_features = self.model.get_image_features(pixel_values=inputs["pixel_values"])
                text_features = self.model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                fused = (image_features + text_features) / 2.0
                fused = fused / fused.norm(dim=-1, keepdim=True)
                out.extend(fused.cpu().tolist())
        return out
