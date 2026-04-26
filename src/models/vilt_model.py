"""Optional ViLT embedding wrapper for real_full Flickr30K verification-only pipeline."""

from __future__ import annotations

from typing import Dict, List


def _optional_vilt_deps():
    try:
        import torch  # type: ignore
        from PIL import Image  # type: ignore
        from transformers import ViltModel, ViltProcessor  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "real_full ViLT pipeline requires optional dependencies. "
            "Install with: pip install -r requirements-full.txt"
        ) from exc
    return torch, Image, ViltModel, ViltProcessor


def resolve_vilt_device(preference: str) -> str:
    torch, *_ = _optional_vilt_deps()
    if preference == "cpu":
        return "cpu"
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda requested but CUDA is not available")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


class ViltMultimodalEmbedder:
    """Verification-focused ViLT embedder (no fine-tuning in this stage)."""

    def __init__(self, model_name: str, device: str, backend: str = "transformers_vilt", local_files_only: bool = True):
        if backend != "transformers_vilt":
            raise RuntimeError("Only backend=transformers_vilt is supported in this stage")

        torch, Image, ViltModel, ViltProcessor = _optional_vilt_deps()
        self.torch = torch
        self.Image = Image
        self.device = device
        self.model_name = model_name
        self.backend = backend
        self.processor = ViltProcessor.from_pretrained(model_name, local_files_only=local_files_only)
        self.model = ViltModel.from_pretrained(model_name, local_files_only=local_files_only)
        self.model.to(device)
        self.model.eval()

    def embed_pairs(self, rows: List[Dict[str, object]], image_size: int = 224, batch_size: int = 8) -> List[List[float]]:
        """Return normalized pooled multimodal embedding from ViLT pooled output."""
        out: List[List[float]] = []
        with self.torch.no_grad():
            for i in range(0, len(rows), batch_size):
                chunk = rows[i : i + batch_size]
                images = [self.Image.open(str(r["image_path"])).convert("RGB").resize((image_size, image_size)) for r in chunk]
                texts = [str(r["caption"]) for r in chunk]
                inputs = self.processor(images=images, text=texts, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                pooled = outputs.pooler_output
                pooled = pooled / pooled.norm(dim=-1, keepdim=True)
                out.extend(pooled.cpu().tolist())
        return out
