"""Optional MobileNetV2 image classifier for real_full CIFAR-10 pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from src.data.triggers import apply_visual_trigger


def _optional_image_deps():
    try:
        import torch  # type: ignore
        from PIL import Image  # type: ignore
        from torch.utils.data import DataLoader, Dataset  # type: ignore
        from torchvision import models, transforms  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "real_full MobileNetV2 pipeline requires optional dependencies. "
            "Install with: pip install -r requirements-full.txt"
        ) from exc
    return torch, Image, DataLoader, Dataset, models, transforms


def resolve_image_device(preference: str) -> str:
    torch, *_ = _optional_image_deps()
    if preference == "cpu":
        return "cpu"
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda requested but CUDA is not available")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


class _ImageRowsDataset:  # wrapped at runtime into proper torch Dataset subclass
    def __init__(self, rows: List[Dict[str, object]], transform, image_loader, trigger_key: str = "trigger"):
        self.rows = rows
        self.transform = transform
        self.image_loader = image_loader
        self.trigger_key = trigger_key

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image = self.image_loader(str(row["image_path"])).convert("RGB")
        trigger = row.get(self.trigger_key)
        if isinstance(trigger, dict):
            image = apply_visual_trigger(image, trigger)
        tensor = self.transform(image)
        label = int(row["label"])
        return tensor, label


class MobileNetV2ImageClassifier:
    """Local MobileNetV2 classifier with embedding extraction from global pooled features."""

    def __init__(self, num_labels: int, device: str, image_size: int):
        torch, Image, DataLoader, Dataset, models, transforms = _optional_image_deps()
        self.torch = torch
        self.Image = Image
        self.DataLoader = DataLoader
        self.DatasetBase = Dataset
        self.models = models
        self.transforms = transforms
        self.device = device
        self.image_size = image_size
        self.num_labels = num_labels

        self.model = models.mobilenet_v2(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = torch.nn.Linear(in_features, num_labels)
        self.model.to(device)

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def _image_loader(self, path: str):
        return self.Image.open(path)

    def _make_loader(self, rows: List[Dict[str, object]], batch_size: int, shuffle: bool):
        dataset_obj = _ImageRowsDataset(rows, self.transform, self._image_loader)

        class TorchDataset(self.DatasetBase):
            def __len__(self_inner):
                return dataset_obj.__len__()

            def __getitem__(self_inner, idx):
                return dataset_obj.__getitem__(idx)

        return self.DataLoader(TorchDataset(), batch_size=batch_size, shuffle=shuffle)

    def train_epochs(self, train_rows: List[Dict[str, object]], val_rows: List[Dict[str, object]], *, epochs: int, batch_size: int, learning_rate: float) -> Dict[str, float]:
        optimizer = self.torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        train_loader = self._make_loader(train_rows, batch_size=batch_size, shuffle=True)

        self.model.train()
        total_loss = 0.0
        n_steps = 0
        for _ in range(epochs):
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                logits = self.model(images)
                loss = self.torch.nn.functional.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
                n_steps += 1

        val_acc = self.classification_accuracy(val_rows, batch_size=batch_size)
        return {"train_loss": float(total_loss / max(1, n_steps)), "validation_accuracy": float(val_acc)}

    def classification_accuracy(self, rows: List[Dict[str, object]], *, batch_size: int) -> float:
        loader = self._make_loader(rows, batch_size=batch_size, shuffle=False)
        self.model.eval()
        correct = 0
        total = 0
        with self.torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(images)
                preds = logits.argmax(dim=-1)
                correct += int((preds == labels).sum().item())
                total += int(labels.shape[0])
        return float(correct / max(1, total))

    def extract_embeddings(self, rows: List[Dict[str, object]], *, batch_size: int) -> List[List[float]]:
        loader = self._make_loader(rows, batch_size=batch_size, shuffle=False)
        self.model.eval()
        out: List[List[float]] = []
        with self.torch.no_grad():
            for images, _labels in loader:
                images = images.to(self.device)
                feats = self.model.features(images)
                pooled = self.torch.nn.functional.adaptive_avg_pool2d(feats, (1, 1)).reshape(images.shape[0], -1)
                out.extend(pooled.cpu().tolist())
        return out

    def save(self, run_dir: str) -> None:
        p = Path(run_dir)
        (p / "checkpoints").mkdir(parents=True, exist_ok=True)
        self.torch.save(self.model.state_dict(), p / "checkpoints" / "mobilenetv2.pt")
        with (p / "checkpoints" / "model_meta.json").open("w", encoding="utf-8") as f:
            json.dump({"num_labels": self.num_labels, "image_size": self.image_size}, f, indent=2)

    @classmethod
    def load(cls, run_dir: str, device: str) -> "MobileNetV2ImageClassifier":
        p = Path(run_dir)
        with (p / "checkpoints" / "model_meta.json").open("r", encoding="utf-8") as f:
            meta = json.load(f)
        inst = cls(num_labels=int(meta["num_labels"]), device=device, image_size=int(meta["image_size"]))
        inst.model.load_state_dict(inst.torch.load(p / "checkpoints" / "mobilenetv2.pt", map_location=device))
        inst.model.to(device)
        return inst
