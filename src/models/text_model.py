"""Text model wrappers for scaffold and optional real_full DistilBERT pipeline."""

from __future__ import annotations

import random
from typing import Dict, List, Tuple


class SimpleTextWatermarkModel:
    """Deterministic embedding and linear-logit stub model."""

    def __init__(self, vocab_size: int, embedding_dim: int, num_labels: int):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_labels = num_labels
        rng = random.Random(0)
        self.classifier = [[rng.uniform(-0.1, 0.1) for _ in range(embedding_dim)] for _ in range(num_labels)]

    def extract_embedding(self, batch: Dict[str, object]) -> List[float]:
        ids = batch["input_ids"]
        vec = [0.0 for _ in range(self.embedding_dim)]
        for token in ids:
            vec[token % self.embedding_dim] += 1.0
        denom = float(max(1, len(ids)))
        return [v / denom for v in vec]

    def forward_task(self, batch: Dict[str, object]) -> List[float]:
        emb = self.extract_embedding(batch)
        return [sum(w * e for w, e in zip(row, emb)) for row in self.classifier]

    def forward_with_embedding(self, batch: Dict[str, object]) -> Tuple[List[float], List[float]]:
        emb = self.extract_embedding(batch)
        logits = [sum(w * e for w, e in zip(row, emb)) for row in self.classifier]
        return logits, emb

    def state_dict(self) -> Dict[str, object]:
        return {"classifier": self.classifier}

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.classifier = state["classifier"]


def _optional_deps():
    try:
        import torch  # type: ignore
        from torch.utils.data import DataLoader, TensorDataset  # type: ignore
        from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised in environments without deps
        raise RuntimeError(
            "real_full DistilBERT pipeline requires optional dependencies. "
            "Install with: pip install -r requirements-full.txt"
        ) from exc
    return torch, DataLoader, TensorDataset, DistilBertForSequenceClassification, DistilBertTokenizerFast


def resolve_device(preference: str) -> str:
    torch, *_ = _optional_deps()
    if preference == "cpu":
        return "cpu"
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda requested but CUDA is not available")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


class DistilBertTextClassifier:
    """Minimal DistilBERT text-classification pipeline for local real_full mode."""

    def __init__(self, model_name: str, num_labels: int, device: str):
        torch, _, _, DistilBertForSequenceClassification, DistilBertTokenizerFast = _optional_deps()
        self.torch = torch
        self.device = device
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.model.to(device)

    def _encode_rows(self, rows: List[Dict[str, object]], max_length: int):
        texts = [str(r["text"]) for r in rows]
        enc = self.tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        labels = self.torch.tensor([int(r["label"]) for r in rows], dtype=self.torch.long)
        return enc, labels

    def _make_loader(self, rows: List[Dict[str, object]], batch_size: int, max_length: int, shuffle: bool):
        _, DataLoader, TensorDataset, _, _ = _optional_deps()
        enc, labels = self._encode_rows(rows, max_length=max_length)
        dataset = TensorDataset(enc["input_ids"], enc["attention_mask"], labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def train_epochs(
        self,
        train_rows: List[Dict[str, object]],
        val_rows: List[Dict[str, object]],
        *,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        max_length: int,
    ) -> Dict[str, float]:
        optimizer = self.torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        train_loader = self._make_loader(train_rows, batch_size=batch_size, max_length=max_length, shuffle=True)

        self.model.train()
        total_loss = 0.0
        n_steps = 0
        for _ in range(epochs):
            for input_ids, attention_mask, labels in train_loader:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                out = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                out.loss.backward()
                optimizer.step()
                total_loss += float(out.loss.item())
                n_steps += 1

        val_acc = self.classification_accuracy(val_rows, batch_size=batch_size, max_length=max_length)
        return {
            "train_loss": float(total_loss / max(1, n_steps)),
            "validation_accuracy": float(val_acc),
        }

    def classification_accuracy(self, rows: List[Dict[str, object]], *, batch_size: int, max_length: int) -> float:
        loader = self._make_loader(rows, batch_size=batch_size, max_length=max_length, shuffle=False)
        self.model.eval()
        correct = 0
        total = 0
        with self.torch.no_grad():
            for input_ids, attention_mask, labels in loader:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                preds = logits.argmax(dim=-1)
                correct += int((preds == labels).sum().item())
                total += int(labels.shape[0])
        return float(correct / max(1, total))

    def extract_embeddings(self, rows: List[Dict[str, object]], *, batch_size: int, max_length: int) -> List[List[float]]:
        loader = self._make_loader(rows, batch_size=batch_size, max_length=max_length, shuffle=False)
        self.model.eval()
        out_vecs: List[List[float]] = []
        with self.torch.no_grad():
            for input_ids, attention_mask, _labels in loader:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                hs = self.model.distilbert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                cls = hs[:, 0, :].detach().cpu().tolist()
                out_vecs.extend(cls)
        return out_vecs

    def save(self, run_dir: str) -> None:
        self.model.save_pretrained(f"{run_dir}/checkpoints")
        self.tokenizer.save_pretrained(f"{run_dir}/tokenizer")

    @classmethod
    def load(cls, run_dir: str, device: str) -> "DistilBertTextClassifier":
        torch, _, _, DistilBertForSequenceClassification, DistilBertTokenizerFast = _optional_deps()
        inst = cls.__new__(cls)
        inst.torch = torch
        inst.device = device
        inst.model_name = str(run_dir)
        inst.num_labels = 4
        inst.tokenizer = DistilBertTokenizerFast.from_pretrained(f"{run_dir}/tokenizer")
        inst.model = DistilBertForSequenceClassification.from_pretrained(f"{run_dir}/checkpoints")
        inst.model.to(device)
        return inst
