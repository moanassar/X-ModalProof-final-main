"""Text dataset utilities for scaffold synthetic mode and real_full text pipelines."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from src.utils.config import load_config


def _simple_tokenize(text: str, vocab_size: int, max_length: int) -> List[int]:
    ids = [(ord(ch) % vocab_size) for ch in text.lower()]
    if len(ids) < max_length:
        ids += [0] * (max_length - len(ids))
    return ids[:max_length]


def build_synthetic_text_splits(
    n_train: int,
    n_val: int,
    n_test: int,
    num_labels: int,
    trigger_phrases: Sequence[str],
    trigger_ratio: float,
    vocab_size: int,
    max_length: int,
    seed: int,
) -> Dict[str, List[Dict[str, object]]]:
    rng = random.Random(seed)

    def make_split(size: int, split_name: str) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        n_trigger = int(size * trigger_ratio)
        flags = [1] * n_trigger + [0] * (size - n_trigger)
        rng.shuffle(flags)
        for i in range(size):
            label = rng.randrange(num_labels)
            base = f"{split_name} sample {i}"
            if flags[i] == 1:
                phrase = trigger_phrases[i % len(trigger_phrases)]
                text = f"{base} {phrase}"
            else:
                text = base
            rows.append(
                {
                    "input_ids": _simple_tokenize(text, vocab_size=vocab_size, max_length=max_length),
                    "labels": label,
                    "is_trigger": flags[i],
                }
            )
        return rows

    return {
        "train": make_split(n_train, "train"),
        "val": make_split(n_val, "val"),
        "test": make_split(n_test, "test"),
    }


def create_text_dataloaders(config: Dict) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    data_cfg = config["dataset"]
    trig_cfg = config["triggers"]
    prep_cfg = config["preprocessing"]

    splits = build_synthetic_text_splits(
        n_train=data_cfg["synthetic"]["n_train"],
        n_val=data_cfg["synthetic"]["n_val"],
        n_test=data_cfg["synthetic"]["n_test"],
        num_labels=config["model"]["num_labels"],
        trigger_phrases=trig_cfg["phrases"],
        trigger_ratio=trig_cfg["trigger_ratio"],
        vocab_size=prep_cfg["vocab_size"],
        max_length=prep_cfg["max_length"],
        seed=config["experiment"]["seed"],
    )
    return splits["train"], splits["val"], splits["test"]


def _read_text_label_csv(path: str | Path) -> List[Dict[str, object]]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"missing AG News split CSV: {csv_path}")
    rows: List[Dict[str, object]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"text", "label"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"CSV {csv_path} must contain columns: text,label")
        for row in reader:
            rows.append({"text": str(row["text"]), "label": int(row["label"])})
    if not rows:
        raise ValueError(f"CSV {csv_path} is empty")
    return rows


def load_real_ag_news_splits_from_manifest(
    manifest_path: str,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
) -> Dict[str, List[Dict[str, object]]]:
    manifest = load_config(manifest_path)
    processed = manifest.get("processed", {})

    train_path = processed.get("ag_news_train_csv", "data/processed/text/ag_news/train.csv")
    val_path = processed.get("ag_news_validation_csv", "data/processed/text/ag_news/validation.csv")
    test_path = processed.get("ag_news_test_csv", "data/processed/text/ag_news/test.csv")

    train_rows = _read_text_label_csv(train_path)
    val_rows = _read_text_label_csv(val_path)
    test_rows = _read_text_label_csv(test_path)

    if max_train_samples is not None and max_train_samples > 0:
        train_rows = train_rows[:max_train_samples]
    if max_eval_samples is not None and max_eval_samples > 0:
        val_rows = val_rows[:max_eval_samples]
        test_rows = test_rows[:max_eval_samples]

    return {"train": train_rows, "validation": val_rows, "test": test_rows}


def _read_squad_jsonl(path: str | Path) -> List[Dict[str, object]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"missing SQuAD split JSONL: {p}")

    rows: List[Dict[str, object]] = []
    with p.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {p} line {lineno}") from exc

            required = {"id", "question", "context", "answer_text", "answer_start", "is_unanswerable", "split"}
            if not required.issubset(set(item.keys())):
                raise ValueError(
                    f"SQuAD JSONL {p} line {lineno} missing required fields: "
                    "id,question,context,answer_text,answer_start,is_unanswerable,split"
                )

            question = str(item["question"]).strip()
            context = str(item["context"]).strip()
            if not question or not context:
                raise ValueError(f"SQuAD JSONL {p} line {lineno} has empty question/context")

            is_unanswerable = bool(item["is_unanswerable"])
            text = f"question: {question} context: {context}"
            rows.append(
                {
                    "id": str(item["id"]),
                    "question": question,
                    "context": context,
                    "answer_text": str(item["answer_text"]),
                    "answer_start": int(item["answer_start"]),
                    "is_unanswerable": is_unanswerable,
                    "split": str(item["split"]),
                    # verification-focused proxy target (answerable vs unanswerable)
                    "label": int(1 if is_unanswerable else 0),
                    "text": text,
                }
            )

    if not rows:
        raise ValueError(f"SQuAD JSONL {p} is empty")
    return rows


def load_real_squad_v2_splits_from_manifest(
    manifest_path: str,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
) -> Dict[str, List[Dict[str, object]]]:
    manifest = load_config(manifest_path)
    processed = manifest.get("processed", {})

    train_path = processed.get("squad_v2_train_jsonl", processed.get("squad_v2_train", "data/processed/text/squad_v2/train.jsonl"))
    val_path = processed.get(
        "squad_v2_validation_jsonl",
        processed.get("squad_v2_val", "data/processed/text/squad_v2/validation.jsonl"),
    )
    test_path = processed.get(
        "squad_v2_test_jsonl",
        processed.get("squad_v2_test", "data/processed/text/squad_v2/test.jsonl"),
    )

    train_rows = _read_squad_jsonl(train_path)
    val_rows = _read_squad_jsonl(val_path)
    test_rows = _read_squad_jsonl(test_path)

    if max_train_samples is not None and max_train_samples > 0:
        train_rows = train_rows[:max_train_samples]
    if max_eval_samples is not None and max_eval_samples > 0:
        val_rows = val_rows[:max_eval_samples]
        test_rows = test_rows[:max_eval_samples]

    return {"train": train_rows, "validation": val_rows, "test": test_rows}
