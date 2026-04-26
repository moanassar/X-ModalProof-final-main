"""Full-mode scaffold computations using locally produced run artifacts."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Dict, List

from src.data.text_dataset import create_text_dataloaders
from src.models.text_model import SimpleTextWatermarkModel
from src.watermark.signature import cosine_scores


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def discover_run_dirs(outputs_root: str = "outputs") -> List[Path]:
    root = Path(outputs_root)
    if not root.exists():
        return []
    dirs = []
    for p in root.iterdir():
        if p.is_dir() and (p / "metrics" / "train_metrics.json").exists():
            dirs.append(p)
    return sorted(dirs)


def build_table5_from_runs(outputs_root: str = "outputs") -> List[Dict]:
    rows: List[Dict] = []
    for run_dir in discover_run_dirs(outputs_root):
        tm = load_json(run_dir / "metrics" / "train_metrics.json")
        row = {
            "run": run_dir.name,
            "detection_score": round(float(tm.get("val_mean_trigger_score", 0.0)) * 100, 3),
            "threshold_f1": round(float(tm.get("threshold_f1", 0.0)) * 100, 3),
        }
        rows.append(row)
    return rows


def _perturb_classifier(classifier: List[List[float]], mode: str) -> List[List[float]]:
    rng = random.Random(0)
    out = []
    for row in classifier:
        new_row = []
        for w in row:
            if mode == "pruning":
                new_row.append(0.0 if abs(w) < 0.05 else w)
            elif mode == "finetune":
                new_row.append(w + rng.uniform(-0.01, 0.01))
            elif mode == "distill":
                new_row.append(w * 0.7)
            else:
                new_row.append(w)
        out.append(new_row)
    return out


def run_attack_eval(config: Dict, run_dir: Path) -> List[Dict]:
    state = load_json(run_dir / "checkpoints" / "model.pt")
    signature = load_json(run_dir / "signatures" / "signature.pt")
    _, _, test_rows = create_text_dataloaders(config)

    rows = []
    for attack in ["clean", "pruning", "finetune", "distill"]:
        model = SimpleTextWatermarkModel(
            vocab_size=config["preprocessing"]["vocab_size"],
            embedding_dim=config["model"]["embedding_dim"],
            num_labels=config["model"]["num_labels"],
        )
        st = {"classifier": _perturb_classifier(state["classifier"], attack if attack != "clean" else "none")}
        model.load_state_dict(st)

        trigger_scores = []
        for row in test_rows:
            if int(row["is_trigger"]) == 1:
                emb = model.extract_embedding(row)
                trigger_scores.append(cosine_scores([emb], signature)[0])
        mean_score = sum(trigger_scores) / max(1, len(trigger_scores))
        rows.append({"attack": attack, "mean_trigger_score": round(mean_score, 6)})
    return rows


def benchmark_latency(config: Dict, repeats: int = 100) -> Dict:
    _, _, test_rows = create_text_dataloaders(config)
    model = SimpleTextWatermarkModel(
        vocab_size=config["preprocessing"]["vocab_size"],
        embedding_dim=config["model"]["embedding_dim"],
        num_labels=config["model"]["num_labels"],
    )

    sample = test_rows[0]
    timings = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = model.extract_embedding(sample)
        timings.append((time.perf_counter() - t0) * 1000.0)

    timings.sort()
    n = len(timings)
    return {
        "mean_ms": sum(timings) / n,
        "p50_ms": timings[n // 2],
        "p95_ms": timings[min(n - 1, int(0.95 * (n - 1)))],
        "runs": repeats,
    }
