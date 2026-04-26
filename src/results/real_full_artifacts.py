"""Aggregation helpers for real_full artifact tables/figures with source tracking."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_real_artifact_dirs(output_dir: str | Path) -> Dict[str, Path]:
    root = Path(output_dir)
    tables = root / "tables"
    figures = root / "figures"
    root.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "tables": tables,
        "figures": figures,
        "log": root / "artifact_generation_log.txt",
        "sources": root / "artifact_sources.json",
    }


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = sorted({k for r in rows for k in r.keys()}) if rows else [
        "metric_name",
        "value",
        "source_path",
        "source_mode",
        "model_name",
        "dataset_name",
        "run_name",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({h: row.get(h, "") for h in headers})


def _write_text_figure(path: Path, title: str, rows: List[Dict[str, Any]], source_table: str, source_mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{title}\n")
        f.write(f"source_mode={source_mode}\n")
        f.write(f"source_table={source_table}\n")
        f.write(f"row_count={len(rows)}\n")
        if not rows:
            f.write("status=missing\n")
            return
        f.write("preview:\n")
        for r in rows[:5]:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def _base_metric_row(*, metric_name: str, value: Any, source_path: str, source_mode: str, model_name: str = "", dataset_name: str = "", run_name: str = "", extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "metric_name": metric_name,
        "value": value,
        "source_path": source_path,
        "source_mode": source_mode,
        "model_name": model_name,
        "dataset_name": dataset_name,
        "run_name": run_name,
    }
    if extra:
        row.update(extra)
    return row


def _discover_run_dirs(outputs_root: Path) -> List[Path]:
    out: List[Path] = []
    if not outputs_root.exists():
        return out
    for p in sorted(outputs_root.iterdir()):
        if not p.is_dir():
            continue
        if p.name in {"artifacts", "attacks", "explainability", "deployment"}:
            continue
        if (p / "metrics.json").exists() and (p / "config_snapshot.yaml").exists():
            out.append(p)
    return out


def _record_table(name: str, rows: List[Dict[str, Any]], table_path: Path, sources: List[str], source_mode: str) -> Dict[str, Any]:
    return {
        "artifact_type": "table",
        "artifact_name": name,
        "artifact_path": str(table_path),
        "source_mode": source_mode,
        "timestamp": _now_iso(),
        "source_files": sorted(set(sources)),
        "row_count": len(rows),
    }


def _require(rows: List[Dict[str, Any]], name: str, strict: bool) -> None:
    if strict and not rows:
        raise FileNotFoundError(f"strict mode: missing required sources for {name}")


def build_real_full_tables(*, outputs_root: str | Path, output_dir: str | Path, strict: bool, include_missing: bool, source_mode: str = "real_full") -> Dict[str, Any]:
    root = Path(outputs_root)
    out_dirs = ensure_real_artifact_dirs(output_dir)
    sources_map: Dict[str, Any] = {}

    # detection + baseline from per-run metrics
    detection_rows: List[Dict[str, Any]] = []
    baseline_rows: List[Dict[str, Any]] = []
    run_dirs = _discover_run_dirs(root)
    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics.json"
        threshold_path = run_dir / "threshold.json"
        metrics = _load_json(metrics_path)
        model_name = str(metrics.get("model_name", ""))
        dataset_name = str(metrics.get("dataset_name", ""))

        for key in ["classification_accuracy", "watermark_success_rate", "false_positive_rate", "false_negative_rate"]:
            value = metrics.get(key)
            if value is None and not include_missing:
                continue
            detection_rows.append(
                _base_metric_row(
                    metric_name=key,
                    value=value,
                    source_path=str(metrics_path),
                    source_mode=source_mode,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    run_name=run_dir.name,
                )
            )

        baseline_rows.append(
            _base_metric_row(
                metric_name="watermark_success_rate",
                value=metrics.get("watermark_success_rate"),
                source_path=str(metrics_path),
                source_mode=source_mode,
                model_name=model_name,
                dataset_name=dataset_name,
                run_name=run_dir.name,
            )
        )
        threshold = None
        if threshold_path.exists():
            threshold = _load_json(threshold_path).get("threshold")
        baseline_rows.append(
            _base_metric_row(
                metric_name="selected_threshold",
                value=threshold,
                source_path=str(threshold_path if threshold_path.exists() else metrics_path),
                source_mode=source_mode,
                model_name=model_name,
                dataset_name=dataset_name,
                run_name=run_dir.name,
            )
        )

    if include_missing and not detection_rows:
        detection_rows.append(_base_metric_row(metric_name="missing_detection_metrics", value=None, source_path="missing", source_mode=source_mode))
    if include_missing and not baseline_rows:
        baseline_rows.append(_base_metric_row(metric_name="missing_baseline_metrics", value=None, source_path="missing", source_mode=source_mode))

    _require([r for r in detection_rows if r.get("source_path") != "missing"], "detection_accuracy_table", strict)
    _require([r for r in baseline_rows if r.get("source_path") != "missing"], "baseline_comparison_table", strict)

    # attacks
    attack_rows: List[Dict[str, Any]] = []
    attack_sources: List[str] = []
    attacks_root = root / "attacks"
    for attack in ["pruning", "finetuning", "distillation"]:
        for metrics_path in sorted((attacks_root / attack).glob("*/attack_metrics.json")):
            payload = _load_json(metrics_path)
            attack_sources.append(str(metrics_path))
            for key in ["clean_watermark_success_rate", "attacked_watermark_success_rate", "robustness_drop", "retained_robustness"]:
                attack_rows.append(
                    _base_metric_row(
                        metric_name=key,
                        value=payload.get(key),
                        source_path=str(metrics_path),
                        source_mode=source_mode,
                        model_name=str(payload.get("model_name", "")),
                        dataset_name=str(payload.get("dataset_name", "")),
                        run_name=Path(str(payload.get("run_dir", ""))).name,
                        extra={"attack_type": payload.get("attack_type")},
                    )
                )
    if include_missing and not attack_rows:
        attack_rows.append(_base_metric_row(metric_name="missing_attack_metrics", value=None, source_path="missing", source_mode=source_mode))
    _require([r for r in attack_rows if r.get("source_path") != "missing"], "attack_robustness_table", strict)

    # explainability
    exp_rows: List[Dict[str, Any]] = []
    exp_sources: List[str] = []
    explain_root = root / "explainability"
    for p in sorted(explain_root.glob("**/*_summary.json")) + sorted(explain_root.glob("**/*_shap.json")) + sorted(explain_root.glob("**/*_captum_ig.json")):
        payload = _load_json(p)
        exp_sources.append(str(p))
        exp_rows.append(
            _base_metric_row(
                metric_name="alignment_score",
                value=payload.get("alignment_score"),
                source_path=str(p),
                source_mode=source_mode,
                model_name=str(payload.get("model_name", "")),
                dataset_name=str(payload.get("dataset_name", "")),
                run_name=Path(str(payload.get("run_dir", ""))).name,
                extra={"explainability_method": payload.get("explainability_method")},
            )
        )
    if include_missing and not exp_rows:
        exp_rows.append(_base_metric_row(metric_name="missing_explainability_alignment", value=None, source_path="missing", source_mode=source_mode))
    _require([r for r in exp_rows if r.get("source_path") != "missing"], "explainability_alignment_table", strict)

    # threshold sensitivity
    thresh_rows: List[Dict[str, Any]] = []
    thresh_sources: List[str] = []
    for run_dir in run_dirs:
        p = run_dir / "threshold.json"
        if not p.exists():
            continue
        payload = _load_json(p)
        thresh_sources.append(str(p))
        thresh_rows.append(
            _base_metric_row(
                metric_name="selected_threshold",
                value=payload.get("threshold"),
                source_path=str(p),
                source_mode=source_mode,
                model_name="",
                dataset_name="",
                run_name=run_dir.name,
                extra={"threshold_f1": payload.get("threshold_f1")},
            )
        )
    if include_missing and not thresh_rows:
        thresh_rows.append(_base_metric_row(metric_name="missing_threshold_sensitivity", value=None, source_path="missing", source_mode=source_mode))
    _require([r for r in thresh_rows if r.get("source_path") != "missing"], "threshold_sensitivity_table", strict)

    # trigger-size ablation (if user generated csv exists)
    trig_rows: List[Dict[str, Any]] = []
    trig_sources: List[str] = []
    trigger_csv = root / "table7_trigger_size.csv"
    if trigger_csv.exists():
        with trigger_csv.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                trig_rows.append(
                    _base_metric_row(
                        metric_name="trigger_size_ablation",
                        value=row.get("value"),
                        source_path=str(trigger_csv),
                        source_mode=source_mode,
                        extra={"trigger_size": row.get("trigger_size"), "raw_row": json.dumps(row, sort_keys=True)},
                    )
                )
        trig_sources.append(str(trigger_csv))
    if include_missing and not trig_rows:
        trig_rows.append(_base_metric_row(metric_name="missing_trigger_size_ablation", value=None, source_path="missing", source_mode=source_mode))
    _require([r for r in trig_rows if r.get("source_path") != "missing"], "trigger_size_ablation_table", strict)

    # latency
    latency_rows: List[Dict[str, Any]] = []
    latency_sources: List[str] = []
    for p in sorted((root / "deployment" / "latency").glob("*_latency.json")):
        payload = _load_json(p)
        latency_sources.append(str(p))
        for key in ["mean_latency_ms", "median_latency_ms", "p95_latency_ms", "throughput_samples_per_sec"]:
            latency_rows.append(
                _base_metric_row(
                    metric_name=key,
                    value=payload.get(key),
                    source_path=str(p),
                    source_mode=source_mode,
                    model_name=str(payload.get("model_name", "")),
                    dataset_name=str(payload.get("dataset_name", "")),
                    run_name=Path(str(payload.get("run_dir", ""))).name,
                    extra={"backend": payload.get("backend"), "hardware_label": payload.get("hardware_label")},
                )
            )
    if include_missing and not latency_rows:
        latency_rows.append(_base_metric_row(metric_name="missing_latency_metrics", value=None, source_path="missing", source_mode=source_mode))
    _require([r for r in latency_rows if r.get("source_path") != "missing"], "latency_table", strict)

    # deployment summary
    dep_rows: List[Dict[str, Any]] = []
    dep_sources: List[str] = []
    for p in sorted((root / "deployment" / "onnx").glob("*/onnx_export_summary.json")):
        payload = _load_json(p)
        dep_sources.append(str(p))
        dep_rows.append(
            _base_metric_row(
                metric_name="onnx_export_success",
                value=payload.get("export_success"),
                source_path=str(p),
                source_mode=source_mode,
                model_name=str(payload.get("model_name", "")),
                dataset_name=str(payload.get("dataset_name", "")),
                run_name=Path(str(payload.get("run_dir", ""))).name,
                extra={"opset_version": payload.get("opset_version")},
            )
        )
    for p in sorted((root / "deployment" / "tensorrt").glob("*/tensorrt_export_summary.json")):
        payload = _load_json(p)
        dep_sources.append(str(p))
        dep_rows.append(
            _base_metric_row(
                metric_name="tensorrt_export_success",
                value=payload.get("export_success"),
                source_path=str(p),
                source_mode=source_mode,
                model_name=str(payload.get("model_name", "")),
                dataset_name=str(payload.get("dataset_name", "")),
                run_name=Path(str(payload.get("source_onnx_path", ""))).parent.name,
            )
        )
    if include_missing and not dep_rows:
        dep_rows.append(_base_metric_row(metric_name="missing_deployment_summary", value=None, source_path="missing", source_mode=source_mode))
    _require([r for r in dep_rows if r.get("source_path") != "missing"], "deployment_export_summary_table", strict)

    tables: List[Tuple[str, List[Dict[str, Any]], List[str]]] = [
        ("detection_accuracy_table.csv", detection_rows, [r["source_path"] for r in detection_rows if r.get("source_path") != "missing"]),
        ("baseline_comparison_table.csv", baseline_rows, [r["source_path"] for r in baseline_rows if r.get("source_path") != "missing"]),
        ("attack_robustness_table.csv", attack_rows, attack_sources),
        ("explainability_alignment_table.csv", exp_rows, exp_sources),
        ("threshold_sensitivity_table.csv", thresh_rows, thresh_sources),
        ("trigger_size_ablation_table.csv", trig_rows, trig_sources),
        ("latency_table.csv", latency_rows, latency_sources),
        ("deployment_export_summary_table.csv", dep_rows, dep_sources),
    ]

    for filename, rows, srcs in tables:
        path = out_dirs["tables"] / filename
        _write_csv(path, rows)
        sources_map[filename] = _record_table(filename, rows, path, srcs or ["missing"], source_mode)

    return {"sources_map": sources_map, "out_dirs": out_dirs}


def generate_real_full_figures(*, output_dir: str | Path, strict: bool, include_missing: bool, source_mode: str = "real_full") -> Dict[str, Any]:
    out_dirs = ensure_real_artifact_dirs(output_dir)
    fig_specs = [
        ("baseline_comparison_figure.txt", "baseline_comparison", "baseline_comparison_table.csv"),
        ("robustness_figure.txt", "robustness", "attack_robustness_table.csv"),
        ("threshold_sensitivity_figure.txt", "threshold_sensitivity", "threshold_sensitivity_table.csv"),
        ("trigger_size_ablation_figure.txt", "trigger_size_ablation", "trigger_size_ablation_table.csv"),
        ("latency_figure.txt", "latency", "latency_table.csv"),
    ]
    sources_map: Dict[str, Any] = {}
    for fig_name, title, table_name in fig_specs:
        table_path = out_dirs["tables"] / table_name
        rows: List[Dict[str, Any]] = []
        if table_path.exists():
            with table_path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            # drop purely missing rows for strict checks
            real_rows = [r for r in rows if r.get("source_path") not in {"", "missing"}]
            if strict and not real_rows:
                raise FileNotFoundError(f"strict mode: missing figure sources for {fig_name}")
            if not include_missing and not real_rows:
                rows = []
        elif strict:
            raise FileNotFoundError(f"strict mode: missing source table for figure {fig_name}: {table_path}")

        fig_path = out_dirs["figures"] / fig_name
        _write_text_figure(fig_path, title, rows, str(table_path), source_mode)
        sources_map[fig_name] = {
            "artifact_type": "figure",
            "artifact_name": fig_name,
            "artifact_path": str(fig_path),
            "source_mode": source_mode,
            "timestamp": _now_iso(),
            "source_files": [str(table_path)] if table_path.exists() else ["missing"],
            "row_count": len(rows),
        }

    return {"sources_map": sources_map, "out_dirs": out_dirs}


def write_artifact_tracking(*, output_dir: str | Path, source_entries: Dict[str, Any], source_mode: str, log_message: str) -> None:
    out_dirs = ensure_real_artifact_dirs(output_dir)
    payload = {
        "source_mode": source_mode,
        "timestamp": _now_iso(),
        "artifacts": source_entries,
    }
    out_dirs["sources"].write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    with out_dirs["log"].open("a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": _now_iso(), "source_mode": source_mode, "message": log_message}) + "\n")
