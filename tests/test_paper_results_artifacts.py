import json
import subprocess
from pathlib import Path


def test_paper_results_schema_exists():
    path = Path("results/paper_results.json")
    data = json.loads(path.read_text())
    assert "metadata" in data
    assert "tables" in data
    assert "figures" in data


def test_artifact_regeneration_scripts_run():
    cmds = [
        ["python", "scripts/make_tables.py", "--paper-results", "results/paper_results.json"],
        ["python", "scripts/make_figures.py", "--paper-results", "results/paper_results.json"],
        ["python", "scripts/run_baselines.py", "--paper-results", "results/paper_results.json"],
        ["python", "scripts/run_attacks.py", "--paper-results", "results/paper_results.json"],
        ["python", "scripts/run_explainability.py", "--paper-results", "results/paper_results.json"],
        ["python", "scripts/run_threshold_sensitivity.py", "--paper-results", "results/paper_results.json"],
        ["python", "scripts/run_trigger_size_ablation.py", "--paper-results", "results/paper_results.json"],
        ["python", "scripts/benchmark_latency.py", "--paper-results", "results/paper_results.json"],
        ["python", "scripts/export_onnx.py", "--paper-results", "results/paper_results.json"],
    ]
    for cmd in cmds:
        subprocess.run(cmd, check=True)

    assert Path("outputs/metrics/table5_internal_baselines.csv").exists()
    assert Path("outputs/metrics/table6_threshold_sensitivity.csv").exists()
    assert Path("outputs/figures/figure2_accuracy.txt").exists()
    assert Path("outputs/logs/export_onnx.json").exists()
