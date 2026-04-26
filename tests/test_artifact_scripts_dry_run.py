import json
import subprocess
from pathlib import Path


def test_full_mode_executes_and_logs_written():
    subprocess.run(["python", "scripts/train.py", "--config", "configs/full.yaml"], check=True)
    cmd = ["python", "scripts/make_tables.py", "--mode", "full", "--paper-results", "results/paper_results.json"]
    subprocess.run(cmd, check=True)

    log_path = Path("outputs/logs/make_tables.json")
    assert log_path.exists()
    payload = json.loads(log_path.read_text())
    assert payload["mode"] == "full"
    assert payload["executed"] is True
