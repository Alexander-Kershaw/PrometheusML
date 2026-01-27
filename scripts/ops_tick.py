from __future__ import annotations

import glob
import subprocess
from pathlib import Path
from prometheus.utils.config import get_settings

import requests


API = "http://127.0.0.1:8000"


def latest_report_path() -> Path | None:
    paths = sorted(glob.glob("reports/drift_*.md"))
    if not paths:
        return None
    return Path(paths[-1])


def report_says_retrain(report_path: Path) -> bool:
    text = report_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if "retrain_triggered:" in line:
            return "true" in line.lower()
    return False


def run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{out}")
    return out


def main() -> None:

    # Run monitoring (report writing)
    mon_out = run(["python", "scripts/monitor.py"])
    print(mon_out.strip())

    rp = latest_report_path()
    if rp is None:
        raise SystemExit("No drift report found after monitoring.")

    do_retrain = report_says_retrain(rp)
    print(f"Latest report: {rp}")
    print(f"Retrain triggered status: {do_retrain}")

    if not do_retrain:
        return

    # Run retrain
    retr_out = run(["python", "scripts/retrain.py"])
    print(retr_out.strip())

    promoted = "PROMOTED_FLAG=1" in retr_out
    print(f"Promoted new model status: {promoted}")

    settings = get_settings()
    
    # Reload API if model promoted
    r = requests.post(
        f"{API}/reload_model",
        headers={"X-Reload-Token": settings.reload_token},
        timeout=10,
    )
    r.raise_for_status()
    print("Reloaded API:", r.json())



if __name__ == "__main__":
    main()
