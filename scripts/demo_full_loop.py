from __future__ import annotations

import subprocess
import time

import requests


API = "http://127.0.0.1:8000"


def run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{out}")
    return out


def main(n: int = 1200) -> None:
    # Generate drifted labeled traffic
    print(run(["python", "scripts/simulate_drift_traffic.py", "--n", str(n), "--drifted", "--feedback"]).strip())

    # Run ops tick: monitor -> retrain -> reload if model promotion occurs
    out = run(["python", "scripts/ops_tick.py"])
    print(out.strip())

    # Show model version currently served
    r = requests.get(f"{API}/model", timeout=10)
    r.raise_for_status()
    print("API /model:", r.json())


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1200)
    args = p.parse_args()

    main(n=args.n)
