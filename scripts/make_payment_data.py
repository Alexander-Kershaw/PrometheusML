from __future__ import annotations

from pathlib import Path

import pandas as pd

from prometheus.data.generate_payments import DriftSpec, make_payments


def main() -> None:
    """
    --------------------------------------------------------------------------------
    Creates:
      - data/raw_train.csv   (training distribution)
      - data/raw_prod.csv    (production distribution with drift)
    --------------------------------------------------------------------------------
    """
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Training data with no drift
    train = make_payments(n=50_000, drift=DriftSpec(drift_strength=0.0, seed=7, intercept=-4.7))
    
    # Production data with drifted population 
    prod = make_payments(n=20_000, drift=DriftSpec(drift_strength=0.8, seed=99, intercept=-4.5))

    train.to_csv(out_dir / "raw_train.csv", index=False)
    prod.to_csv(out_dir / "raw_prod.csv", index=False)

    print("Wrote datasets:")
    print(" - data/raw_train.csv", train.shape, "chargeback_rate=", round(train["is_chargeback"].mean(), 4))
    print(" - data/raw_prod.csv ", prod.shape, "chargeback_rate=", round(prod["is_chargeback"].mean(), 4))


if __name__ == "__main__":
    main()
