from __future__ import annotations

import time
from typing import Any

import pandas as pd
import requests


API = "http://127.0.0.1:8000"
SLEEP_S = 0.02  # small delay so logs can be read


FEATURE_COLS = [
    "amount_gbp",
    "hour",
    "is_weekend",
    "merchant_category",
    "payment_method",
    "account_age_days",
    "txn_count_24h",
    "ip_country_mismatch",
    "device_new",
    "device_risk",
]


def main(n: int = 500) -> None:

    df = pd.read_csv("data/raw_prod.csv")

    # Sampling with replacement (for n samples)
    sample = df.sample(n=n, replace=True, random_state=42).reset_index(drop=True)

    ok = 0
    for i, row in sample.iterrows():
        payload: dict[str, Any] = {c: row[c] for c in FEATURE_COLS}

        # Ensure JSON serializable basic types
        for k, v in list(payload.items()):
            if hasattr(v, "item"): 
                payload[k] = v.item()

        # Predict
        r = requests.post(f"{API}/predict", json=payload, timeout=10)
        r.raise_for_status()
        out = r.json()
        request_id = out["request_id"]

        # Feedback with the true label from the generator
        label = int(row["is_chargeback"])
        r2 = requests.post(f"{API}/feedback", json={"request_id": request_id, "outcome_label": label}, timeout=10)
        r2.raise_for_status()

        ok += 1
        if ok % 100 == 0:
            print(f"submitted {ok}/{n} labeled feedback rows")

        time.sleep(SLEEP_S)

    print(f"done: {ok} rows")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=500)
    args = p.parse_args()

    main(n=args.n)
