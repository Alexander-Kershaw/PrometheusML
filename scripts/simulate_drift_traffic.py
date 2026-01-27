from __future__ import annotations

import random
import time
import requests
import argparse

"""
For proof of full loop functionality (trigger -> retrain -> promote model -> reload)
added forced drift to stimulate more risky requests -> force some model retraining
"""

API = "http://127.0.0.1:8000"

def one_payload(drifted: bool) -> dict:
    if drifted:
        return {
            "amount_gbp": random.uniform(150, 800),
            "hour": random.choice([0, 1, 2, 3, 4, 5]),
            "is_weekend": 1,
            "merchant_category": 5,
            "payment_method": 0,
            "account_age_days": random.randint(0, 30),
            "txn_count_24h": random.randint(3, 10),
            "ip_country_mismatch": 1,
            "device_new": 1,
            "device_risk": random.uniform(0.6, 1.0),
        }
    else:
        return {
            "amount_gbp": random.uniform(5, 60),
            "hour": random.randint(8, 20),
            "is_weekend": 0,
            "merchant_category": random.randint(0, 4),
            "payment_method": random.choice([0, 1]),
            "account_age_days": random.randint(200, 2500),
            "txn_count_24h": random.randint(0, 2),
            "ip_country_mismatch": 0,
            "device_new": 0,
            "device_risk": random.uniform(0.0, 0.3),
        }
    
# added feedback support -> can run in demo_full_loop
def main(n: int = 300, drifted: bool = False, feedback: bool = False) -> None:

    ok = 0
    for _ in range(n):
        payload = one_payload(drifted)

        r = requests.post(f"{API}/predict", json=payload, timeout=10)
        r.raise_for_status()
        out = r.json()
        request_id = out["request_id"]

        if feedback:
            label = int(
                payload["device_risk"] > 0.75
                and payload["ip_country_mismatch"] == 1
                and payload["device_new"] == 1
                and payload["txn_count_24h"] >= 5
            )
            r2 = requests.post(
                f"{API}/feedback",
                json={"request_id": request_id, "outcome_label": label},
                timeout=10,
            )
            r2.raise_for_status()

        ok += 1

    print(f"sent {ok} requests (drifted={drifted}, feedback={feedback})")


    print(f"sent {ok} requests (drifted={drifted})")

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--drifted", action="store_true")
    p.add_argument("--feedback", action="store_true")
    args = p.parse_args()

    main(n=args.n, drifted=args.drifted, feedback=args.feedback)

