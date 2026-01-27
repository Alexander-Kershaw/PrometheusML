from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DriftSpec:
    """
    ------------------------------------------------------------------------------------------------
    DriftSpec modulates how production data changes temporally

    drift_strength modulator:
      0.0  -> has the appearance of a training distribution
      1.0+ -> clearly shifted distribution
    ------------------------------------------------------------------------------------------------
    """
    drift_strength: float = 0.0
    seed: int = 7


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_payments(n: int, drift: DriftSpec) -> pd.DataFrame:
    """
    ------------------------------------------------------------------------------------------------
    Generate synthetic payments dataset for chargeback risk

    Returns a DataFrame with features and target binary:
      - is_chargeback (0/1)

    Approach:
    generate features and compute a probability using a logistic-style formula,
    then sample the label from that probability.
    ------------------------------------------------------------------------------------------------
    """
    rng = np.random.default_rng(drift.seed)
    s = float(drift.drift_strength)

    # Core features
    amount_gbp = rng.lognormal(mean=3.3 + 0.25 * s, sigma=0.75 + 0.10 * s, size=n)
    hour = rng.integers(0, 24, size=n)
    is_weekend = (rng.integers(0, 7, size=n) >= 5).astype(int)

    # Categorical features (integer encoded)
    merchant_category = rng.choice([0, 1, 2, 3, 4, 5], size=n, p=[0.20, 0.18, 0.16, 0.16, 0.15, 0.15])
    payment_method = rng.choice([0, 1, 2], size=n, p=[0.55, 0.35, 0.10])  # card, wallet, bank

    # Behavior and risk signals
    account_age_days = rng.integers(0, 3650, size=n)
    txn_count_24h = rng.poisson(lam=1.2 + 0.8 * s, size=n) 
    ip_country_mismatch = rng.binomial(1, p=min(0.04 + 0.05 * s, 0.30), size=n)
    device_new = rng.binomial(1, p=min(0.10 + 0.08 * s, 0.45), size=n)

    # continuous device risk score that drifts upward
    device_risk = np.clip(rng.normal(loc=0.15 + 0.12 * s, scale=0.20 + 0.05 * s, size=n), 0.0, 1.0)

    # Label generation
    # Baseline risk is low as chargeback is not super common, risk increases with following factors:
    log_amount = np.log1p(amount_gbp)

    logit = (
        -5.0
        + 0.55 * (log_amount - 3.8)                 # larger amounts are riskier
        + 0.35 * ip_country_mismatch
        + 0.40 * device_new
        + 1.10 * device_risk
        + 0.20 * (txn_count_24h >= 3).astype(int)   # velocity threshold
        + 0.15 * is_weekend
        + 0.10 * (hour <= 5).astype(int)            # late-night activity
        + 0.12 * (merchant_category == 5).astype(int)  # assuming one riskier merchant category
        - 0.15 * np.log1p(account_age_days)         # older accounts safer
        + 0.25 * s                                  # drift increases base risk only slightly
    )

    p_chargeback = _sigmoid(logit)
    is_chargeback = rng.binomial(1, p=np.clip(p_chargeback, 0.0, 1.0), size=n)

    df = pd.DataFrame(
        {
            "amount_gbp": amount_gbp,
            "hour": hour,
            "is_weekend": is_weekend,
            "merchant_category": merchant_category,
            "payment_method": payment_method,
            "account_age_days": account_age_days,
            "txn_count_24h": txn_count_24h,
            "ip_country_mismatch": ip_country_mismatch,
            "device_new": device_new,
            "device_risk": device_risk,
            "p_chargeback_true": p_chargeback,
            "is_chargeback": is_chargeback,
        }
    )

    return df
