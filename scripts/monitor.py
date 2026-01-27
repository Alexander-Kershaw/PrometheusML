from __future__ import annotations

from datetime import datetime
from pathlib import Path

import duckdb
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from prometheus.utils.config import get_settings


NUMERIC_FEATURES = [
    "amount_gbp",
    "hour",
    "account_age_days",
    "txn_count_24h",
    "device_risk",
]

PSI_BINS = 10


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = PSI_BINS) -> float:
    """
    ------------------------------------------------------------------------------------
    Population Stability Index (PSI)
    ------------------------------------------------------------------------------------

    expected = baseline distribution (training or older data / production window)
    actual = recent distribution (newer data / last N inferences)

    PSI interpretation:

    How mich the probability mass has shifted across bins

    <0.1  = little change
    0.1-0.2 = moderate shift
    >0.2  = large shift
    ------------------------------------------------------------------------------------
    """
    # Clear missing values (so histograms work)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    # Sample size guard (too small sample -> PSI can end up just being random noise)
    if len(expected) < 100 or len(actual) < 100:
        return float("nan")

    # quantile bins from expected distribution (stable comparisions since each bin will have more equal counts)
    quantiles = np.linspace(0, 1, bins + 1)
    cutpoints = np.unique(np.quantile(expected, quantiles))

    # Ensure PSI is reliable with sufficient cutpoints (no degenerate bins)
    if len(cutpoints) < 3:
        return float("nan")

    exp_counts, _ = np.histogram(expected, bins=cutpoints)
    act_counts, _ = np.histogram(actual, bins=cutpoints)

    exp_perc = exp_counts / max(1, exp_counts.sum())
    act_perc = act_counts / max(1, act_counts.sum())

    # prevent division by zero or log(0)
    eps = 1e-6
    exp_perc = np.clip(exp_perc, eps, 1.0)
    act_perc = np.clip(act_perc, eps, 1.0)

    return float(np.sum((act_perc - exp_perc) * np.log(act_perc / exp_perc))) #PSI formula


def main() -> None:
    settings = get_settings()
    db_path = settings.duckdb_path

    con = duckdb.connect(db_path)
    try:
        # Pull all inference logs
        inf = con.execute(
            """
            select ts_utc, request_id, model_version, features_json, proba, pred
            from inference_logs
            order by ts_utc asc
            """
        ).fetchdf()

        # Order from oldest to newest
        fb = con.execute(
            """
            select ts_utc, request_id, outcome_label
            from feedback
            order by ts_utc asc
            """
        ).fetchdf()
    finally:
        con.close()

    if inf.empty:
        raise SystemExit("No inferences logged yet. Call /predict first.")

    # Parse feature JSON into columns
    feats = pd.json_normalize(inf["features_json"].apply(lambda s: json.loads(s)))

    df = pd.concat(
        [inf.drop(columns=["features_json"]).reset_index(drop=True),
        feats.reset_index(drop=True)],
        axis=1,
    )
    
    # Define baseline and recent windows
    # Baseline = older half, Recent = newest half
    split_idx = len(df) // 2
    baseline = df.iloc[:split_idx].copy()
    recent = df.iloc[split_idx:].copy()

    # Prediction drift
    baseline_mean_proba = float(baseline["proba"].mean())
    recent_mean_proba = float(recent["proba"].mean())
    proba_shift = recent_mean_proba - baseline_mean_proba

    # Feature drift
    psi_rows = []
    for col in NUMERIC_FEATURES:
        p = psi(baseline[col].to_numpy(dtype=float), recent[col].to_numpy(dtype=float))
        psi_rows.append((col, p))

    psi_df = pd.DataFrame(psi_rows, columns=["feature", "psi"]).sort_values("psi", ascending=False)

    # If we have feedback labels, compute performance on matched request_ids
    perf_md = ""
    if not fb.empty:
        merged = df.merge(fb[["request_id", "outcome_label"]], on="request_id", how="inner")
        if merged["outcome_label"].nunique() >= 2:
            ap = float(average_precision_score(merged["outcome_label"], merged["proba"]))
            auc = float(roc_auc_score(merged["outcome_label"], merged["proba"]))
            perf_md = (
                f"\n## Labeled performance (matched feedback)\n"
                f"- labeled_rows: {len(merged)}\n"
                f"- AP (PR-AUC): {ap:.4f}\n"
                f"- ROC-AUC: {auc:.4f}\n"
            )
        else:
            perf_md = (
                f"\n## Labeled performance (matched feedback)\n"
                f"- labeled_rows: {len(merged)}\n"
                f"- Not enough label variety yet to compute AP/AUC.\n"
            )
    else:
        perf_md = "\n## Labeled performance\n- No feedback rows yet.\n"

    # Define triggers:
    # - Any PSI above threshold -> drift trigger
    # - Mean proba shift big -> drift trigger
    psi_trigger = bool((psi_df["psi"] > settings.drift_psi_threshold).any())
    proba_trigger = abs(proba_shift) > 0.01  # trigger threshold

    trigger = psi_trigger or proba_trigger

    # Write report
    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M")
    out_path = report_dir / f"drift_{ts}.md"

    md = []
    md.append(f"# Prometheus Drift Report ({ts} UTC)\n")
    md.append(f"DB: `{db_path}`\n")
    md.append(f"Rows: total={len(df)} baseline={len(baseline)} recent={len(recent)}\n")

    md.append("## Prediction drift\n")
    md.append(f"- baseline_mean_proba: {baseline_mean_proba:.6f}\n")
    md.append(f"- recent_mean_proba: {recent_mean_proba:.6f}\n")
    md.append(f"- shift (recent - baseline): {proba_shift:.6f}\n")
    md.append(f"- trigger (|shift|>0.01): {proba_trigger}\n")

    md.append("\n## Feature drift (PSI)\n")
    md.append(f"- psi_threshold: {settings.drift_psi_threshold}\n\n")
    md.append(psi_df.to_markdown(index=False))
    md.append("\n")
    md.append(f"\n- trigger (any PSI > threshold): {psi_trigger}\n")

    md.append(perf_md)

    md.append("\n## Retrain decision\n")
    md.append(f"- retrain_triggered: {trigger}\n")

    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote report: {out_path}")


if __name__ == "__main__":
    main()
