from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import duckdb
import logging
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from prometheus.utils.config import get_settings


logging.getLogger("mlflow").setLevel(logging.WARNING)
logging.getLogger("alembic").setLevel(logging.WARNING)


BASE_NUMERIC = ["amount_gbp", "hour", "account_age_days", "txn_count_24h", "device_risk"]
BASE_CATEGORICAL = ["is_weekend", "merchant_category", "payment_method", "ip_country_mismatch", "device_new"]

ENG_NUMERIC = ["amount_per_txn_24h", "velocity_log"]
ENG_CATEGORICAL = ["night_and_mismatch", "new_device_and_high_velocity"]


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["amount_per_txn_24h"] = df["amount_gbp"] / (1.0 + df["txn_count_24h"])
    df["night_and_mismatch"] = ((df["hour"] <= 5) & (df["ip_country_mismatch"] == 1)).astype(int)
    df["new_device_and_high_velocity"] = ((df["device_new"] == 1) & (df["txn_count_24h"] >= 3)).astype(int)
    df["velocity_log"] = np.log1p(df["txn_count_24h"])
    return df


def load_labeled_feedback(db_path: str) -> pd.DataFrame:
    
    # Join inference logs and feedback to obtain labeled training rows

    con = duckdb.connect(db_path)
    try:
        inf = con.execute(
            """
            select request_id, model_version, features_json
            from inference_logs
            """
        ).fetchdf()
        fb = con.execute(
            """
            select request_id, outcome_label
            from feedback
            """
        ).fetchdf()
    finally:
        con.close()

    if inf.empty or fb.empty:
        return pd.DataFrame()

    merged = inf.merge(fb, on="request_id", how="inner")
    if merged.empty:
        return pd.DataFrame()

    feats = pd.json_normalize(merged["features_json"].apply(json.loads))
    out = pd.concat([merged.drop(columns=["features_json"]).reset_index(drop=True),
                     feats.reset_index(drop=True)], axis=1)

    out = out.rename(columns={"outcome_label": "is_chargeback"})
    out["is_chargeback"] = out["is_chargeback"].astype(int)
    return out


def build_preprocessor() -> ColumnTransformer:
    numeric_features = BASE_NUMERIC + ENG_NUMERIC
    categorical_features = BASE_CATEGORICAL + ENG_CATEGORICAL

    numeric_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ]
    )


@dataclass(frozen=True)
class Candidate:
    name: str
    model: Any


def get_current_champion_uri(model_name: str) -> tuple[str, str]:
    
    # Returns (model_uri, version_str) for the latest registered version
    
    client = mlflow.tracking.MlflowClient()
    latest = client.get_latest_versions(model_name)
    if not latest:
        raise RuntimeError(f"No registered versions for model '{model_name}'. Train first.")

    v = max(latest, key=lambda mv: int(mv.version))
    return f"models:/{model_name}/{v.version}", str(v.version)


def main() -> None:
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)

    df = load_labeled_feedback(settings.duckdb_path)

    # Retrain only on sample sizes of sufficient size (otherwise just noise)
    if df.empty or len(df) < 200:
        raise SystemExit(
            f"Not enough labeled feedback to retrain (need >=200 rows). "
            f"Currently: {0 if df.empty else len(df)}. Add more /feedback first."
        )

    y = df["is_chargeback"].astype(int)

    base_cols = BASE_NUMERIC + BASE_CATEGORICAL
    X_base = df[base_cols].copy()
    X_feat = feature_engineering(X_base)

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X_feat, y, test_size=0.25, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()

    candidates = [
        Candidate(
            "logreg",
            LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs"),
        ),
        Candidate(
            "hgb",
            HistGradientBoostingClassifier(max_depth=6, learning_rate=0.08, max_iter=300, random_state=42),
        ),
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    with mlflow.start_run(run_name="retrain_from_feedback"):
  
        # Selects best candidate by CV AP on training data

        best_name = None
        best_model = None
        best_cv_ap = -1.0

        for c in candidates:
            pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", c.model)])
            scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="average_precision")
            mean_ap = float(np.mean(scores))
            std_ap = float(np.std(scores))

            mlflow.log_metric(f"{c.name}_cv_ap_mean", mean_ap)
            mlflow.log_metric(f"{c.name}_cv_ap_std", std_ap)
            print(f"[{c.name}] cv_ap={mean_ap:.4f}±{std_ap:.4f}")

            if mean_ap > best_cv_ap:
                best_cv_ap = mean_ap
                best_name = c.name
                best_model = c.model

        assert best_name is not None and best_model is not None

        candidate_pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", best_model)])
        candidate_pipe.fit(X_train, y_train)

        cand_probs = candidate_pipe.predict_proba(X_holdout)[:, 1]
        cand_ap = float(average_precision_score(y_holdout, cand_probs))
        cand_auc = float(roc_auc_score(y_holdout, cand_probs))

        # Load current champion and score on same holdout
        champion_uri, champion_version = get_current_champion_uri(settings.mlflow_model_name)

        champion_pipe = mlflow.sklearn.load_model(champion_uri)

        champ_probs = champion_pipe.predict_proba(X_holdout)[:, 1]
        champ_ap = float(average_precision_score(y_holdout, champ_probs))
        champ_auc = float(roc_auc_score(y_holdout, champ_probs))

        print(f"Champion v{champion_version}: AP={champ_ap:.4f} AUC={champ_auc:.4f}")
        print(f"Candidate ({best_name}): AP={cand_ap:.4f} AUC={cand_auc:.4f}")

        mlflow.log_param("chosen_candidate", best_name)
        mlflow.log_param("champion_version", champion_version)
        mlflow.log_metric("candidate_cv_ap", best_cv_ap)
        mlflow.log_metric("candidate_holdout_ap", cand_ap)
        mlflow.log_metric("candidate_holdout_auc", cand_auc)
        mlflow.log_metric("champion_holdout_ap", champ_ap)
        mlflow.log_metric("champion_holdout_auc", champ_auc)

    
        # Promotion rule: must beat champion by at least a small margin
        margin = 0.001  
        mlflow.log_param("promotion_margin_ap", margin)

        if cand_ap > champ_ap + margin:
            mlflow.sklearn.log_model(
                sk_model=candidate_pipe,
                name="model",
                registered_model_name=settings.mlflow_model_name,
            )

            print("PROMOTED: candidate registered as a new model version.")
            print("PROMOTED_FLAG=1")
        else:
            print("PROMOTED_FLAG=0")


if __name__ == "__main__":
    main()
