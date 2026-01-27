from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier

from prometheus.utils.config import get_settings


def main() -> None:
    settings = get_settings()

    
    # Load training data
    df = pd.read_csv(Path("data/raw_train.csv"))

    target = "is_chargeback" # chargeback binary target
    drop_cols = ["p_chargeback_true"]  # drop unecessary column

    X = df.drop(columns=[target] + drop_cols)
    y = df[target].astype(int)

   
    # Stratified train/validation/test split 
    # (since chargeback is rare statified split balances and preserves original dataset proptions)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )


    # Feature groups (for preprocessing numeric and categorical distinctly)
    numeric_features = [
        "amount_gbp",
        "hour",
        "account_age_days",
        "txn_count_24h",
        "device_risk",
    ]
    categorical_features = [
        "is_weekend",
        "merchant_category",
        "payment_method",
        "ip_country_mismatch",
        "device_new",
    ]

 
    # Preprocess and model pipeline
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    # model candidates (Regression model and tree based model)
    candidates = {
        "logreg": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
        ),
        "hgb": HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.08,
            max_iter=300,
            random_state=42,
        ),
    }


    # MLflow logging
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)

    with mlflow.start_run(run_name="logreg_hgb_baseline"):
        
        def eval_model(pipe: Pipeline) -> dict[str, float]:
            pipe.fit(X_train, y_train)

            val_probs = pipe.predict_proba(X_val)[:, 1]
            test_probs = pipe.predict_proba(X_test)[:, 1]

            val_auc = roc_auc_score(y_val, val_probs)
            test_auc = roc_auc_score(y_test, test_probs)

            val_ap = average_precision_score(y_val, val_probs)
            test_ap = average_precision_score(y_test, test_probs)

            return {
                "val_auc": float(val_auc),
                "test_auc": float(test_auc),
                "val_ap": float(val_ap),
                "test_ap": float(test_ap),
            }, val_probs, test_probs

        best_name = None
        best_pipe = None
        best_val_ap = -1.0
        best_metrics = None
        best_test_probs = None

        for name, model in candidates.items():
            pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
            metrics, _, test_probs = eval_model(pipe)

            # Log logreg and hgb metrics
            for k, v in metrics.items():
                mlflow.log_metric(f"{name}_{k}", v)

            print(f"[{name}] val_ap={metrics['val_ap']:.4f} test_ap={metrics['test_ap']:.4f}")

            if metrics["val_ap"] > best_val_ap:
                best_val_ap = metrics["val_ap"]
                best_name = name
                best_pipe = pipe
                best_metrics = metrics
                best_test_probs = test_probs

        assert best_pipe is not None and best_name is not None and best_metrics is not None

        # Log chosen model and metrics
        mlflow.log_param("chosen_model", best_name)
        for k, v in best_metrics.items():
            mlflow.log_metric(k, v)

        # Log and register the chosen model
        mlflow.sklearn.log_model(
            best_pipe,
            name="model",
            registered_model_name=settings.mlflow_model_name,
        )

        print("Chosen model:", best_name)
        print(f"Val AP : {best_metrics['val_ap']:.4f}")
        print(f"Test AP: {best_metrics['test_ap']:.4f}")
        print(f"Val AUC : {best_metrics['val_auc']:.4f}")
        print(f"Test AUC: {best_metrics['test_auc']:.4f}")



if __name__ == "__main__":
    main()
