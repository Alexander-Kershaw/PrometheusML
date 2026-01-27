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

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    
    # MLflow logging
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)

    with mlflow.start_run(run_name="logreg_baseline"):
        pipeline.fit(X_train, y_train)

        # Validation
        val_probs = pipeline.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_probs)

        # Test
        test_probs = pipeline.predict_proba(X_test)[:, 1]
        test_preds = (test_probs >= 0.5).astype(int)

        """
        ---------------------------------------------------------------------------------------------------
        Rare-event metrics 
        ---------------------------------------------------------------------------------------------------
        PR-AUC (Average Precision) is usually more meaningful than ROC-AUC for heavy imbalance.

        ROC-AUC can look decent even when a model doesn't provide any meaningful fraud indicators

        A more realistic approach: Taking X percent of the most riskiest transactions, and
        evaluating the true positives of chargebacks, aiming for high precision
        ---------------------------------------------------------------------------------------------------
        """
        val_ap = average_precision_score(y_val, val_probs)
        test_ap = average_precision_score(y_test, test_probs)

        # Precision/Recall at top-k percent of riskiest transactions
        def pr_at_k(y_true: pd.Series, probs: np.ndarray, k_frac: float) -> tuple[float, float]:
            k = max(1, int(len(probs) * k_frac))
            idx = np.argsort(-probs)[:k]  # indices of top-k probabilities
            y_top = y_true.iloc[idx].to_numpy()
            precision = float(y_top.mean())
            recall = float(y_top.sum() / max(1, y_true.sum()))  # positives caught
            return precision, recall

        p_at_1pct, r_at_1pct = pr_at_k(y_test, test_probs, 0.01)
        p_at_5pct, r_at_5pct = pr_at_k(y_test, test_probs, 0.05)

        # Log extra metrics
        mlflow.log_metric("val_ap", float(val_ap))
        mlflow.log_metric("test_ap", float(test_ap))
        mlflow.log_metric("test_precision_at_1pct", float(p_at_1pct))
        mlflow.log_metric("test_recall_at_1pct", float(r_at_1pct))
        mlflow.log_metric("test_precision_at_5pct", float(p_at_5pct))
        mlflow.log_metric("test_recall_at_5pct", float(r_at_5pct))

        print(f"Val AP  : {val_ap:.4f}")
        print(f"Test AP : {test_ap:.4f}")
        print(f"P@1% / R@1%: {p_at_1pct:.4f} / {r_at_1pct:.4f}")
        print(f"P@5% / R@5%: {p_at_5pct:.4f} / {r_at_5pct:.4f}")


        test_auc = roc_auc_score(y_test, test_probs)
        test_acc = accuracy_score(y_test, test_preds)

        # Log params
        mlflow.log_param("model_type", "logistic_regression")
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("threshold", 0.5)

        # Log metrics
        mlflow.log_metric("val_auc", float(val_auc))
        mlflow.log_metric("test_auc", float(test_auc))
        mlflow.log_metric("test_accuracy", float(test_acc))

        # Log and register the pipeline as a model artifact
        # (Prevents train/serve skew)
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            registered_model_name=settings.mlflow_model_name,
        )

        print("Training complete")
        print(f"Val AUC : {val_auc:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
        print(f"Test ACC: {test_acc:.4f}")


if __name__ == "__main__":
    main()
