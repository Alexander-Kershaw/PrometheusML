from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
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
            artifact_path="model",
            registered_model_name=settings.mlflow_model_name,
        )

        print("Training complete")
        print(f"Val AUC : {val_auc:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
        print(f"Test ACC: {test_acc:.4f}")


if __name__ == "__main__":
    main()
