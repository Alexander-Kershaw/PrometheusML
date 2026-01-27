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
from sklearn.model_selection import StratifiedKFold, cross_val_score

from prometheus.utils.config import get_settings

# To silence the excessive MLflow logs
import logging
logging.getLogger("mlflow").setLevel(logging.WARNING)
logging.getLogger("alembic").setLevel(logging.WARNING)


def main() -> None:
    settings = get_settings()
    print("MLFLOW_TRACKING_URI =", settings.mlflow_tracking_uri)

    
    # Load training data
    df = pd.read_csv(Path("data/raw_train.csv"))

    target = "is_chargeback" # chargeback binary target
    drop_cols = ["p_chargeback_true"]  # drop unecessary column

    X = df.drop(columns=[target] + drop_cols)
    y = df[target].astype(int)

    # Feature engineering
    X = X.copy()

    # Amount normalized by recent activity 
    X["amount_per_txn_24h"] = X["amount_gbp"] / (1.0 + X["txn_count_24h"])

    # Some interaction flags that tend to matter in payment risk
    X["night_and_mismatch"] = ((X["hour"] <= 5) & (X["ip_country_mismatch"] == 1)).astype(int)
    X["new_device_and_high_velocity"] = ((X["device_new"] == 1) & (X["txn_count_24h"] >= 3)).astype(int)

    # Smooth velocity feature
    X["velocity_log"] = np.log1p(X["txn_count_24h"])

   
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
        "amount_per_txn_24h",
        "velocity_log",
    ]
    categorical_features = [
        "is_weekend",
        "merchant_category",
        "payment_method",
        "ip_country_mismatch",
        "device_new",
        "night_and_mismatch",
        "new_device_and_high_velocity",
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

    # Single validation splitis too brittle
    # Each candidate gets a 5-fold stratified cross validation AP on training data
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

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
        best_model = None
        best_cv_ap = -1.0

        # Choose by cross-validated Average Precision on training set
        for name, model in candidates.items():
            pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

            cv_ap_scores = cross_val_score(
                pipe,
                X_train,
                y_train,
                cv=cv,
                scoring="average_precision",
                n_jobs=None,
            )
            cv_ap_mean = float(np.mean(cv_ap_scores))
            cv_ap_std = float(np.std(cv_ap_scores))

            mlflow.log_metric(f"{name}_cv_ap_mean", cv_ap_mean)
            mlflow.log_metric(f"{name}_cv_ap_std", cv_ap_std)

            print(f"[{name}] cv_ap={cv_ap_mean:.4f}±{cv_ap_std:.4f}")

            if cv_ap_mean > best_cv_ap:
                best_cv_ap = cv_ap_mean
                best_name = name
                best_model = model

        assert best_name is not None and best_model is not None

        # Fit chosen model (best CV AP) on full train -> then evaluate on val and test
        best_pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", best_model)])
        best_pipe.fit(X_train, y_train)

        val_probs = best_pipe.predict_proba(X_val)[:, 1]
        test_probs = best_pipe.predict_proba(X_test)[:, 1]

        val_auc = roc_auc_score(y_val, val_probs)
        test_auc = roc_auc_score(y_test, test_probs)
        val_ap = average_precision_score(y_val, val_probs)
        test_ap = average_precision_score(y_test, test_probs)

        mlflow.log_param("chosen_model", best_name)
        mlflow.log_metric("cv_ap_mean", best_cv_ap)
        mlflow.log_metric("val_ap", float(val_ap))
        mlflow.log_metric("test_ap", float(test_ap))
        mlflow.log_metric("val_auc", float(val_auc))
        mlflow.log_metric("test_auc", float(test_auc))

        mlflow.sklearn.log_model(
            sk_model=best_pipe,
            name="model",
            registered_model_name=settings.mlflow_model_name,
        )

        print("Chosen model:", best_name)
        print(f"CV AP  : {best_cv_ap:.4f}")
        print(f"Val AP : {val_ap:.4f}")
        print(f"Test AP: {test_ap:.4f}")
        print(f"Val AUC : {val_auc:.4f}")
        print(f"Test AUC: {test_auc:.4f}")


if __name__ == "__main__":
    main()
