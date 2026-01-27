from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

# Inference logs table in duckdb
DDL = """
create table if not exists inference_logs (
    ts_utc           timestamp,
    request_id       varchar,
    model_name       varchar,
    model_version    varchar,
    features_json    varchar,
    proba            double,
    pred             integer
);
"""

# Ensure DuckDB file exists containing the inference logs table
def ensure_store(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(path)
    try:
        con.execute(DDL)
    finally:
        con.close()

# Insert logs (predictions), with standard log fields (timestamp, model_name ..etc)
def log_inference(
    *,
    path: str,
    request_id: str,
    model_name: str,
    model_version: str,
    features: dict[str, Any],
    proba: float,
    pred: int,
) -> None:
    con = duckdb.connect(path)
    try:
        con.execute(DDL)
        con.execute(
            "insert into inference_logs values (?, ?, ?, ?, ?, ?, ?)",
            [
                datetime.now(timezone.utc).replace(tzinfo=None),
                request_id,
                model_name,
                model_version,
                json.dumps(features, separators=(",", ":"), sort_keys=True),
                float(proba),
                int(pred),
            ],
        )
    finally:
        con.close()
