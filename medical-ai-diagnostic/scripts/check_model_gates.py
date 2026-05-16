#!/usr/bin/env python3
"""
CI gate: verify the latest MLflow training run meets medical performance thresholds.

Safety-critical gates (model cannot be promoted without passing all):
  - Top-3 accuracy     >= 80%   (diagnostic accuracy)
  - Urgency AUC        >= 95%   (emergency detection)
  - Red flag recall    >= 98%   (life-threatening conditions — highest priority)
  - Calibration ECE   <=  5%   (confidence reliability)

Usage:
    MLFLOW_TRACKING_URI=http://mlflow:5000 python scripts/check_model_gates.py
    python scripts/check_model_gates.py  # falls back to local artifact file
Exit codes: 0 = all gates passed, 1 = one or more gates failed.
"""
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

PERFORMANCE_GATES = {
    "test_top3_accuracy":   (">=", 0.80),
    "test_urgency_auc":     (">=", 0.95),
    "test_red_flag_recall": (">=", 0.98),
    "test_calibration_ece": ("<=", 0.05),
}


def load_from_mlflow() -> dict:
    import mlflow
    from mlflow.tracking import MlflowClient

    uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(uri)
    client = MlflowClient()

    experiment = client.get_experiment_by_name("medical-diagnostic-model")
    if experiment is None:
        raise RuntimeError("MLflow experiment 'medical-diagnostic-model' not found")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No finished runs found")

    run = runs[0]
    logger.info(f"Checking run: {run.info.run_id} ({run.info.run_name})")
    return run.data.metrics


def load_from_file() -> dict:
    path = Path("mlops/mlflow/artifacts/evaluation/eval_metrics.json")
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    with open(path) as f:
        return json.load(f)


def check_gates(metrics: dict) -> bool:
    logger.info("=" * 56)
    logger.info("  Performance Gate Check")
    logger.info("=" * 56)

    all_passed = True
    for metric, (op, threshold) in PERFORMANCE_GATES.items():
        value = metrics.get(metric)
        if value is None:
            logger.error(f"  MISSING  {metric:<30} (not logged in run)")
            all_passed = False
            continue

        passed = (value >= threshold) if op == ">=" else (value <= threshold)
        mark = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"  {mark}  {metric:<30} {value:.4f}  (gate: {op}{threshold})")
        if not passed:
            all_passed = False

    logger.info("=" * 56)
    return all_passed


def main() -> None:
    try:
        metrics = load_from_mlflow()
    except Exception as exc:
        logger.warning(f"MLflow unavailable ({exc}) — falling back to local metrics file")
        try:
            metrics = load_from_file()
        except FileNotFoundError as fe:
            logger.error(str(fe))
            sys.exit(1)

    if check_gates(metrics):
        logger.info("✅  All gates PASSED — model is eligible for staging promotion")
        sys.exit(0)
    else:
        logger.error("❌  Gates FAILED — model cannot be promoted")
        sys.exit(1)


if __name__ == "__main__":
    main()
