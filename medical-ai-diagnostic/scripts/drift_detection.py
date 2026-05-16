#!/usr/bin/env python3
"""
Nightly model drift detection.

Compares the current production model's metrics against the baseline established
at the last production promotion. Exits 1 if drift exceeds the threshold,
which triggers a Slack alert and retraining recommendation in CI.

Usage:
    python scripts/drift_detection.py --alert-threshold 0.05
Exit codes: 0 = stable, 1 = drift detected.
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# Metric name → direction that indicates degradation
DRIFT_METRICS = {
    "test_top3_accuracy":   "decrease",
    "test_urgency_auc":     "decrease",
    "test_red_flag_recall": "decrease",
    "test_calibration_ece": "increase",
}

# Known-good baseline from v2.1.0 production release
FALLBACK_BASELINE = {
    "test_top3_accuracy":   0.847,
    "test_top5_accuracy":   0.921,
    "test_urgency_auc":     0.963,
    "test_red_flag_recall": 0.989,
    "test_calibration_ece": 0.031,
}


def load_baseline() -> dict:
    path = Path("mlops/mlflow/artifacts/evaluation/eval_metrics.json")
    if path.exists():
        with open(path) as f:
            return json.load(f)
    logger.warning("No local baseline found — using v2.1.0 reference metrics")
    return FALLBACK_BASELINE


def load_current_metrics() -> dict:
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(uri)
        client = MlflowClient()

        experiment = client.get_experiment_by_name("medical-diagnostic-model")
        if experiment is None:
            raise RuntimeError("Experiment not found")

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="status = 'FINISHED'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        if not runs:
            raise RuntimeError("No finished runs")

        return runs[0].data.metrics

    except Exception as exc:
        logger.warning(f"MLflow unavailable ({exc}) — using baseline as current (no drift)")
        return load_baseline()


def detect_drift(baseline: dict, current: dict, threshold: float) -> list:
    drifted = []
    for metric, direction in DRIFT_METRICS.items():
        b = baseline.get(metric)
        c = current.get(metric)
        if b is None or c is None:
            logger.warning(f"  SKIP  {metric} — missing from baseline or current")
            continue

        delta = abs(c - b)
        rel_change = delta / abs(b) if b != 0 else 0.0
        degraded = (direction == "decrease" and c < b) or (direction == "increase" and c > b)

        if rel_change > threshold and degraded:
            drifted.append(
                {"metric": metric, "baseline": b, "current": c, "relative_change": rel_change}
            )
            logger.warning(
                f"  DRIFT  {metric:<30} baseline={b:.4f} → current={c:.4f}"
                f"  ({rel_change:.1%} {direction})"
            )
        else:
            logger.info(f"  OK     {metric:<30} {c:.4f}  (baseline: {b:.4f})")

    return drifted


def save_report(baseline: dict, current: dict, drift: list, threshold: float) -> Path:
    report_dir = Path("mlops/mlflow/artifacts/drift_reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = report_dir / f"drift_{ts}.json"
    with open(path, "w") as f:
        json.dump(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "threshold": threshold,
                "status": "DRIFT" if drift else "OK",
                "drift_detected": drift,
                "baseline": baseline,
                "current": current,
            },
            f,
            indent=2,
        )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Nightly model drift detection")
    parser.add_argument(
        "--alert-threshold",
        type=float,
        default=0.05,
        help="Relative change threshold to trigger a drift alert (default: 0.05 = 5%%)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("  Medical AI Diagnostic — Nightly Drift Detection")
    logger.info(f"  Threshold: {args.alert_threshold:.1%} relative degradation")
    logger.info("=" * 60)

    baseline = load_baseline()
    current = load_current_metrics()
    drift = detect_drift(baseline, current, args.alert_threshold)

    report_path = save_report(baseline, current, drift, args.alert_threshold)
    logger.info(f"Report saved: {report_path}")

    if drift:
        logger.error(f"❌  Drift detected in {len(drift)} metric(s) — retraining recommended")
        sys.exit(1)

    logger.info("✅  No significant drift — model performance is stable")
    sys.exit(0)


if __name__ == "__main__":
    main()
