"""
DVC Stage 5 — Model Evaluation

Loads the best checkpoint, runs inference on the held-out test split, and
computes the medical performance metrics that gate production promotion.
Writes metrics.json, confusion_matrix.json, and roc_curve.json for DVC plots.

DVC command:
    python -m src.evaluation.evaluate \\
        --model mlops/mlflow/artifacts/best_model \\
        --test-data data/processed/test.tfrecord \\
        --output mlops/mlflow/artifacts/evaluation
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

TEMPORAL_STEPS = 30
METADATA_DIM   = 20
MAX_SEQ_LEN    = 256


# ── Data loading ──────────────────────────────────────────────────────────────

def load_test_data(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import tensorflow as tf

    feature_spec = {
        "structured_symptoms": tf.io.FixedLenFeature([500], tf.float32),
        "temporal_features":   tf.io.FixedLenFeature([TEMPORAL_STEPS * 3], tf.float32),
        "patient_metadata":    tf.io.FixedLenFeature([METADATA_DIM], tf.float32),
        "label":               tf.io.FixedLenFeature([], tf.int64),
        "text":                tf.io.FixedLenFeature([], tf.string),
    }

    def _parse(proto):
        return tf.io.parse_single_example(proto, feature_spec)

    dataset = tf.data.TFRecordDataset(str(path)).map(_parse).batch(256)

    structured_l, temporal_l, metadata_l, labels_l = [], [], [], []
    for batch in dataset:
        structured_l.append(batch["structured_symptoms"].numpy())
        temporal_l.append(batch["temporal_features"].numpy().reshape(-1, TEMPORAL_STEPS, 3))
        metadata_l.append(batch["patient_metadata"].numpy())
        labels_l.append(batch["label"].numpy())

    return (
        np.concatenate(structured_l),
        np.concatenate(temporal_l),
        np.concatenate(metadata_l),
        np.concatenate(labels_l),
    )


# ── Metrics ───────────────────────────────────────────────────────────────────

def top_k_accuracy(probs: np.ndarray, labels: np.ndarray, k: int) -> float:
    top_k = np.argsort(probs, axis=-1)[:, -k:]
    return float(np.mean([labels[i] in top_k[i] for i in range(len(labels))]))


def roc_auc(scores: np.ndarray, binary_labels: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if binary_labels.sum() == 0 or binary_labels.sum() == len(binary_labels):
        return 0.5  # degenerate case
    return float(roc_auc_score(binary_labels, scores))


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    max_p    = probs.max(axis=-1)
    predicted = probs.argmax(axis=-1)
    correct  = (predicted == labels).astype(float)
    bins     = np.linspace(0.0, 1.0, n_bins + 1)
    ece      = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (max_p >= lo) & (max_p < hi)
        if mask.sum() > 0:
            ece += mask.mean() * abs(correct[mask].mean() - max_p[mask].mean())
    return float(ece)


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(model_path: Path, test_data_path: Path, output_dir: Path) -> Dict[str, Any]:
    import tensorflow as tf

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model from {model_path}…")
    model = tf.saved_model.load(str(model_path))

    logger.info(f"Loading test data from {test_data_path}…")
    structured, temporal, metadata, labels = load_test_data(test_data_path)
    n = len(labels)
    logger.info(f"Running inference on {n:,} test samples…")

    # Padding: text inputs set to zeros (structured inference only)
    input_ids      = tf.zeros((n, MAX_SEQ_LEN), dtype=tf.int32)
    attention_mask = tf.ones((n, MAX_SEQ_LEN),  dtype=tf.int32)

    outputs = model(
        {
            "input_ids":           input_ids,
            "attention_mask":      attention_mask,
            "structured_symptoms": tf.constant(structured),
            "temporal_features":   tf.constant(temporal),
            "patient_metadata":    tf.constant(metadata),
        },
        training=False,
    )

    diag_probs    = outputs["diagnosis"].numpy()          # (n, 150)
    urgency       = outputs["urgency_score"].numpy().squeeze()  # (n,)
    red_flag_probs = outputs["red_flags"].numpy()          # (n, 50)

    # Binarise urgency and red-flag ground truth (heuristic for synthetic data)
    urgency_gt    = (labels % 4 == 0).astype(int)
    red_flag_gt   = (labels % 7 == 0).astype(int)
    red_flag_pred = (red_flag_probs.max(axis=-1) > 0.5).astype(int)

    metrics: Dict[str, Any] = {
        "test_top3_accuracy":   top_k_accuracy(diag_probs, labels, k=3),
        "test_top5_accuracy":   top_k_accuracy(diag_probs, labels, k=5),
        "test_urgency_auc":     roc_auc(urgency, urgency_gt),
        "test_red_flag_recall": float(
            (red_flag_pred & red_flag_gt).sum() / max(int(red_flag_gt.sum()), 1)
        ),
        "test_calibration_ece": expected_calibration_error(diag_probs, labels),
        "n_test_samples":       n,
    }

    logger.info("─" * 50)
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k:<30} {v:.4f}")
    logger.info("─" * 50)

    # ── Artifacts ────────────────────────────────────────────
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Confusion matrix (top-10 predicted classes)
    predicted  = diag_probs.argmax(axis=-1)
    top_classes = list(range(min(10, diag_probs.shape[-1])))
    cm = []
    for true_cls in top_classes:
        mask = labels == true_cls
        if mask.sum() == 0:
            continue
        for pred_cls in top_classes:
            cm.append({"actual": true_cls, "predicted": pred_cls,
                        "count": int((predicted[mask] == pred_cls).sum())})
    with open(output_dir / "confusion_matrix.json", "w") as f:
        json.dump(cm, f)

    # ROC curve (urgency detection)
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(urgency_gt, urgency)
    step = max(1, len(fpr) // 100)   # keep ≤100 points for readability
    roc  = [{"fpr": float(f), "tpr": float(t)} for f, t in zip(fpr[::step], tpr[::step])]
    with open(output_dir / "roc_curve.json", "w") as f:
        json.dump(roc, f)

    logger.info(f"✅  Evaluation artifacts saved → {output_dir}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DVC Stage 5: model evaluation")
    parser.add_argument("--model",     required=True, help="Path to TF SavedModel directory")
    parser.add_argument("--test-data", required=True, help="Path to test.tfrecord")
    parser.add_argument("--output",    required=True, help="Output directory for artifacts")
    args = parser.parse_args()
    evaluate(Path(args.model), Path(args.test_data), Path(args.output))
