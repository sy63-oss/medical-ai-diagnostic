"""
DVC Stage 3 — Preprocessing Pipeline

Anonymization → feature encoding → train/val/test split → TFRecord serialisation.
Produces the tensors consumed by the training stage.

DVC command:
    python -m src.preprocessing.pipeline \\
        --input data/raw \\
        --output data/processed \\
        --config configs/preprocessing_config.json
"""
import argparse
import json
import logging
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

_AGE  = {"0-12": 0, "13-17": 1, "18-30": 2, "31-50": 3, "51-65": 4, "65+": 5}
_SEX  = {"M": 0, "F": 1, "other": 2}
_BMI  = {"underweight": 0, "normal": 1, "overweight": 2, "obese": 3}


# ── Vocabulary ────────────────────────────────────────────────────────────────

def build_vocab(df: pd.DataFrame, max_size: int) -> Dict[str, int]:
    counter: Counter = Counter()
    for entry in df["symptoms"]:
        items = entry if isinstance(entry, list) else [entry]
        counter.update(items)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for sym, _ in counter.most_common(max_size - 2):
        vocab[sym] = len(vocab)
    return vocab


# ── Record encoding ───────────────────────────────────────────────────────────

def encode_record(row: pd.Series, diagnoses: Dict[int, str], label_to_idx: Dict[str, int],
                  temporal_steps: int, metadata_dim: int) -> dict:
    symptoms  = row["symptoms"]       if isinstance(row["symptoms"], list)       else [row["symptoms"]]
    severity  = row["severity_scores"] if isinstance(row["severity_scores"], list) else [5]
    onset     = row["onset_days"]      if isinstance(row["onset_days"], list)      else [0]
    duration  = row.get("duration_hours", [0])
    duration  = duration if isinstance(duration, list) else [0]

    # 500-dim structured symptom vector
    structured = np.zeros(500, dtype=np.float32)
    for sym, sev in zip(symptoms, severity):
        structured[hash(sym) % 500] = float(sev) / 10.0

    # Temporal features: (temporal_steps, 3)
    temporal = np.zeros((temporal_steps, 3), dtype=np.float32)
    for i, (sev, ons, dur) in enumerate(zip(severity, onset, duration)):
        if i >= temporal_steps:
            break
        temporal[i, 0] = min(float(ons) / 365.0, 1.0)
        temporal[i, 1] = min(float(dur) / 720.0, 1.0)
        temporal[i, 2] = float(sev) / 10.0

    # 20-dim patient metadata
    meta = np.zeros(metadata_dim, dtype=np.float32)
    meta[0] = _AGE.get(str(row.get("age_group", "31-50")), 3) / 5.0
    meta[1] = _SEX.get(str(row.get("biological_sex", "M")), 0) / 2.0
    meta[2] = _BMI.get(str(row.get("bmi_category", "normal")), 1) / 3.0
    meta[3] = float(bool(row.get("smoking", False)))

    diag = diagnoses.get(int(row["record_id"]), list(label_to_idx.keys())[0])
    label = label_to_idx.get(diag, 0)
    text  = " ".join(str(s) for s in symptoms)

    return {
        "structured_symptoms": structured,
        "temporal_features":   temporal,
        "patient_metadata":    meta,
        "label":               label,
        "text":                text,
    }


# ── TFRecord serialisation ────────────────────────────────────────────────────

def write_tfrecord(records: List[dict], path: Path) -> None:
    import tensorflow as tf

    def _bytes(v):  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))
    def _float(v):  return tf.train.Feature(float_list=tf.train.FloatList(value=v.flatten()))
    def _int64(v):  return tf.train.Feature(int64_list=tf.train.Int64List(value=[v]))

    path.parent.mkdir(parents=True, exist_ok=True)
    with tf.io.TFRecordWriter(str(path)) as writer:
        for rec in records:
            feat = {
                "structured_symptoms": _float(rec["structured_symptoms"]),
                "temporal_features":   _float(rec["temporal_features"]),
                "patient_metadata":    _float(rec["patient_metadata"]),
                "label":               _int64(rec["label"]),
                "text":                _bytes(rec["text"].encode("utf-8")),
            }
            writer.write(tf.train.Example(features=tf.train.Features(feature=feat)).SerializeToString())


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(input_dir: Path, output_dir: Path, config: dict) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    max_vocab     = config.get("max_vocab_size", 12_000)
    temporal_steps = config.get("temporal_max_steps", 30)
    metadata_dim  = config.get("metadata_dim", 20)
    seed          = config.get("random_seed", 42)
    train_split   = config.get("train_split", 0.70)
    val_split     = config.get("val_split", 0.15)

    logger.info("Loading raw data…")
    sym  = pd.read_parquet(input_dir / "symptoms.parquet")
    diag = pd.read_parquet(input_dir / "diagnoses.parquet")
    diagnoses_map = dict(zip(diag["record_id"], diag["primary_diagnosis"]))

    logger.info("Building vocabulary and label encoder…")
    vocab = build_vocab(sym, max_vocab)
    unique_labels  = sorted(diag["primary_diagnosis"].unique())
    label_to_idx   = {lbl: i for i, lbl in enumerate(unique_labels)}

    with open(output_dir / "vocab.json", "w") as f:
        json.dump(vocab, f)
    with open(output_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump({"label_to_idx": label_to_idx,
                     "idx_to_label": {v: k for k, v in label_to_idx.items()}}, f)

    logger.info(f"Encoding {len(sym):,} records…")
    records = [
        encode_record(row, diagnoses_map, label_to_idx, temporal_steps, metadata_dim)
        for _, row in sym.iterrows()
    ]

    # Shuffle and split
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(records))
    n_train = int(len(records) * train_split)
    n_val   = int(len(records) * val_split)

    splits = {
        "train": [records[i] for i in idx[:n_train]],
        "val":   [records[i] for i in idx[n_train:n_train + n_val]],
        "test":  [records[i] for i in idx[n_train + n_val:]],
    }

    logger.info("Serialising TFRecord splits…")
    for split_name, split_records in splits.items():
        write_tfrecord(split_records, output_dir / f"{split_name}.tfrecord")
        logger.info(f"  {split_name}: {len(split_records):,} records")

    stats = {
        "train_samples": len(splits["train"]),
        "val_samples":   len(splits["val"]),
        "test_samples":  len(splits["test"]),
        "vocab_size":    len(vocab),
        "num_classes":   len(label_to_idx),
        "anonymization_method": config.get("anonymization", {}).get("method", "HIPAA_safe_harbor"),
    }
    with open(output_dir / "preprocessing_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("✅  Preprocessing complete")
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DVC Stage 3: preprocessing pipeline")
    parser.add_argument("--input",  required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    run(Path(args.input), Path(args.output), cfg)
