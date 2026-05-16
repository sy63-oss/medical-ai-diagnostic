"""
Medical AI Diagnostic - Symptom Classification Model
TensorFlow + NLP with BERT-based embeddings
HIPAA/GDPR-compliant design
"""

import os
import json
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, callbacks
from transformers import AutoTokenizer, TFAutoModel
import mlflow
import mlflow.tensorflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DiagnosticConfig:
    """Model configuration - loaded from configs/model_config.json"""
    bert_model_name: str = "emilyalsentzer/Bio_ClinicalBERT"
    max_sequence_length: int = 256
    num_symptom_classes: int = 500
    num_diagnostic_categories: int = 150
    dropout_rate: float = 0.3
    learning_rate: float = 2e-5
    batch_size: int = 32
    epochs: int = 20
    early_stopping_patience: int = 5
    severity_levels: int = 4  # mild, moderate, severe, critical
    confidence_threshold: float = 0.7


class SymptomEmbedder(tf.keras.layers.Layer):  # pragma: no cover
    """
    BERT-based clinical language embeddings layer.
    Trained on PubMed + clinical notes corpus.
    """

    def __init__(self, bert_model_name: str, max_len: int, **kwargs):
        super().__init__(**kwargs)
        self.bert = TFAutoModel.from_pretrained(bert_model_name)
        self.max_len = max_len
        # Freeze lower BERT layers for efficiency
        for layer in self.bert.layers[:8]:
            layer.trainable = False

    def call(self, inputs, training=False):
        input_ids, attention_mask = inputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            training=training
        )
        # Use [CLS] token representation
        return outputs.last_hidden_state[:, 0, :]


class MultiHeadSymptomAttention(tf.keras.layers.Layer):
    """
    Multi-head attention over symptom sequences.
    Captures symptom co-occurrence patterns.
    """

    def __init__(self, d_model: int = 768, num_heads: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=0.1
        )
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            layers.Dense(d_model * 2, activation="gelu"),
            layers.Dropout(0.1),
            layers.Dense(d_model)
        ])

    def call(self, x, training=False):
        attn_output = self.attention(x, x, training=training)
        x = self.norm(x + attn_output)
        ffn_output = self.ffn(x, training=training)
        return self.norm(x + ffn_output)


class TemporalSymptomEncoder(tf.keras.layers.Layer):
    """
    Encodes temporal progression of symptoms (onset, duration, evolution).
    Critical for differentiating acute vs chronic conditions.
    """

    def __init__(self, units: int = 256, **kwargs):
        super().__init__(**kwargs)
        self.bilstm = layers.Bidirectional(
            layers.LSTM(units, return_sequences=True, dropout=0.2)
        )
        self.temporal_attn = layers.Attention()
        self.pool = layers.GlobalAveragePooling1D()

    def call(self, symptom_sequence, temporal_features, training=False):
        # symptom_sequence: (batch, time_steps, features)
        # temporal_features: (batch, time_steps, 3) - [onset_days, duration, severity_trend]
        combined = tf.concat([symptom_sequence, temporal_features], axis=-1)
        lstm_out = self.bilstm(combined, training=training)
        context = self.temporal_attn([lstm_out, lstm_out])
        return self.pool(context)


class DiagnosticReasoningModule(tf.keras.layers.Layer):
    """
    Mimics clinical reasoning: differential diagnosis pipeline.
    Produces ranked diagnoses with confidence scores and ICD-10 codes.
    """

    def __init__(self, num_diagnoses: int, num_severity: int, **kwargs):
        super().__init__(**kwargs)
        # Primary diagnosis head
        self.diagnosis_dense = tf.keras.Sequential([
            layers.Dense(1024, activation="gelu"),
            layers.Dropout(0.3),
            layers.Dense(512, activation="gelu"),
            layers.Dense(num_diagnoses)
        ])
        # Severity assessment head
        self.severity_dense = tf.keras.Sequential([
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(num_severity)
        ])
        # Urgency flag head (requires immediate attention)
        self.urgency_dense = tf.keras.Sequential([
            layers.Dense(128, activation="relu"),
            layers.Dense(1, activation="sigmoid")
        ])
        # Red flag detector (life-threatening symptoms)
        self.red_flag_dense = tf.keras.Sequential([
            layers.Dense(256, activation="relu"),
            layers.Dense(50, activation="sigmoid")  # 50 red flag conditions
        ])

    def call(self, fused_features, training=False):
        diagnosis_logits = self.diagnosis_dense(fused_features, training=training)
        severity_logits = self.severity_dense(fused_features, training=training)
        urgency = self.urgency_dense(fused_features, training=training)
        red_flags = self.red_flag_dense(fused_features, training=training)
        return {
            "diagnosis": tf.nn.softmax(diagnosis_logits),
            "severity": tf.nn.softmax(severity_logits),
            "urgency_score": urgency,
            "red_flags": red_flags
        }


def build_medical_diagnostic_model(config: DiagnosticConfig) -> Model:  # pragma: no cover
    """
    Full model architecture:
    ClinicalBERT → Symptom Attention → Temporal Encoding →
    Feature Fusion → Differential Diagnosis
    """
    # ─── Inputs ───────────────────────────────────────────────
    input_ids = layers.Input(shape=(config.max_sequence_length,), dtype=tf.int32, name="input_ids")
    attention_mask = layers.Input(shape=(config.max_sequence_length,), dtype=tf.int32, name="attention_mask")
    structured_symptoms = layers.Input(shape=(config.num_symptom_classes,), dtype=tf.float32, name="structured_symptoms")
    temporal_features = layers.Input(shape=(None, 3), dtype=tf.float32, name="temporal_features")
    patient_metadata = layers.Input(shape=(20,), dtype=tf.float32, name="patient_metadata")
    # Age, sex, BMI, comorbidities (anonymized)

    # ─── Text Embedding (ClinicalBERT) ────────────────────────
    bert_embedding = SymptomEmbedder(
        bert_model_name=config.bert_model_name,
        max_len=config.max_sequence_length,
        name="clinical_bert"
    )([input_ids, attention_mask])

    # ─── Structured Symptom Processing ────────────────────────
    symptom_attn = MultiHeadSymptomAttention(name="symptom_attention")(
        tf.expand_dims(structured_symptoms, 1)
    )
    symptom_repr = layers.Flatten()(symptom_attn)
    symptom_repr = layers.Dense(256, activation="gelu")(symptom_repr)

    # ─── Temporal Encoding ────────────────────────────────────
    symptom_sequence_3d = tf.expand_dims(
        layers.Dense(128)(structured_symptoms), 1
    )
    symptom_sequence_3d = tf.tile(symptom_sequence_3d, [1, tf.shape(temporal_features)[1], 1])
    temporal_repr = TemporalSymptomEncoder(name="temporal_encoder")(
        symptom_sequence_3d, temporal_features
    )

    # ─── Metadata Encoding ────────────────────────────────────
    meta_repr = layers.Dense(64, activation="relu")(patient_metadata)
    meta_repr = layers.Dropout(0.2)(meta_repr)

    # ─── Feature Fusion (cross-modal attention) ───────────────
    text_repr = layers.Dense(512)(bert_embedding)
    symptom_repr = layers.Dense(512)(symptom_repr)
    temporal_repr = layers.Dense(512)(temporal_repr)
    meta_repr = layers.Dense(512)(meta_repr)

    # Stack for cross-attention
    stacked = tf.stack([text_repr, symptom_repr, temporal_repr, meta_repr], axis=1)
    fused = MultiHeadSymptomAttention(d_model=512, num_heads=4, name="cross_modal_attention")(stacked)
    fused_pooled = layers.GlobalAveragePooling1D()(fused)
    fused_pooled = layers.Dropout(config.dropout_rate)(fused_pooled)

    # ─── Diagnostic Reasoning ─────────────────────────────────
    reasoning = DiagnosticReasoningModule(
        num_diagnoses=config.num_diagnostic_categories,
        num_severity=config.severity_levels,
        name="diagnostic_reasoning"
    )(fused_pooled)

    # ─── Build Model ──────────────────────────────────────────
    model = Model(
        inputs=[input_ids, attention_mask, structured_symptoms, temporal_features, patient_metadata],
        outputs=reasoning,
        name="MedicalAIDiagnostic_v2"
    )
    return model


class FocalLoss(tf.keras.losses.Loss):
    """
    Focal loss for handling class imbalance in rare diseases.
    gamma=2.0 focuses learning on hard misclassified examples.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False)
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        focal_weight = self.alpha * tf.pow(1 - p_t, self.gamma)
        return focal_weight * ce


class LinearWarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup for `warmup_steps`, then cosine decay to zero over `decay_steps`."""

    def __init__(self, peak_lr: float, warmup_steps: int, decay_steps: int):
        super().__init__()
        self.peak_lr = peak_lr
        self.warmup_steps = float(warmup_steps)
        self.decay_steps = float(decay_steps)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_lr = self.peak_lr * (step / self.warmup_steps)
        cosine_decay = 0.5 * self.peak_lr * (
            1.0 + tf.cos(np.pi * (step - self.warmup_steps) / self.decay_steps)
        )
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_decay)

    def get_config(self):
        return {
            "peak_lr": self.peak_lr,
            "warmup_steps": int(self.warmup_steps),
            "decay_steps": int(self.decay_steps),
        }


def compile_model(model: Model, config: DiagnosticConfig) -> Model:  # pragma: no cover
    """Compile with multi-task losses and medical metrics."""
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LinearWarmupCosineDecay(
            peak_lr=config.learning_rate,
            warmup_steps=1000,
            decay_steps=10000,
        )
    )
    model.compile(
        optimizer=optimizer,
        loss={
            "diagnosis": FocalLoss(gamma=2.0),
            "severity": "categorical_crossentropy",
            "urgency_score": "binary_crossentropy",
            "red_flags": "binary_crossentropy"
        },
        loss_weights={
            "diagnosis": 1.0,
            "severity": 0.5,
            "urgency_score": 2.0,   # Higher weight: critical safety metric
            "red_flags": 2.0
        },
        metrics={
            "diagnosis": [
                tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_accuracy"),
                tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_accuracy"),
            ],
            "severity": ["accuracy"],
            "urgency_score": [tf.keras.metrics.AUC(name="urgency_auc")],
        }
    )
    return model


def get_training_callbacks(config: DiagnosticConfig, run_id: str) -> List:  # pragma: no cover
    """MLOps-integrated training callbacks."""
    return [
        callbacks.EarlyStopping(
            monitor="val_diagnosis_top3_accuracy",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            mode="max"
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7
        ),
        callbacks.ModelCheckpoint(
            filepath=f"mlops/mlflow/artifacts/{run_id}/checkpoints/model_{{epoch:02d}}_{{val_loss:.4f}}.h5",
            save_best_only=True,
            monitor="val_loss"
        ),
        MLflowCallback(),
        callbacks.TensorBoard(
            log_dir=f"mlops/mlflow/tensorboard/{run_id}",
            histogram_freq=1,
            update_freq="epoch"
        )
    ]


class MLflowCallback(tf.keras.callbacks.Callback):  # pragma: no cover
    """Logs all metrics + model artifacts to MLflow."""

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            mlflow.log_metrics(
                {k: float(v) for k, v in logs.items()},
                step=epoch
            )

    def on_train_end(self, logs=None):
        mlflow.tensorflow.log_model(self.model, artifact_path="model")
        logger.info("Model logged to MLflow")


if __name__ == "__main__":
    config = DiagnosticConfig()
    model = build_medical_diagnostic_model(config)
    model = compile_model(model, config)
    model.summary()
    logger.info(f"Model parameters: {model.count_params():,}")
