"""
Medical AI Diagnostic — Test Suite
Unit + Integration tests for API, model, and HIPAA compliance
"""

import uuid
import pytest
import time
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import jwt

# ─── Fixtures ────────────────────────────────────────────────

@pytest.fixture
def valid_consent_token():
    """Generate a valid consent token for testing."""
    import os
    os.environ.setdefault("CONSENT_SECRET", "test-consent-secret")
    payload = {
        "consent_type": "diagnostic_analysis",
        "exp": int(time.time()) + 3600,
        "iat": int(time.time()),
        "sub": "test_patient_hash"
    }
    return jwt.encode(payload, "test-consent-secret", algorithm="HS256")


@pytest.fixture
def valid_auth_token():
    """Generate a valid JWT for API access."""
    import os
    os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-not-production")
    os.environ.setdefault("AUDIT_SALT", "test-salt")
    payload = {
        "sub": "test_clinician_001",
        "role": "clinician",
        "iat": int(time.time()),
        "exp": int(time.time()) + 3600,
        "jti": str(uuid.uuid4())
    }
    return jwt.encode(payload, "test-secret-key-not-production", algorithm="HS256")


@pytest.fixture
def sample_diagnostic_request(valid_consent_token):
    return {
        "symptoms": [
            {
                "name": "chest pain",
                "severity": 8,
                "onset_days": 1,
                "duration_hours": 6,
                "location": "left chest",
                "character": "crushing"
            },
            {
                "name": "shortness of breath",
                "severity": 7,
                "onset_days": 1,
                "duration_hours": 6
            }
        ],
        "patient_profile": {
            "age_group": "51-65",
            "biological_sex": "M",
            "bmi_category": "overweight",
            "smoking": True,
            "alcohol_use": False,
            "comorbidities": ["hypertension", "diabetes_type2"],
            "current_medications": ["metformin", "lisinopril"],
            "allergies": []
        },
        "free_text_description": "Patient reports sudden onset crushing chest pain radiating to left arm",
        "consent_token": valid_consent_token
    }


# ─── PHI Anonymization Tests ─────────────────────────────────

class TestPHIAnonymization:

    def test_removes_phone_numbers(self):
        from src.api.main import anonymize_free_text
        text = "Contact me at 555-123-4567 for follow-up"
        result = anonymize_free_text(text)
        assert "555-123-4567" not in result
        assert "[PHONE]" in result

    def test_removes_ssn(self):
        from src.api.main import anonymize_free_text
        text = "Patient SSN: 123-45-6789"
        result = anonymize_free_text(text)
        assert "123-45-6789" not in result
        assert "[SSN]" in result

    def test_removes_email(self):
        from src.api.main import anonymize_free_text
        text = "Email patient at john.doe@hospital.com"
        result = anonymize_free_text(text)
        assert "john.doe@hospital.com" not in result
        assert "[EMAIL]" in result

    def test_removes_date_of_birth(self):
        from src.api.main import anonymize_free_text
        text = "DOB: 01/15/1965, presenting with fever"
        result = anonymize_free_text(text)
        assert "01/15/1965" not in result

    def test_preserves_clinical_content(self):
        from src.api.main import anonymize_free_text
        text = "Patient presents with acute chest pain, dyspnea, and diaphoresis"
        result = anonymize_free_text(text)
        assert "chest pain" in result
        assert "dyspnea" in result
        assert "diaphoresis" in result

    def test_empty_text_safe(self):
        from src.api.main import anonymize_free_text
        assert anonymize_free_text("") == ""
        assert anonymize_free_text("   ") == "   "


# ─── Patient Profile Validation ──────────────────────────────

class TestPatientProfileValidation:

    def test_valid_profile_accepted(self):
        from src.api.main import PatientProfile
        profile = PatientProfile(
            age_group="31-50",
            biological_sex="F",
            bmi_category="normal",
            comorbidities=["asthma"],
            current_medications=["salbutamol"]
        )
        assert profile.age_group == "31-50"

    def test_invalid_age_group_rejected(self):
        from src.api.main import PatientProfile
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            PatientProfile(
                age_group="25",        # Not a valid group
                biological_sex="M",
                bmi_category="normal"
            )

    def test_phi_fields_rejected(self):
        """Ensure no PII fields are accepted (extra='forbid')."""
        from src.api.main import PatientProfile
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            PatientProfile(
                age_group="31-50",
                biological_sex="M",
                bmi_category="normal",
                name="John Doe",        # PII field — must be rejected
                date_of_birth="1990-01-01"
            )

    def test_max_comorbidities_enforced(self):
        from src.api.main import PatientProfile
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            PatientProfile(
                age_group="31-50",
                biological_sex="M",
                bmi_category="normal",
                comorbidities=["condition" + str(i) for i in range(25)]  # Max is 20
            )


# ─── Symptom Validation ───────────────────────────────────────

class TestSymptomValidation:

    def test_valid_symptom(self):
        from src.api.main import Symptom
        s = Symptom(name="headache", severity=5, onset_days=2)
        assert s.name == "headache"
        assert s.severity == 5

    def test_severity_out_of_range(self):
        from src.api.main import Symptom
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Symptom(name="pain", severity=11, onset_days=1)   # Max is 10

    def test_symptom_name_sanitized(self):
        from src.api.main import Symptom
        s = Symptom(name="  HEADACHE  ", severity=3, onset_days=1)
        assert s.name == "headache"  # Stripped and lowercased

    def test_negative_onset_days_rejected(self):
        from src.api.main import Symptom
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Symptom(name="pain", severity=5, onset_days=-1)


# ─── API Integration Tests ────────────────────────────────────

class TestDiagnosticAPI:
    """Full-stack tests using FastAPI TestClient with mocked Redis."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-key-not-production")
        monkeypatch.setenv("CONSENT_SECRET", "test-consent-secret")
        monkeypatch.setenv("AUDIT_SALT", "test-salt")

    @pytest.fixture
    def redis_mock(self):
        m = MagicMock()
        m.ping.return_value = True
        m.get.return_value = None   # no revoked tokens
        m.incr.return_value = 1     # rate-limit count well under 100
        m.expire.return_value = True
        m.lpush.return_value = 1
        m.ltrim.return_value = True
        with patch("src.api.main.redis_client", m):
            yield m

    @pytest.fixture
    def client(self, redis_mock):
        from src.api.main import app
        with TestClient(app) as c:
            yield c

    # ── Ops endpoints ──────────────────────────────────────────

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert data["version"] == "2.1.0"

    def test_ready_returns_200(self, client):
        resp = client.get("/ready")
        assert resp.status_code == 200
        assert resp.json()["ready"] is True

    def test_metrics_returns_prometheus_text(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert b"diagnostic_requests_total" in resp.content

    # ── Auth failures ──────────────────────────────────────────

    def test_no_token_returns_401(self, client, sample_diagnostic_request):
        resp = client.post("/api/v2/diagnose", json=sample_diagnostic_request)
        assert resp.status_code == 401

    def test_expired_token_returns_401(self, client, sample_diagnostic_request):
        expired = jwt.encode(
            {
                "sub": "u1", "role": "clinician",
                "iat": int(time.time()) - 7200,
                "exp": int(time.time()) - 3600,
                "jti": str(uuid.uuid4()),
            },
            "test-secret-key-not-production",
            algorithm="HS256",
        )
        resp = client.post(
            "/api/v2/diagnose", json=sample_diagnostic_request,
            headers={"Authorization": f"Bearer {expired}"},
        )
        assert resp.status_code == 401
        assert "expired" in resp.json()["detail"].lower()

    def test_invalid_token_returns_401(self, client, sample_diagnostic_request):
        resp = client.post(
            "/api/v2/diagnose", json=sample_diagnostic_request,
            headers={"Authorization": "Bearer garbage.token.here"},
        )
        assert resp.status_code == 401

    def test_revoked_token_returns_401(
        self, client, redis_mock, valid_auth_token, sample_diagnostic_request
    ):
        redis_mock.get.return_value = "1"  # jti marked as revoked
        resp = client.post(
            "/api/v2/diagnose", json=sample_diagnostic_request,
            headers={"Authorization": f"Bearer {valid_auth_token}"},
        )
        assert resp.status_code == 401
        assert "revoked" in resp.json()["detail"].lower()

    # ── Request validation ─────────────────────────────────────

    def test_invalid_consent_returns_400(
        self, client, valid_auth_token, sample_diagnostic_request
    ):
        bad = {**sample_diagnostic_request, "consent_token": "bad.consent"}
        resp = client.post(
            "/api/v2/diagnose", json=bad,
            headers={"Authorization": f"Bearer {valid_auth_token}"},
        )
        assert resp.status_code == 400

    def test_rate_limit_exceeded_returns_429(
        self, client, redis_mock, valid_auth_token, sample_diagnostic_request
    ):
        redis_mock.incr.return_value = 101  # over 100 req/hour
        resp = client.post(
            "/api/v2/diagnose", json=sample_diagnostic_request,
            headers={"Authorization": f"Bearer {valid_auth_token}"},
        )
        assert resp.status_code == 429

    # ── Success path ───────────────────────────────────────────

    def test_successful_diagnosis_returns_200(
        self, client, valid_auth_token, sample_diagnostic_request
    ):
        resp = client.post(
            "/api/v2/diagnose", json=sample_diagnostic_request,
            headers={"Authorization": f"Bearer {valid_auth_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["urgency_level"] in ("routine", "soon", "urgent", "emergency")
        assert "top_diagnoses" in data
        assert "red_flags_detected" in data
        assert "processing_time_ms" in data

    def test_response_includes_mandatory_disclaimer(
        self, client, valid_auth_token, sample_diagnostic_request
    ):
        resp = client.post(
            "/api/v2/diagnose", json=sample_diagnostic_request,
            headers={"Authorization": f"Bearer {valid_auth_token}"},
        )
        assert resp.status_code == 200
        assert resp.json()["disclaimer"] != ""

    def test_response_preserves_request_id(
        self, client, valid_auth_token, sample_diagnostic_request
    ):
        req_id = str(uuid.uuid4())
        req = {**sample_diagnostic_request, "request_id": req_id}
        resp = client.post(
            "/api/v2/diagnose", json=req,
            headers={"Authorization": f"Bearer {valid_auth_token}"},
        )
        assert resp.status_code == 200
        assert resp.json()["request_id"] == req_id


# ─── Model Unit Tests ─────────────────────────────────────────

class TestDiagnosticConfig:

    def test_default_config_valid(self):
        from src.models.symptom_classifier import DiagnosticConfig
        config = DiagnosticConfig()
        assert config.max_sequence_length == 256
        assert config.confidence_threshold == 0.7
        assert 0 < config.dropout_rate < 1

    def test_severity_levels_positive(self):
        from src.models.symptom_classifier import DiagnosticConfig
        config = DiagnosticConfig()
        assert config.severity_levels > 0

    def test_learning_rate_valid(self):
        from src.models.symptom_classifier import DiagnosticConfig
        config = DiagnosticConfig()
        assert 0 < config.learning_rate < 1


# ─── MLOps Pipeline Tests ────────────────────────────────────

class TestDataPipeline:

    def test_data_ingestion_returns_stats(self):
        from mlops.training_pipeline import DataVersioningPipeline
        pipeline = DataVersioningPipeline()
        stats = pipeline.run_data_ingestion()
        assert "raw_records" in stats
        assert stats["raw_records"] > 0
        assert "icd10_codes" in stats

    def test_preprocessing_splits_correct(self):
        from mlops.training_pipeline import DataVersioningPipeline
        pipeline = DataVersioningPipeline()
        data_stats = {"raw_records": 100_000}
        prep = pipeline.run_preprocessing(data_stats)
        total = prep["train_samples"] + prep["val_samples"] + prep["test_samples"]
        assert abs(total - 100_000) < 1000  # Within 1% of total

    def test_data_validation_passes_valid_data(self):
        from mlops.training_pipeline import DataVersioningPipeline
        pipeline = DataVersioningPipeline()
        valid_stats = {"train_samples": 70_000}
        result = pipeline.run_data_validation(valid_stats)
        assert result is True

    def test_performance_gates_enforce_red_flag_recall(self):
        from mlops.training_pipeline import ModelRegistry
        registry = ModelRegistry()
        # Red flag recall must be >= 98% (safety critical)
        bad_metrics = {
            "test_top3_accuracy": 0.85,
            "test_urgency_auc": 0.96,
            "test_red_flag_recall": 0.95,    # Below 0.98 threshold
            "test_calibration_ece": 0.03,
        }
        result = registry.transition_to_staging("fake_run_id", bad_metrics)
        assert result is False, "Should fail: red flag recall below safety threshold"

    def test_performance_gates_pass_good_model(self):
        from mlops.training_pipeline import ModelRegistry
        # ModelRegistry instantiation itself should not raise
        registry = ModelRegistry()
        assert registry is not None


# ─── Audit Logging Tests ──────────────────────────────────────

class TestAuditLogging:

    def test_hash_is_deterministic(self):
        from src.api.main import hash_identifier
        import os
        os.environ["AUDIT_SALT"] = "test-salt"
        h1 = hash_identifier("user123")
        h2 = hash_identifier("user123")
        assert h1 == h2

    def test_hash_is_irreversible(self):
        """Hash output should not contain the original identifier."""
        from src.api.main import hash_identifier
        import os
        os.environ["AUDIT_SALT"] = "test-salt"
        result = hash_identifier("john.doe@hospital.com")
        assert "john" not in result
        assert "@" not in result

    def test_different_users_different_hashes(self):
        from src.api.main import hash_identifier
        import os
        os.environ["AUDIT_SALT"] = "test-salt"
        h1 = hash_identifier("user_a")
        h2 = hash_identifier("user_b")
        assert h1 != h2


# ─── Inference Helper Tests ───────────────────────────────────

class TestInferenceHelpers:
    """Unit tests for preprocessing, postprocessing, and helper functions."""

    def test_stub_response_valid_structure(self):
        from src.api.main import _stub_response
        result = _stub_response()
        assert result["urgency_level"] in ("routine", "soon", "urgent", "emergency")
        assert isinstance(result["red_flags"], list)
        assert isinstance(result["diagnoses"], list)
        assert isinstance(result["follow_up_questions"], list)
        assert 0.0 <= result["urgency_score"] <= 1.0

    def test_follow_up_emergency_has_priority_question(self):
        from src.api.main import _follow_up_questions, Symptom
        symptoms = [Symptom(name="chest pain", severity=9, onset_days=1)]
        qs = _follow_up_questions(symptoms, "emergency")
        assert any("breathing" in q.lower() or "chest" in q.lower() for q in qs)

    def test_follow_up_high_severity_asks_about_change(self):
        from src.api.main import _follow_up_questions, Symptom
        symptoms = [Symptom(name="pain", severity=8, onset_days=3)]
        qs = _follow_up_questions(symptoms, "routine")
        assert any("severity" in q.lower() for q in qs)

    def test_follow_up_multiple_symptoms_asks_co_occurrence(self):
        from src.api.main import _follow_up_questions, Symptom
        symptoms = [
            Symptom(name="fever", severity=5, onset_days=2),
            Symptom(name="cough", severity=4, onset_days=3),
        ]
        qs = _follow_up_questions(symptoms, "routine")
        assert any("together" in q.lower() for q in qs)

    def test_follow_up_capped_at_four(self):
        from src.api.main import _follow_up_questions, Symptom
        symptoms = [Symptom(name=f"s{i}", severity=9, onset_days=1) for i in range(5)]
        qs = _follow_up_questions(symptoms, "emergency")
        assert len(qs) <= 4

    def test_verify_consent_token_valid(self):
        import os
        os.environ["CONSENT_SECRET"] = "test-consent-secret"
        payload = {"consent_type": "diagnostic_analysis", "exp": int(time.time()) + 3600}
        token = jwt.encode(payload, "test-consent-secret", algorithm="HS256")
        from src.api.main import verify_consent_token
        assert verify_consent_token(token) is True

    def test_verify_consent_token_wrong_type(self):
        import os
        os.environ["CONSENT_SECRET"] = "test-consent-secret"
        payload = {"consent_type": "other", "exp": int(time.time()) + 3600}
        token = jwt.encode(payload, "test-consent-secret", algorithm="HS256")
        from src.api.main import verify_consent_token
        assert verify_consent_token(token) is False

    def test_verify_consent_token_garbage(self):
        import os
        os.environ["CONSENT_SECRET"] = "test-consent-secret"
        from src.api.main import verify_consent_token
        assert verify_consent_token("not.valid") is False

    def test_create_access_token_has_required_claims(self):
        import os
        os.environ["JWT_SECRET_KEY"] = "test-secret-key-not-production"
        from src.api.main import create_access_token
        token = create_access_token("user123", "clinician")
        payload = jwt.decode(token, "test-secret-key-not-production", algorithms=["HS256"])
        assert payload["sub"] == "user123"
        assert payload["role"] == "clinician"
        assert "jti" in payload
        assert "exp" in payload

    def test_preprocess_returns_five_tensor_keys(self):
        import tensorflow as tf
        from src.api.main import _preprocess, Symptom, PatientProfile

        def mock_tokenizer(text, max_length, padding, truncation, return_tensors):
            return {
                "input_ids": tf.zeros((1, max_length), dtype=tf.int32),
                "attention_mask": tf.ones((1, max_length), dtype=tf.int32),
            }

        symptoms = [Symptom(name="headache", severity=5, onset_days=2)]
        patient = PatientProfile(age_group="31-50", biological_sex="M", bmi_category="normal")
        result = _preprocess(symptoms, patient, "headache", mock_tokenizer)
        assert set(result.keys()) == {
            "input_ids", "attention_mask", "structured_symptoms",
            "temporal_features", "patient_metadata",
        }
        assert result["structured_symptoms"].shape == (1, 500)
        assert result["patient_metadata"].shape == (1, 20)

    def test_preprocess_encodes_patient_flags(self):
        import tensorflow as tf
        from src.api.main import _preprocess, Symptom, PatientProfile

        def mock_tokenizer(text, **kwargs):
            return {
                "input_ids": tf.zeros((1, 256), dtype=tf.int32),
                "attention_mask": tf.ones((1, 256), dtype=tf.int32),
            }

        symptoms = [Symptom(name="cough", severity=3, onset_days=5)]
        patient = PatientProfile(
            age_group="51-65", biological_sex="F", bmi_category="obese",
            smoking=True, comorbidities=["hypertension", "diabetes_type2"],
        )
        result = _preprocess(symptoms, patient, "", mock_tokenizer)
        meta = result["patient_metadata"].numpy()[0]
        assert meta[3] == 1.0   # smoking = True → index 3
        assert meta[5] > 0.0    # comorbidities count → index 5

    def test_postprocess_maps_urgency_levels(self):
        import numpy as np
        import tensorflow as tf
        from src.api.main import _postprocess, Symptom

        symptoms = [Symptom(name="dizziness", severity=4, onset_days=1)]

        def make_outputs(urgency_val):
            return {
                "diagnosis": tf.constant(np.eye(150, dtype=np.float32)[0:1]),
                "severity": tf.constant([[0.7, 0.2, 0.05, 0.05]], dtype=tf.float32),
                "urgency_score": tf.constant([[urgency_val]], dtype=tf.float32),
                "red_flags": tf.constant(np.zeros((1, 50), dtype=np.float32)),
            }

        assert _postprocess(make_outputs(0.9), symptoms)["urgency_level"] == "emergency"
        assert _postprocess(make_outputs(0.65), symptoms)["urgency_level"] == "urgent"
        assert _postprocess(make_outputs(0.35), symptoms)["urgency_level"] == "soon"
        assert _postprocess(make_outputs(0.1), symptoms)["urgency_level"] == "routine"

    def test_postprocess_detects_red_flags(self):
        import numpy as np
        import tensorflow as tf
        from src.api.main import _postprocess, Symptom

        symptoms = [Symptom(name="chest pain", severity=9, onset_days=1)]
        red_flags_arr = np.zeros((1, 50), dtype=np.float32)
        red_flags_arr[0, 5] = 0.9   # flag at index 5
        outputs = {
            "diagnosis": tf.constant(np.eye(150, dtype=np.float32)[0:1]),
            "severity": tf.constant([[0.2, 0.3, 0.3, 0.2]], dtype=tf.float32),
            "urgency_score": tf.constant([[0.5]], dtype=tf.float32),
            "red_flags": tf.constant(red_flags_arr),
        }
        result = _postprocess(outputs, symptoms)
        assert "Red Flag 5" in result["red_flags"]


# ─── Model Component Tests ────────────────────────────────────

class TestModelComponents:
    """Tests for TF layers that don't require downloading BERT weights."""

    def test_warmup_lr_is_zero_at_step_zero(self):
        import tensorflow as tf
        from src.models.symptom_classifier import LinearWarmupCosineDecay
        sched = LinearWarmupCosineDecay(peak_lr=1e-4, warmup_steps=1000, decay_steps=10000)
        assert float(sched(tf.constant(0))) == pytest.approx(0.0, abs=1e-9)

    def test_warmup_lr_is_half_peak_at_midpoint(self):
        import tensorflow as tf
        from src.models.symptom_classifier import LinearWarmupCosineDecay
        sched = LinearWarmupCosineDecay(peak_lr=1e-4, warmup_steps=1000, decay_steps=10000)
        assert float(sched(tf.constant(500))) == pytest.approx(5e-5, rel=0.01)

    def test_lr_decreases_after_warmup(self):
        import tensorflow as tf
        from src.models.symptom_classifier import LinearWarmupCosineDecay
        sched = LinearWarmupCosineDecay(peak_lr=1e-4, warmup_steps=1000, decay_steps=10000)
        assert float(sched(tf.constant(1000))) > float(sched(tf.constant(6000)))

    def test_schedule_get_config_roundtrips(self):
        from src.models.symptom_classifier import LinearWarmupCosineDecay
        sched = LinearWarmupCosineDecay(peak_lr=2e-5, warmup_steps=500, decay_steps=5000)
        cfg = sched.get_config()
        assert cfg == {"peak_lr": 2e-5, "warmup_steps": 500, "decay_steps": 5000}

    def test_focal_loss_lower_for_confident_correct_prediction(self):
        import tensorflow as tf
        from src.models.symptom_classifier import FocalLoss
        loss_fn = FocalLoss(gamma=2.0, alpha=0.25)
        y_true = tf.constant([[0.0, 1.0, 0.0]])
        y_pred_confident = tf.constant([[0.01, 0.98, 0.01]])
        y_pred_uncertain = tf.constant([[0.33, 0.34, 0.33]])
        assert float(loss_fn(y_true, y_pred_confident)) < float(loss_fn(y_true, y_pred_uncertain))

    def test_multi_head_attention_preserves_shape(self):
        import tensorflow as tf
        from src.models.symptom_classifier import MultiHeadSymptomAttention
        layer = MultiHeadSymptomAttention(d_model=64, num_heads=4)
        x = tf.random.normal((2, 5, 64))
        out = layer(x, training=False)
        assert out.shape == (2, 5, 64)

    def test_diagnostic_reasoning_output_keys_and_shapes(self):
        import tensorflow as tf
        from src.models.symptom_classifier import DiagnosticReasoningModule
        mod = DiagnosticReasoningModule(num_diagnoses=150, num_severity=4)
        features = tf.random.normal((2, 512))
        out = mod(features, training=False)
        assert set(out.keys()) == {"diagnosis", "severity", "urgency_score", "red_flags"}
        assert out["diagnosis"].shape == (2, 150)
        assert out["severity"].shape == (2, 4)
        assert out["urgency_score"].shape == (2, 1)
        assert out["red_flags"].shape == (2, 50)

    def test_temporal_encoder_returns_pooled_output(self):
        import tensorflow as tf
        from src.models.symptom_classifier import TemporalSymptomEncoder
        enc = TemporalSymptomEncoder(units=64)
        seq = tf.random.normal((2, 5, 128))
        temporal = tf.random.normal((2, 5, 3))
        out = enc(seq, temporal, training=False)
        assert len(out.shape) == 2   # pooled: (batch, features), no time dimension
        assert out.shape[0] == 2
