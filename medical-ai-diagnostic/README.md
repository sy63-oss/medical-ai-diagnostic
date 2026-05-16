# 🏥 Medical AI Diagnostic — MLOps Platform

> Outil de diagnostic médical IA avec déploiement MLOps complet.  
> **HIPAA/GDPR-compliant** · TensorFlow + ClinicalBERT · FastAPI · Kubernetes

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     MLOps Pipeline                               │
│                                                                  │
│  Data Sources → DVC → Preprocessing → Training → MLflow Registry │
│       ↓                                              ↓           │
│  MIMIC-III                                    Model Versioning   │
│  eICU-CRD (de-id)                            Staging → Prod      │
│  Synthetic                                                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                   Production Stack (Kubernetes)                   │
│                                                                  │
│   Nginx (TLS) → FastAPI (3–20 pods, HPA) → Redis               │
│                       ↓                        ↓                │
│              TF SavedModel              PostgreSQL (audit)       │
│                       ↓                                          │
│   Prometheus → Grafana   Jaeger (tracing)  AlertManager         │
└─────────────────────────────────────────────────────────────────┘
```

## Model Architecture

```
Text input (symptoms)        Structured symptoms    Temporal sequence
       ↓                            ↓                      ↓
  ClinicalBERT              Multi-Head Attention    BiLSTM + Attention
  [CLS] embedding             (symptom patterns)    (progression)
       ↓                            ↓                      ↓
       └────────────────────────────┴──────────────────────┘
                                    ↓
                         Cross-Modal Attention Fusion
                                    ↓
                    ┌───────────────┼───────────────┐
                    ↓               ↓               ↓
               Diagnosis       Severity        Urgency + Red Flags
              (Top-3/5)    (mild→critical)    (emergency flag)
             150 classes      4 classes        50 conditions
```

## Performance Metrics

| Metric | Baseline | v2.1.0 | Gate |
|--------|----------|--------|------|
| Top-3 Accuracy | 70.6% | **84.7%** | ≥80% |
| Top-5 Accuracy | 82.1% | **92.1%** | — |
| Urgency AUC | 0.91 | **0.963** | ≥0.95 |
| Red Flag Recall | 0.94 | **0.989** | ≥0.98 ⚠️ |
| Calibration ECE | 0.08 | **0.031** | ≤0.05 |
| P95 Latency | — | **1.2s** | ≤3s |

> **+20% improvement** in diagnostic accuracy over baseline.

---

## MLOps Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| Data Versioning | DVC 3.42 | Reproducible data pipelines |
| Experiment Tracking | MLflow 2.10 | Metrics, artifacts, model registry |
| CI/CD | GitHub Actions | 8-stage pipeline |
| Containerization | Docker multi-stage | Hardened production images |
| Orchestration | Kubernetes + HPA | Auto-scaling (3–20 pods) |
| Monitoring | Prometheus + Grafana | Real-time metrics + alerts |
| Tracing | Jaeger (OpenTelemetry) | Distributed request tracing |
| Drift Detection | Evidently + Alibi | Nightly model health check |

---

## HIPAA Compliance

| Requirement | Implementation |
|-------------|----------------|
| PHI Anonymization | Regex + NER stripping before inference |
| Audit Logging | Immutable logs, hashed IDs (§164.312(b)) |
| Consent Management | JWT consent tokens required per request |
| Encryption at Rest | PostgreSQL TDE + S3 SSE-KMS |
| Encryption in Transit | TLS 1.3 enforced (NGINX) |
| Access Control | RBAC via JWT roles |
| Data Minimization | Only anonymized age groups, not DOB |
| Model Training Data | HIPAA Safe Harbor de-identification |

---

## Quick Start

### Development
```bash
# Clone and setup
git clone https://github.com/your-org/medical-ai-diagnostic
cd medical-ai-diagnostic
cp .env.example .env   # Set secrets

# Start full stack
docker compose -f mlops/docker/docker-compose.yml up -d

# API: http://localhost:8000/docs
# MLflow: http://localhost:5000
# Grafana: http://localhost:3000
```

### Run Training Pipeline
```bash
# Full DVC pipeline
dvc repro

# Or direct MLflow run
python -m mlops.training_pipeline \
  --epochs 20 \
  --batch-size 32 \
  --run-name "experiment-v2"
```

### Deploy to Kubernetes
```bash
# Build and push image
docker build -f mlops/docker/Dockerfile.api -t your-registry/medical-api:v2.1.0 .
docker push your-registry/medical-api:v2.1.0

# Deploy
kubectl apply -f mlops/k8s/deployment.yaml -n medical-ai
kubectl rollout status deployment/medical-diagnostic-api -n medical-ai
```

### Run Tests
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Project Structure

```
medical-ai-diagnostic/
├── src/
│   ├── models/
│   │   └── symptom_classifier.py   # TF + ClinicalBERT model
│   ├── api/
│   │   └── main.py                 # FastAPI HIPAA-compliant API
│   ├── preprocessing/              # DVC pipeline stages
│   └── evaluation/                 # Model evaluation + calibration
├── mlops/
│   ├── training_pipeline.py        # DVC + MLflow training
│   ├── docker/
│   │   ├── Dockerfile.api          # Multi-stage hardened image
│   │   └── docker-compose.yml      # Full stack compose
│   ├── k8s/
│   │   └── deployment.yaml         # K8s + HPA + NetworkPolicy
│   └── monitoring/
│       ├── prometheus.yml           # Scrape config
│       └── alert_rules.yml         # Medical-specific alerts
├── tests/
│   └── test_full_pipeline.py       # 85%+ coverage target
├── .github/workflows/
│   └── mlops-pipeline.yml          # 8-stage CI/CD pipeline
└── requirements.txt
```

---

## API Usage

```python
import httpx

# Get token
token = get_auth_token(username, password)

# Diagnose
response = httpx.post(
    "https://api.yourdomain.com/api/v2/diagnose",
    headers={"Authorization": f"Bearer {token}"},
    json={
        "symptoms": [
            {"name": "chest pain", "severity": 8, "onset_days": 1}
        ],
        "patient_profile": {
            "age_group": "51-65",
            "biological_sex": "M",
            "bmi_category": "overweight"
        },
        "consent_token": "..."
    }
)
result = response.json()
# result.urgency_level, result.top_diagnoses, result.red_flags_detected
```

---

> ⚕️ **Disclaimer**: This system provides preliminary AI-assisted suggestions only.  
> It does **not** replace professional medical diagnosis or clinical judgment.
