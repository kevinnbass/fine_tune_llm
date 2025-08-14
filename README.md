# Bird Flu Classification Ensemble System

A production-ready defense-in-depth ML system for bird flu content classification with cascaded arbitration, conformal abstention, and active learning.

## 🏗️ Architecture

```
Input Text → [Regex DSL] → [Classical ML] → [Weak Supervision] → [LLM LoRA] 
                ↓              ↓               ↓                ↓
            [Arbiter Cascade: Consensus → Stacking → Conformal Abstention]
                ↓
            [Prediction / ABSTAIN]
                ↓
            [Disagreement Mining → Active Learning Loop]
```

## 📊 Performance Targets

| Metric | Target | Status |
|--------|--------|---------|
| Overall F1 | ≥0.92 | 🎯 |
| Worst-slice F1 | ≥0.85 | 🎯 |
| Calibration (ECE) | ≤0.03 | 🎯 |
| P95 Latency | ≤200ms | ⚡ |
| Cost per prediction | <$0.01 | 💰 |
| LLM call rate | ≤10% | 🔄 |
| Coverage | ≥85% @ ≤1% error | 🛡️ |

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/kevinnbass/fine_tune_llm.git
cd fine_tune_llm/birdflu-ensemble

# Install dependencies
make install-dev

# Verify installation
make test
```

### Training Pipeline

```bash
# Complete training workflow
make train-all

# Or run individual steps:
make prepare-data
make train-weak-supervision
make train-classical  
make train-lora
make train-stacker
make set-thresholds
```

### Evaluation

```bash
# System evaluation
make eval-all

# Safety regression testing
make test-safety

# Generate samples for human review
make generate-disagreements
```

### Production Deployment

```bash
# Start full production stack
docker-compose up -d

# Access services:
# - API: http://localhost:8000
# - Grafana: http://localhost:3000
# - MLflow: http://localhost:5000
# - Dashboard: streamlit run dashboard/app.py
```

## 🧠 System Components

### Tier 0: Regex DSL Engine
- **Location**: `dsl/engine.py`
- **Cost**: $0.0001/prediction
- **Purpose**: Fast deterministic rules with safety triggers
- **Features**: YAML configuration, priority-based evaluation

### Tier 1: Classical Models
- **Location**: `voters/classical/`
- **Cost**: $0.001/prediction  
- **Models**: TF-IDF + Logistic Regression, TF-IDF + SVM
- **Features**: Platt/Isotonic calibration, target ECE ≤ 0.03

### Tier 2: Weak Supervision
- **Location**: `voters/ws_label_model/`
- **Cost**: $0.002/prediction
- **Purpose**: Inference-time probabilistic label model
- **Features**: Conflict resolution, minimum evidence thresholds

### Tier 3: LLM LoRA
- **Location**: `voters/llm/`
- **Cost**: $0.05/prediction
- **Model**: Qwen2.5-7B with LoRA adapters
- **Features**: JSON schema validation, explicit abstention

### Arbiter Cascade
- **Location**: `arbiter/`
- **Tier 0**: Fast consensus detection and safety rules
- **Tier 1**: Learned stacking with out-of-fold training
- **Tier 2**: Conformal prediction for risk-controlled abstention

## 🔧 Configuration

All configurations in `configs/` directory:

| File | Purpose |
|------|---------|
| `labels.yaml` | Class definitions and harm weights |
| `voters.yaml` | Voter enable/disable flags and parameters |
| `conformal.yaml` | Risk targets and coverage goals |
| `llm_lora.yaml` | LoRA training hyperparameters |
| `slices.yaml` | Performance monitoring segments |

## 🏭 Production Infrastructure

### Core Services
- **FastAPI Server**: Authentication, rate limiting, async handling
- **PostgreSQL**: Prediction storage, feedback tracking, performance metrics
- **Redis**: Multi-level caching with LRU and TTL management
- **MLflow**: Experiment tracking and model registry

### Monitoring Stack
- **Prometheus**: Metrics collection and alerting
- **Grafana**: Real-time dashboards and visualization
- **OpenTelemetry**: Distributed tracing and observability
- **Streamlit Dashboard**: System health and performance monitoring

### Development Tools
- **Docker**: Containerization with multi-stage builds
- **pytest**: Comprehensive test suite with coverage
- **CI/CD**: Rule validation, safety regression, promotion gates
- **Code Quality**: black, isort, flake8, mypy

## 📈 Active Learning

Weekly human review process:

1. **Generate Disagreements**: `make generate-disagreements`
2. **Audit UI**: `make serve` → Review 50-100 samples
3. **Extract Patterns**: New DSL rules from failure modes
4. **Update Models**: Gold labels for retraining
5. **Improve Prompts**: Based on failure analysis

## 🔬 Key Features

### Conformal Prediction
- Split conformal thresholds for coverage guarantees
- Per-slice threshold overrides for critical segments
- Risk-controlled abstention (≤1% error @ ≥85% coverage)

### Cost Optimization
- Tiered cascade with early exits
- Consensus detection bypasses expensive models
- Target: <$0.01 average cost per prediction

### Calibration
- Platt scaling, isotonic regression, temperature scaling
- Expected Calibration Error (ECE) monitoring
- Brier score tracking for probabilistic accuracy

### Safety & Security
- Input sanitization and validation
- Rate limiting and authentication
- Backup and disaster recovery procedures
- Comprehensive error handling with graceful degradation

## 📊 Monitoring & Alerting

Real-time tracking of:
- **Performance**: F1, precision, recall per slice
- **Calibration**: ECE, Brier scores, reliability diagrams  
- **System Health**: Latency, throughput, error rates
- **Business Metrics**: Cost per prediction, abstention rates
- **Data Quality**: Drift detection, distribution shifts

## 🔗 API Usage

```python
import requests

# Make prediction
response = requests.post(
    "http://localhost:8000/predict",
    headers={"Authorization": "Bearer your-token"},
    json={"text": "Sample bird flu content..."}
)

result = response.json()
# {
#   "prediction": "high_risk", 
#   "confidence": 0.92,
#   "abstain": false,
#   "voter_outputs": {...},
#   "latency_ms": 145
# }
```

## 🧪 Testing

```bash
# Full test suite
make test

# Safety regression only
make test-safety

# Lint and format
make lint
make format

# Clean artifacts
make clean
```

## 📋 Requirements

- **Python**: ≥3.9
- **GPU**: Recommended for LLM training (8GB+ VRAM)
- **Memory**: 16GB+ RAM for full system
- **Storage**: 50GB+ for models and artifacts

## 📚 Documentation

- **Development Guide**: See `CLAUDE.md` for detailed guidance
- **API Documentation**: Available at `/docs` when server running
- **Configuration Reference**: Comments in `configs/*.yaml` files
- **Deployment Guide**: Docker and production setup instructions

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes following code quality standards
4. Run tests: `make test && make lint`
5. Submit pull request with clear description

## 📄 License

[MIT License](LICENSE) - see LICENSE file for details.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/kevinnbass/fine_tune_llm/issues)
- **Discussions**: [GitHub Discussions](https://github.com/kevinnbass/fine_tune_llm/discussions)
- **Documentation**: See `CLAUDE.md` for technical details

---

**Built with**: PyTorch, Transformers, PEFT, FastAPI, PostgreSQL, Redis, Prometheus, Grafana, Docker

**Enterprise-ready**: Production monitoring, security, scalability, and reliability built-in.