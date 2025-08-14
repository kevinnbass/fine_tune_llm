# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture Overview

This is a production-ready bird flu classification ensemble system implementing defense-in-depth ML with cascaded arbitration, conformal abstention, and active learning. The system operates as a 4-tier cascade:

1. **Tier 0: Regex DSL** (`dsl/engine.py`) - YAML-based deterministic rules ($0.0001/prediction)
2. **Tier 1: Classical Models** (`voters/classical/`) - Calibrated TF-IDF+LR/SVM (target ECE ≤ 0.03)  
3. **Tier 2: Weak Supervision** (`voters/ws_label_model/`) - Inference-time label model voter
4. **Tier 3: LLM LoRA** (`voters/llm/`) - Qwen2.5-7B with structured JSON output ($0.05/prediction)

The **Arbiter Cascade** (`arbiter/`) combines voter outputs through consensus detection, learned stacking, and conformal prediction for risk-controlled abstention (target ≤1% error at ≥85% coverage).

## Development Commands

### Environment Setup
```bash
# Install for development (includes all tools)
make install-dev

# Basic installation
make install
```

### Code Quality
```bash
# Format code (black + isort)
make format

# Run all linting (flake8, mypy, black check, isort check)
make lint

# Run tests with coverage
make test

# Run safety regression tests only
make test-safety
```

### Training Pipeline
```bash
# Complete training pipeline (runs all steps in order)
make train-all

# Individual training steps:
make prepare-data           # Data preparation
make train-weak-supervision # Labeling functions + label model
make train-classical        # TF-IDF models with calibration
make train-lora            # LoRA fine-tuning on Qwen2.5-7B
make train-stacker         # Arbiter stacker with out-of-fold predictions
make set-thresholds        # Conformal prediction thresholds
```

### Evaluation & Active Learning
```bash
# Complete system evaluation
make eval-all

# Generate disagreement samples for human review
make generate-disagreements

# Start audit UI for reviewing samples
make serve
```

### Production Deployment
```bash
# Start full production stack (API, database, monitoring, caching)
docker-compose up -d

# Start API server only
python api/server.py

# Start monitoring dashboard
streamlit run dashboard/app.py
```

## Key Configuration Files

All configurations are in `configs/` directory:

- **`labels.yaml`**: Class definitions with harm weights for cost-sensitive evaluation
- **`voters.yaml`**: Voter enable/disable flags, cost parameters, and model paths
- **`conformal.yaml`**: Risk targets, coverage goals, and per-slice threshold overrides
- **`llm_lora.yaml`**: LoRA training hyperparameters (r=16, alpha=32, learning rates)
- **`slices.yaml`**: Slice definitions for performance monitoring (source, length, etc.)

## Performance Targets & Monitoring

The system targets enterprise-grade performance:

- **Overall F1**: ≥0.92
- **Worst-slice F1**: ≥0.85  
- **ECE**: ≤0.03 (calibrated models)
- **LLM call rate**: ≤10% (cost optimization)
- **Abstention rate**: ≤15%
- **P95 latency**: ≤200ms via tiered cascade
- **Cost per prediction**: <$0.01 average

## Critical Implementation Details

### Out-of-Fold Training (`arbiter/stacker_lr.py`)
The stacker uses out-of-fold predictions to prevent leakage. Never train the stacker on the same data used to train the voters - this is enforced by the OOF collection process.

### Conformal Prediction (`arbiter/conformal.py`)
Uses split conformal prediction with per-slice threshold overrides. The system calibrates thresholds on a held-out set to achieve coverage guarantees. Per-slice thresholds handle performance disparities across data segments.

### Cost-Aware Cascade (`arbiter/features.py`)
The cascade makes early exits when possible:
- Tier 0: Fast consensus detection (all voters agree)
- Safety rules trigger immediate abstention
- High-confidence predictions bypass expensive models

### LLM Integration (`voters/llm/`)
- Uses PEFT with LoRA adapters for efficient fine-tuning
- Enforces JSON schema validation with explicit abstention support
- Extracts logprobs for calibration
- Falls back to abstention on invalid JSON output

## Production Infrastructure

### Database (`database/models.py`)
PostgreSQL with SQLAlchemy ORM tracking:
- Prediction records with voter outputs and metadata
- Human feedback with error type classification  
- Model performance metrics over time
- Data drift detection results

### Caching (`cache/prediction_cache.py`)
Multi-level Redis caching with:
- LRU eviction and TTL management
- Cache warming for common patterns
- Comprehensive hit/miss statistics

### Monitoring (`monitoring/metrics.py`)
Comprehensive observability stack:
- Prometheus metrics export
- OpenTelemetry distributed tracing
- Structured logging with correlation IDs
- Real-time alerting on performance degradation

### API Server (`api/server.py`)
Production FastAPI server with:
- Bearer token authentication
- Rate limiting per client
- Async request handling with connection pooling
- Health checks and graceful shutdown
- Comprehensive error handling with retry logic

## Testing Strategy

The test suite (`tests/`) includes:
- **Unit tests**: Individual component testing with mocks
- **Integration tests**: End-to-end voter compatibility  
- **Safety regression**: Frozen safety set evaluation to prevent regressions
- **Performance tests**: Latency and throughput validation

Use `conftest.py` fixtures for consistent test data including sample texts, labels, voter outputs, and metadata.

## Active Learning Workflow

The **Disagreement Miner** (`miner/disagreement.py`) identifies high-value samples for human review:
- High entropy predictions (uncertainty)
- Vote splits between models
- Rare slice coverage gaps
- Novel failure modes

Weekly review process (50-100 samples):
1. Run `make generate-disagreements` 
2. Use audit UI to label samples
3. Extract new DSL rules from patterns
4. Add gold labels for model retraining
5. Update prompts based on failure analysis

## Container Orchestration

The `docker-compose.yml` defines a complete production stack:
- **API service**: Main application with model serving
- **Redis**: Caching layer with persistence
- **PostgreSQL**: Primary database with MLflow backend
- **Prometheus/Grafana**: Metrics and dashboards
- **OpenTelemetry**: Distributed tracing
- **MLflow**: Experiment tracking and model registry
- **Nginx**: Reverse proxy with SSL termination

Environment variables control configuration - see `.env.example` for required settings.

## Deployment Checklist

Before production deployment:
1. Train all models with sufficient data coverage
2. Calibrate conformal thresholds on held-out set
3. Validate safety set performance meets requirements
4. Configure monitoring alerts and dashboards
5. Set up backup procedures for models and data
6. Test disaster recovery mechanisms
7. Perform load testing at expected traffic volumes

## Git Automation Setup

To enable automatic git operations after task completion, configure Claude Code hooks in your settings to run:
```bash
git add .
git commit -m "$(generate_commit_message_for_task)"
git push origin main
```

This ensures all work is automatically committed and pushed to maintain development momentum.