# Bird Flu Classification Ensemble System

A defense-in-depth classification system for bird flu content with cascaded arbitration, conformal abstention, and active learning.

## Architecture Overview

```
Input → [Regex DSL] → [Classical Models] → [Label Model] → [LLM LoRA] 
         ↓              ↓                    ↓               ↓
      [Arbiter Cascade: Consensus → Stacker → Conformal Abstention]
         ↓
      [Output / ABSTAIN]
         ↓
      [Disagreement Miner → Active Learning Loop]
```

## Key Components

### 1. Voters (Ordered by Cost)

1. **Regex DSL** (`dsl/engine.py`)
   - YAML-based rules with priorities
   - Immediate abstention for safety rules
   - Cost: $0.0001/prediction

2. **Classical Models** (`voters/classical/`)
   - TF-IDF + Logistic Regression
   - TF-IDF + SVM
   - Calibrated with Platt/Isotonic scaling
   - Target ECE ≤ 0.03
   - Cost: $0.001/prediction

3. **Weak Supervision Label Model** (`voters/ws_label_model/`)
   - Inference-time probabilistic voter
   - Combines labeling function outputs
   - Cost: $0.002/prediction

4. **LLM with LoRA** (`voters/llm/`)
   - Qwen2.5-7B base model (recommended)
   - JSON schema validation
   - Explicit abstention support
   - Cost: $0.05/prediction

### 2. Arbiter Cascade

**Tier 0: Fast Rules**
- Consensus detection (all voters agree)
- Safety rule enforcement
- High-confidence fast acceptance

**Tier 1: Learned Stacker**
- Logistic regression or XGBoost
- Features: voter probs, entropy, margin, metadata
- Out-of-fold training to prevent leakage

**Tier 2: Conformal Abstention**
- Split conformal prediction
- Target ≤1% error at ≥85% coverage
- Per-slice thresholds available

### 3. Active Learning

- **Disagreement Miner** (`miner/disagreement.py`)
  - High entropy samples
  - Vote splits
  - Rare slice coverage
  
- **Weekly Review** (50-100 samples)
  - New DSL rules
  - Gold labels for retraining
  - Prompt improvements

## Quick Start

### Installation

```bash
# Clone repo
git clone <repo-url>
cd birdflu-ensemble

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"
```

### Basic Training Pipeline

```bash
# 1. Prepare data
make prepare-data

# 2. Train all models
make train-all

# 3. Evaluate system
make eval-all

# 4. Generate disagreements for review
make generate-disagreements
```

### Configuration

All configs in `configs/` directory:
- `labels.yaml`: Class definitions and harm weights
- `slices.yaml`: Slice definitions for evaluation
- `voters.yaml`: Voter enable/disable and parameters
- `conformal.yaml`: Risk targets and coverage goals
- `llm_lora.yaml`: LoRA training hyperparameters

## Implementation Status

### ✅ Completed Components

1. **Calibration Utilities** (`voters/classical/calibrate.py`)
   - Platt scaling
   - Isotonic regression
   - Temperature scaling
   - ECE/Brier computation

2. **Regex DSL Engine** (`dsl/engine.py`)
   - YAML rule loading
   - Priority-based evaluation
   - Combination rules
   - Abstention triggers

3. **Weak Supervision Voter** (`voters/ws_label_model/snorkel_like.py`)
   - Probabilistic label model
   - Conflict resolution
   - Minimum evidence thresholds

4. **LLM Dataset Builder** (`voters/llm/dataset.py`)
   - JSON schema formatting
   - Abstention examples
   - Balanced dataset creation

### 🚧 Components to Implement

#### 1. LLM LoRA Training (`voters/llm/sft_lora.py`)
```python
# Key functionality needed:
- Load Qwen2.5-7B with PEFT
- LoRA configuration (r=16, alpha=32)
- Training loop with early stopping
- Checkpoint saving
```

#### 2. LLM Inference (`voters/llm/infer.py`)
```python
# Key functionality needed:
- Load model + LoRA adapter
- Generate with JSON constraints
- Schema validation
- Logprob extraction for calibration
- ABSTAIN on invalid output
```

#### 3. Arbiter Stacker (`arbiter/stacker_lr.py`)
```python
# Key functionality needed:
- Feature engineering from voter outputs
- Out-of-fold prediction collection
- Logistic regression training
- Calibration of stacker outputs
```

#### 4. Conformal Prediction (`arbiter/conformal.py`)
```python
# Key functionality needed:
- Split conformal threshold selection
- Per-slice threshold overrides
- Coverage monitoring
- Risk-controlled abstention
```

#### 5. Disagreement Miner (`miner/disagreement.py`)
```python
# Key functionality needed:
- Entropy computation
- Vote pattern analysis
- Slice-aware sampling
- Priority queue generation
```

#### 6. Evaluation Metrics (`eval/metrics.py`)
```python
# Key functionality needed:
- Overall and per-slice F1/precision/recall
- ECE and Brier scores
- Coverage and abstention rates
- Cost and latency tracking
```

#### 7. Main Scripts (`scripts/`)
Key scripts to implement:
- `train_lora_sft.py`: LoRA fine-tuning orchestration
- `predict_all_voters.py`: Run all voters and save predictions
- `train_stacker.py`: Train arbiter stacker
- `set_conformal_threshold.py`: Calibrate conformal thresholds
- `eval_full_system.py`: Complete system evaluation

## Model Selection Rationale

**Primary: Qwen2.5-7B**
- Strong 7B performance (July 2025 updates)
- Good multilingual support
- Efficient LoRA fine-tuning
- Active community support

**Alternatives:**
- Llama 3.3 8B: If Meta ecosystem preferred
- Gemma 3 (1B/12B): For edge deployment or more compute

## Performance Targets

- **Overall F1**: ≥0.92
- **Worst-slice F1**: ≥0.85
- **ECE**: ≤0.03 (calibrated models)
- **LLM call rate**: ≤10%
- **Abstention rate**: ≤15%
- **Cost per 1K**: ≤$10
- **P95 latency**: ≤500ms

## CI/CD Pipeline

### Tests (`ci/`)
1. **Unit tests**: DSL rules, calibration
2. **Integration tests**: Voter compatibility
3. **Safety regression**: Frozen safety set
4. **Promotion gates**: Champion/challenger

### Monitoring
- Coverage at fixed risk levels
- Per-slice performance tracking
- Cost and latency metrics
- Abstention analysis

## Recent Research Integration

The system incorporates findings from recent papers (May-Aug 2025):

1. **Cascaded LLM frameworks** with explicit deferral
2. **Conformal abstention** for risk control
3. **Stacking over heterogeneous voters**
4. **Weak supervision with inference-time label models**
5. **Selective prediction** with coverage guarantees

## Next Steps for Implementation

1. **Immediate Priority:**
   - Complete LLM LoRA training pipeline
   - Implement arbiter stacker with OOF
   - Add conformal threshold selection

2. **Testing Infrastructure:**
   - Create sample data for testing
   - Unit tests for each component
   - Integration test suite

3. **Deployment Readiness:**
   - Docker containerization
   - API endpoint design
   - Monitoring dashboards

## License

[Your License Here]

## Contributing

See CONTRIBUTING.md for guidelines.

## Support

For issues and questions, please use GitHub Issues.