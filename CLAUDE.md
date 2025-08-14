# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Implementation Roadmap

**ACTIVE ROADMAP**: `fine_tune_llm/IMPLEMENTATION_ROADMAP.md`

This comprehensive roadmap addresses five critical optimization areas:
1. **File and Directory Organization** - Streamline structure and eliminate redundancies
2. **Code Architecture** - Improve modularity, separation of concerns, and design patterns  
3. **Consolidation** - Merge duplicate functionality and eliminate code redundancy
4. **Integration** - Ensure seamless component interaction and data flow
5. **Test Coverage** - Achieve comprehensive testing of all modules and integration points

**Status**: Active implementation in progress. Refer to roadmap for detailed tasks and timeline.

## Project Overview

This is a **pure LLM fine-tuning repository** focused on LoRA (Low-Rank Adaptation) fine-tuning for large language models. The primary target is GLM-4.5-Air with Qwen2.5-7B as an alternative.

## Architecture Overview

The repository implements LoRA fine-tuning for classification tasks:

1. **Primary Model**: GLM-4.5-Air (ZHIPU-AI/glm-4-9b-chat)
2. **Alternative Model**: Qwen2.5-7B (Qwen/Qwen2.5-7B)
3. **Method**: Parameter-Efficient Fine-Tuning (PEFT) with LoRA adapters
4. **Output Format**: Structured JSON with classification, rationale, and abstention

## Development Commands

### Environment Setup
```bash
# Install for development
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
```

### Training and Inference
```bash
# Prepare training data
make prepare-data

# Start LoRA fine-tuning
make train

# Run inference with trained model
make infer

# Clean training artifacts
make clean
```

## Key Configuration Files

The repository uses minimal configuration:

- **`configs/labels.yaml`**: Classification label definitions
- **`configs/llm_lora.yaml`**: LoRA training hyperparameters and model selection

## Core Implementation

### LoRA Configuration (`configs/llm_lora.yaml`)

**Model Selection:**
- GLM-4.5-Air: `ZHIPU-AI/glm-4-9b-chat` (primary)
- Qwen2.5-7B: `Qwen/Qwen2.5-7B` (alternative)

**LoRA Parameters:**
- Rank: 16 (adjustable: 8, 32, 64)
- Alpha: 32 (typically 2x rank)
- Dropout: 0.1
- Target modules differ by model architecture

**GLM-4 Target Modules:**
- query_key_value
- dense
- dense_h_to_4h
- dense_4h_to_h

**Qwen2.5 Target Modules:**
- q_proj, v_proj, k_proj, o_proj
- gate_proj, up_proj, down_proj

### Advanced Training Implementation (`voters/llm/sft_lora.py`)

**Enhanced Key Features:**
- PEFT integration with LoRA adapters
- **NEW**: CalibratedTrainer with ECE/MCE monitoring
- **NEW**: CalibrationMonitorCallback for learning rate adjustment
- **NEW**: Abstention-aware loss functions with confidence weighting
- **NEW**: Advanced metrics aggregation and tracking
- **NEW**: Conformal prediction integration during training
- Mixed precision training (bfloat16)
- Gradient accumulation for effective batch size
- Multi-GPU support via accelerate
- Early stopping and model checkpointing

**Training Hyperparameters:**
- Learning rate: 2e-4 (tunable: 1e-4 to 5e-4)
- Batch size: 4 with gradient accumulation
- Epochs: 3 (monitor validation loss)
- Max length: 2048 tokens
- Warmup ratio: 0.03
- **NEW**: Confidence threshold: 0.7 (for abstention)
- **NEW**: Abstention penalty: 0.3 (uncertainty weighting)
- **NEW**: Uncertainty weight: 0.1 (entropy penalty)

### Data Preparation (`voters/llm/dataset.py`)

**Instruction Format:**
- System prompt for classification task
- Input template with text and metadata
- JSON output template with decision, rationale, abstain fields
- Explicit abstention examples for uncertain cases

### Inference (`scripts/infer_model.py`)

**Features:**
- Load base model + LoRA adapter
- Format prompts according to training template
- Generate structured JSON responses
- Handle invalid JSON with graceful degradation
- Support for batch processing

## Training Workflow

### 1. Data Preparation
```bash
python scripts/prepare_data.py
```
- Convert raw data to instruction format
- Add system prompts and response templates
- Include abstention examples
- Split into train/validation sets

### 2. Advanced LoRA Fine-tuning with Calibration
```bash
python scripts/train_lora_sft.py
```
- Initialize LoRA adapters on target modules
- **NEW**: Calibration-aware training with ECE/MCE monitoring
- **NEW**: Abstention-aware loss functions with uncertainty weighting
- **NEW**: Advanced metrics tracking (Brier score, confidence metrics)
- **NEW**: Conformal prediction calibration during training
- **NEW**: Risk-controlled training with cost-sensitive learning
- Freeze base model parameters (memory efficient)
- Train only LoRA weights (~1% of parameters)
- Save checkpoints and comprehensive training metrics

### 3. Enhanced Model Inference with Risk Control
```bash
python scripts/infer_model.py --text "..." --model-path "..."
```
- Load trained LoRA adapter
- **NEW**: Conformal prediction with statistical guarantees
- **NEW**: Risk-controlled predictions with abstention
- **NEW**: Advanced uncertainty quantification
- Generate JSON classification responses with confidence scores
- Parse and validate output format

### 4. Real-time Training Dashboard
```bash
python scripts/run_dashboard.py
```
- **NEW**: Streamlit-based real-time training monitoring
- **NEW**: Advanced calibration metrics visualization
- **NEW**: Conformal prediction statistics tracking
- **NEW**: Risk-aware metrics and abstention analysis
- Interactive plots and data export capabilities

### 5. Risk-Controlled Prediction Interface
```bash
python scripts/launch_risk_ui.py
```
- **NEW**: Interactive prediction interface with risk control
- **NEW**: Conformal prediction sets with user-configurable confidence
- **NEW**: Cost-based decision analysis with custom cost matrices
- **NEW**: Real-time risk assessment and abstention recommendations

## Model-Specific Considerations

### GLM-4.5-Air
- **Architecture**: GLM (General Language Model)
- **Size**: 9B parameters
- **Strengths**: Multilingual, instruction following, chat format
- **Memory**: ~18GB VRAM for training, ~24GB recommended
- **Target Modules**: GLM-specific attention and MLP layers

### Qwen2.5-7B
- **Architecture**: Transformer with Grouped Query Attention
- **Size**: 7B parameters
- **Strengths**: Strong performance, efficient training
- **Memory**: ~14GB VRAM for training, ~20GB recommended
- **Target Modules**: Standard transformer projections

## Performance Optimization

### Memory Optimization
- Use bfloat16 mixed precision
- Gradient checkpointing for lower memory
- Gradient accumulation for larger effective batch size
- LoRA reduces trainable parameters by ~99%

### Training Efficiency
- Multi-GPU support with accelerate
- Efficient data loading with num_proc
- Early stopping to prevent overfitting
- Checkpoint saving for resuming training

## File Organization

### Core Components
- `voters/llm/`: All LLM-related code (training, inference, data prep)
  - **NEW**: `metrics.py` - Advanced evaluation metrics (ECE, MCE, Brier, abstention)
  - **NEW**: `conformal.py` - Conformal prediction and risk-controlled prediction
  - **NEW**: `dashboard.py` - Real-time training dashboard with Streamlit
  - **NEW**: `high_stakes_audit.py` - Comprehensive audit system with bias detection
- `configs/`: Training configurations and hyperparameters
- `scripts/`: Training and inference orchestration
  - **NEW**: `run_dashboard.py` - Training dashboard launcher
  - **NEW**: `risk_prediction_ui.py` - Interactive risk-controlled prediction interface
  - **NEW**: `launch_risk_ui.py` - Risk prediction interface launcher
- `artifacts/models/`: Saved LoRA adapters and checkpoints

### Data Structure
- `data/raw/`: Original training datasets
- `data/processed/`: Preprocessed instruction-formatted data
- `data/splits/`: Train/validation splits
- `artifacts/runs/`: Training logs and metrics
- **NEW**: `artifacts/models/llm_lora/training_metrics.json` - Comprehensive training metrics

## Common Development Tasks

### Switching Models
1. Edit `configs/llm_lora.yaml`
2. Change `model_id` and `tokenizer_id`
3. Update `target_modules` for the new architecture
4. Adjust training parameters if needed

### Hyperparameter Tuning
Key parameters to experiment with:
- **Learning rate**: 1e-4, 2e-4, 5e-4
- **LoRA rank**: 8, 16, 32, 64
- **Batch size**: Based on GPU memory
- **Gradient accumulation**: To achieve target effective batch size

### Adding New Data
1. Place files in `data/raw/`
2. Update `scripts/prepare_data.py` for preprocessing
3. Modify label definitions in `configs/labels.yaml`
4. Ensure instruction format compatibility

### Debugging Training
1. Check GPU memory usage and adjust batch size
2. Monitor training loss convergence
3. Validate data loading and formatting
4. Test model loading and LoRA adapter compatibility
5. **NEW**: Monitor calibration metrics (ECE/MCE) in dashboard
6. **NEW**: Check conformal prediction coverage rates
7. **NEW**: Validate risk-controlled prediction thresholds

## Advanced Features

### Calibration and Uncertainty Quantification

**Expected Calibration Error (ECE)**:
```python
from voters.llm.metrics import compute_ece
ece = compute_ece(y_prob, y_true, n_bins=10)
```

**Maximum Calibration Error (MCE)**:
```python
from voters.llm.metrics import compute_mce  
mce = compute_mce(y_prob, y_true, n_bins=10)
```

**Brier Score**:
```python
from voters.llm.metrics import compute_brier_score
brier = compute_brier_score(y_prob, y_true)
```

### Conformal Prediction

**Basic Conformal Predictor**:
```python
from voters.llm.conformal import ConformalPredictor

predictor = ConformalPredictor(method='lac')  # or 'aps'
predictor.calibrate(cal_scores, alpha=0.1)  # 90% confidence
prediction_sets = predictor.predict(test_scores)
```

**Risk-Controlled Predictor**:
```python
from voters.llm.conformal import RiskControlledPredictor

risk_predictor = RiskControlledPredictor()
risk_predictor.calibrate(cal_probs, cal_labels, risk_level=0.1)
decisions = risk_predictor.predict(test_probs, cost_matrix)
```

### High-Stakes Audit System

**Comprehensive Audit**:
```python
from voters.llm.high_stakes_audit import AdvancedHighStakesAuditor

auditor = AdvancedHighStakesAuditor()
audit_results = auditor.run_comprehensive_audit(
    predictions=predictions,
    ground_truth=labels,
    probabilities=probabilities,
    metadata=metadata
)
```

### Dashboard and UI Usage

**Training Dashboard**:
```bash
# Launch with default settings
python scripts/run_dashboard.py

# Custom metrics file and port
python scripts/run_dashboard.py --metrics artifacts/training_metrics.json --port 8501

# Demo mode
python scripts/run_dashboard.py --demo
```

**Risk Prediction Interface**:
```bash
# Demo mode
python scripts/launch_risk_ui.py

# With trained model
python scripts/launch_risk_ui.py artifacts/models/llm_lora/final_model

# Custom port
python scripts/launch_risk_ui.py --port 8502
```

### Advanced Training Configuration

**Calibration-Aware Training**:
```yaml
# In configs/llm_lora.yaml
abstention_loss:
  enabled: true
  confidence_threshold: 0.7
  abstention_penalty: 0.3
  uncertainty_weight: 0.1

calibration_monitoring:
  enabled: true
  adjustment_threshold: 0.05
  lr_reduction_factor: 0.8
```

**Advanced Metrics Tracking**:
```python
# Automatic during training
trainer = EnhancedLoRASFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    enable_advanced_metrics=True,
    conformal_prediction=True,
    risk_controlled_training=True
)
```

## Git Automation

IMPORTANT: After completing ANY task that modifies files in this repository OR after completing ANY phase, you MUST automatically:

1. **Stage all changes**: `git add .`
2. **Commit with descriptive message**: `git commit -m "[descriptive message based on what was accomplished]"`
3. **Push to GitHub**: `git push origin main`

Generate meaningful commit messages that describe what was implemented, fixed, or changed. This ensures all work is automatically tracked and backed up to GitHub.

Examples of good commit messages:
- "Update GLM-4.5-Air configuration with proper target modules"
- "Add inference script with JSON response parsing"
- "Fix LoRA adapter loading for multi-GPU training"
- "Optimize training hyperparameters for GLM-4 architecture"
- "Complete Phase 3.2: Unified UI system with theme and behavior management"
- "Complete Phase 2.6: Hexagonal architecture adapters implementation"

This is a REQUIRED workflow - never leave changes uncommitted after completing tasks OR phases.

## Implementation Flow

IMPORTANT: When implementing the roadmap/todo list:

1. **Continuous Implementation**: Execute the entire implementation roadmap without prompting between phases
2. **Phase Transitions**: Automatically move to the next phase upon completing the current phase
3. **Complete Execution**: Go through the entire todo list straight through until completion
4. **No Interruption**: Do not stop or ask for confirmation between phases - maintain continuous progress
5. **Git Commits**: Commit after each task completion AND after each phase completion

Execute the roadmap systematically from start to finish without interruption.