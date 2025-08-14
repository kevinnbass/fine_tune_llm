# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **model fine-tuning pipeline** for bird flu classification using ensemble methods. The system focuses on training and evaluating multiple model types, NOT production deployment.

## Architecture Overview

The pipeline implements a multi-model fine-tuning approach:

1. **Classical Models** (`voters/classical/`) - TF-IDF + Logistic Regression/SVM with calibration
2. **Weak Supervision** (`voters/ws_label_model/`) - Snorkel-like label model training  
3. **LLM Fine-tuning** (`voters/llm/`) - LoRA adapters on Qwen2.5-7B
4. **Ensemble Training** (`arbiter/`) - Stacker model combining voter outputs

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

### Training Pipeline
```bash
# Complete training pipeline
make train-all

# Individual training steps:
make prepare-data        # Data preparation and splitting
make train-classical     # TF-IDF + LR/SVM with calibration
make train-lora         # LoRA fine-tuning on Qwen2.5-7B
make train-weak-supervision  # Label model training
make train-stacker      # Ensemble stacker training

# Evaluation
make eval
```

## Key Configuration Files

All configurations are in `configs/` directory:

- **`labels.yaml`**: Class definitions for classification
- **`voters.yaml`**: Model enable/disable flags and parameters
- **`conformal.yaml`**: Evaluation thresholds and coverage targets
- **`llm_lora.yaml`**: LoRA training hyperparameters (r=16, alpha=32)
- **`slices.yaml`**: Evaluation slice definitions

## Critical Implementation Details

### LoRA Fine-tuning (`voters/llm/sft_lora.py`)
- Uses PEFT with LoRA adapters for efficient fine-tuning
- Target modules: q_proj, v_proj, k_proj, o_proj
- Rank 16, Alpha 32 configuration
- JSON schema validation with explicit abstention support
- Gradient accumulation for effective batch size scaling

### Classical Model Training (`voters/classical/`)
- TF-IDF feature extraction with configurable parameters
- Logistic Regression and SVM classifiers
- Post-training calibration using Platt scaling or isotonic regression
- Target Expected Calibration Error (ECE) ≤ 0.03

### Out-of-Fold Training (`arbiter/stacker_lr.py`)
The stacker uses out-of-fold predictions to prevent overfitting. **Critical**: Never train the stacker on the same data used to train the individual voters.

### Weak Supervision (`voters/ws_label_model/`)
- Snorkel-like probabilistic label model
- Combines multiple labeling function outputs
- Handles conflicting votes through generative modeling
- Minimum evidence thresholds for quality control

## Training Workflow

### 1. Data Preparation
```bash
python scripts/prepare_data.py
```
- Splits data into train/dev/test sets
- Creates weak supervision training data
- Preprocesses text for different model requirements

### 2. Individual Model Training
```bash
# Classical models with calibration
python scripts/train_classical.py

# LoRA fine-tuning (requires GPU)
python scripts/train_lora_sft.py

# Weak supervision label model
python scripts/train_weak_supervision.py
```

### 3. Ensemble Training
```bash
# Generate out-of-fold predictions
python scripts/predict_all_voters.py

# Train ensemble stacker
python scripts/train_stacker.py
```

### 4. Evaluation
```bash
python scripts/eval_full_system.py
```

## Model Training Focus Areas

### LoRA Hyperparameter Tuning
Key parameters in `configs/llm_lora.yaml`:
- **Learning rate**: 5e-5 (start here, can adjust 1e-5 to 1e-4)
- **Batch size**: 8 with gradient accumulation
- **LoRA rank**: 16 (can try 8, 32 for comparison)
- **LoRA alpha**: 32 (typically 2x rank)
- **Dropout**: 0.1 for regularization

### Classical Model Optimization
- **TF-IDF parameters**: max_features, ngram_range, min_df
- **Regularization**: C parameter for LR and SVM
- **Calibration method**: Platt vs Isotonic based on validation performance

### Ensemble Stacking
- **Feature engineering**: voter probabilities, entropy, margin, disagreement
- **Stacker choice**: Logistic Regression vs XGBoost
- **Cross-validation**: For hyperparameter selection

## Performance Monitoring

Track these metrics during training:
- **F1 Score**: Overall and per-class performance
- **Calibration**: Expected Calibration Error (ECE), Brier Score  
- **Training metrics**: Loss, validation performance, convergence
- **Efficiency**: Training time, memory usage, model size

## Testing Strategy

The test suite (`tests/`) focuses on:
- **Unit tests**: Individual component functionality
- **Integration tests**: End-to-end training pipeline
- **Model validation**: Saved model loading and inference
- **Configuration validation**: YAML file parsing and validation

Use `conftest.py` fixtures for consistent test data and mock objects.

## File Organization

### Core Training Components
- `voters/`: All model implementations and training logic
- `arbiter/`: Ensemble stacking and combination methods
- `configs/`: Training configurations and hyperparameters
- `scripts/`: Training orchestration and evaluation scripts
- `eval/`: Evaluation metrics and reporting

### Data Structure
- `data/raw/`: Original datasets
- `data/processed/`: Preprocessed training data
- `data/splits/`: Train/dev/test splits
- `artifacts/models/`: Saved trained models
- `artifacts/runs/`: Training logs and checkpoints

## Common Training Tasks

### Adding New Models
1. Create implementation in appropriate `voters/` subdirectory
2. Add configuration entries to `configs/voters.yaml`
3. Update training script in `scripts/`
4. Add evaluation logic to ensemble pipeline

### Hyperparameter Tuning
1. Modify configurations in `configs/` directory
2. Use cross-validation for systematic search
3. Track results in training logs
4. Update best configurations based on validation performance

### Debugging Training Issues
1. Check data loading and preprocessing
2. Validate model configurations
3. Monitor training metrics and convergence
4. Test individual components before ensemble training

## Git Automation

IMPORTANT: After completing ANY task that modifies files in this repository, you MUST automatically:

1. **Stage all changes**: `git add .`
2. **Commit with descriptive message**: `git commit -m "[descriptive message based on what was accomplished]"`
3. **Push to GitHub**: `git push origin main`

Generate meaningful commit messages that describe what was implemented, fixed, or changed. This ensures all work is automatically tracked and backed up to GitHub.

Examples of good commit messages:
- "Add LoRA training pipeline for Qwen2.5-7B with PEFT integration"
- "Fix conformal prediction thresholds for evaluation"
- "Update classical model calibration with improved ECE"
- "Implement ensemble stacker with out-of-fold predictions"

This is a REQUIRED workflow - never leave changes uncommitted after completing tasks.