# Bird Flu Classification Fine-Tuning Pipeline

A comprehensive fine-tuning pipeline for bird flu content classification using ensemble methods and LoRA adapters.

## 🎯 Overview

This repository implements a multi-model fine-tuning approach for bird flu classification:

- **Classical Models**: TF-IDF + Logistic Regression/SVM with calibration
- **Weak Supervision**: Snorkel-like label model training
- **LLM Fine-tuning**: LoRA adapters on Qwen2.5-7B
- **Ensemble Training**: Stacker model combining all voter outputs

## 📁 Project Structure

```
birdflu-ensemble/
├── configs/           # Training configurations
├── data/             # Training and evaluation data
├── voters/           # Model implementations
│   ├── classical/    # TF-IDF + LR/SVM models
│   ├── llm/         # LoRA fine-tuning pipeline
│   └── ws_label_model/ # Weak supervision training
├── arbiter/         # Ensemble stacking logic
├── eval/            # Evaluation metrics
├── scripts/         # Training scripts
└── tests/           # Unit tests
```

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
# Prepare training data
make prepare-data

# Train all models
make train-all

# Or train individual components:
make train-classical     # TF-IDF models
make train-lora         # LoRA fine-tuning
make train-weak-supervision  # Label model
make train-stacker      # Ensemble training

# Evaluate trained models
make eval
```

## 🧠 Model Components

### Classical Models (`voters/classical/`)

**TF-IDF + Logistic Regression**
- Feature extraction with TF-IDF vectorization
- Logistic regression with regularization
- Platt scaling for probability calibration

**TF-IDF + SVM**
- Same feature extraction pipeline
- Support Vector Machine classifier
- Isotonic regression for calibration

Both models target Expected Calibration Error (ECE) ≤ 0.03.

### LLM Fine-tuning (`voters/llm/`)

**LoRA Fine-tuning on Qwen2.5-7B**
- Parameter-Efficient Fine-Tuning (PEFT) with LoRA
- Structured JSON output with schema validation
- Explicit abstention support for uncertain cases
- Configurable LoRA parameters (rank, alpha, learning rates)

**Key Features:**
- Rank 16, Alpha 32 LoRA configuration
- JSON schema enforcement for consistent outputs
- Logprob extraction for uncertainty quantification
- Graceful fallback to abstention on invalid JSON

### Weak Supervision (`voters/ws_label_model/`)

**Snorkel-like Label Model**
- Combines multiple labeling function outputs
- Probabilistic consensus via generative model
- Handles conflicting labeling function votes
- Minimum evidence thresholds for quality control

### Ensemble Training (`arbiter/`)

**Stacker Model**
- Logistic regression or XGBoost stacker
- Out-of-fold training to prevent overfitting
- Feature engineering from voter outputs
- Entropy, margin, and disagreement signals

**Conformal Prediction**
- Split conformal prediction for coverage guarantees
- Per-slice threshold adjustments
- Risk-controlled abstention

## ⚙️ Configuration

### Training Parameters (`configs/llm_lora.yaml`)

```yaml
lora:
  r: 16
  alpha: 32
  dropout: 0.1
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]

training:
  learning_rate: 5e-5
  batch_size: 8
  gradient_accumulation_steps: 4
  num_epochs: 3
  warmup_ratio: 0.1
```

### Model Selection (`configs/voters.yaml`)

```yaml
voters:
  classical_lr:
    enabled: true
    cost: 0.001
  classical_svm:
    enabled: true
    cost: 0.001
  llm_lora:
    enabled: true
    cost: 0.05
    model_name: "Qwen/Qwen2.5-7B"
```

## 📊 Evaluation Metrics

The pipeline tracks multiple evaluation metrics:

- **F1 Score**: Overall and per-class performance
- **Calibration**: Expected Calibration Error (ECE), Brier Score
- **Coverage**: Abstention rates and coverage guarantees
- **Efficiency**: Training time, inference speed, model size

## 🔬 Training Scripts

### Data Preparation
```bash
python scripts/prepare_data.py
# - Splits data into train/dev/test sets
# - Creates weak supervision training data
# - Preprocesses text for different model types
```

### Model Training
```bash
python scripts/train_lora_sft.py
# - Fine-tunes Qwen2.5-7B with LoRA adapters
# - Saves checkpoints and training metrics
# - Validates on development set

python scripts/train_classical.py
# - Trains TF-IDF + LR/SVM models
# - Applies calibration methods
# - Saves trained models and vectorizers
```

### Ensemble Training
```bash
python scripts/predict_all_voters.py
# - Generates out-of-fold predictions from all voters
# - Ensures no data leakage for stacker training

python scripts/train_stacker.py
# - Trains ensemble stacker on voter outputs
# - Feature engineering and model selection
# - Cross-validation for hyperparameter tuning
```

## 🧪 Testing

```bash
# Full test suite
make test

# Code formatting and linting
make format
make lint

# Clean artifacts
make clean
```

## 📋 Requirements

- **Python**: ≥3.9
- **GPU**: Recommended for LLM fine-tuning (8GB+ VRAM)
- **Memory**: 16GB+ RAM for full pipeline
- **Storage**: 20GB+ for models and data

### Key Dependencies

- `torch>=2.0.0` - PyTorch framework
- `transformers>=4.35.0` - Hugging Face transformers
- `peft>=0.7.0` - Parameter-Efficient Fine-Tuning
- `scikit-learn>=1.3.0` - Classical ML models
- `datasets>=2.14.0` - Data loading and processing

## 🔧 Development

### Adding New Models

1. Create new voter in `voters/` directory
2. Implement training and inference methods
3. Add configuration to `configs/voters.yaml`
4. Update training pipeline in `scripts/`

### Custom Datasets

1. Place data files in `data/raw/`
2. Update `scripts/prepare_data.py` for preprocessing
3. Modify label definitions in `configs/labels.yaml`

## 📚 Documentation

- **CLAUDE.md**: Detailed technical guidance for development
- **configs/**: Inline documentation in YAML configuration files
- **Code Comments**: Implementation details in Python modules

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Follow code quality standards: `make lint && make test`
4. Submit pull request with clear description

## 📄 License

[MIT License](LICENSE) - see LICENSE file for details.

---

**Focus**: Model fine-tuning and ensemble training for bird flu classification
**Built with**: PyTorch, Transformers, PEFT, scikit-learn, Snorkel