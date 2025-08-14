# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Production-ready LLM fine-tuning platform implementing advanced PEFT methods (LoRA, QLoRA, DoRA, AdaLoRA) with multi-model support and comprehensive tooling. Features a Gradio web UI, hyperparameter tuning, and structured output generation.

## Commands

### Core Development Commands
```bash
# Installation (from fine_tune_llm/ directory)
cd fine_tune_llm
make install-dev         # Install with development dependencies
make install             # Basic installation

# Code Quality
make format              # Auto-format with black + isort
make lint                # Run flake8, mypy, black --check, isort --check
make test                # Run pytest with coverage

# Data Preparation
make prepare-data        # Prepare training data with optional augmentation

# Training Methods
make train               # Standard LoRA training
make train-qlora         # QLoRA with 4-bit quantization (memory efficient)
make train-dora          # DoRA method (often better than LoRA)
make train-adalora       # AdaLoRA with adaptive rank allocation

# Evaluation & Inference
make eval                # Comprehensive model evaluation with metrics
make infer               # Basic inference
make infer-structured    # Inference with JSON schema validation

# Advanced Features
make tune                # Hyperparameter optimization with Optuna
make merge               # Merge LoRA adapters for deployment
make ui                  # Launch Gradio web interface (localhost:7860)

# Docker Operations
make docker-build        # Build Docker image
make docker-run          # Run with Docker Compose
make docker-logs         # View container logs

# Cleanup
make clean               # Remove artifacts and cache
```

### Testing Individual Components
```bash
# Run specific test file
pytest tests/test_module.py -v

# Run with specific marker
pytest -m "not slow" -v

# Debug mode with pdb
pytest --pdb tests/test_module.py
```

## Architecture

### Multi-Model Support
The platform supports 4 models configured in `configs/llm_lora.yaml`:
- **GLM-4.5-Air** (9B): Primary model with GLM architecture
- **Qwen2.5-7B**: High-performance alternative
- **Mistral-7B**: Popular open-source option
- **Llama-3-8B**: Meta's latest architecture

Model selection is automatic via `selected_model` field. Each model has specific target modules for LoRA adaptation.

### Training Methods
**EnhancedLoRASFTTrainer** (`voters/llm/sft_lora.py`) orchestrates training with:
- **Standard LoRA**: Low-rank adaptation of attention/MLP layers
- **QLoRA**: 4-bit/8-bit quantization via BitsAndBytesConfig
- **DoRA**: Decomposed LoRA (set `method: dora`)
- **AdaLoRA**: Adaptive rank allocation with pruning

The trainer handles model loading, PEFT configuration, and automatic optimization based on available hardware.

### Data Pipeline
1. **Raw Data** → `scripts/prepare_data.py` → **Instruction Format**
2. Optional augmentation via nlpaug (synonym replacement, random insertion)
3. Automatic train/validation splitting
4. Format includes system prompt, input template, and JSON output schema

### Structured Output Generation
Uses the `outlines` library for schema-guided generation:
- JSON schema validation during inference
- Enforced output format compliance
- Graceful degradation for invalid responses

## Key Configuration

### Main Config: `configs/llm_lora.yaml`

Critical parameters to adjust:
```yaml
selected_model: glm-4.5-air  # Change model here

lora:
  method: lora               # lora, dora, or adalora
  r: 16                      # LoRA rank (memory vs quality tradeoff)
  quantization:
    enabled: false           # Set true for QLoRA
    bits: 4                  # 4 or 8 bit quantization

training:
  learning_rate: 2e-4        # Key hyperparameter
  batch_size: 4              # Adjust based on GPU memory
  gradient_accumulation_steps: 4  # Effective batch = 16
  num_epochs: 3              # Monitor validation loss
  
  logging:
    wandb: false             # Enable for experiment tracking
```

### Memory Requirements by Configuration

| Model | Standard LoRA | QLoRA 4-bit | DoRA |
|-------|--------------|-------------|------|
| GLM-4.5-Air | 18GB | 8GB | 20GB |
| Qwen2.5-7B | 14GB | 6GB | 16GB |

## Common Workflows

### Quick Training with Limited VRAM
```bash
# Use QLoRA for 8GB GPUs
cd fine_tune_llm
make train-qlora
```

### Full Training Pipeline
```bash
cd fine_tune_llm
make prepare-data
make tune          # Find optimal hyperparameters
make train-dora    # Train with best method
make eval          # Validate performance
make merge         # Create deployment model
```

### Using Web UI
```bash
cd fine_tune_llm
make ui
# Navigate to http://localhost:7860
# Upload data, configure training, monitor progress
```

## Troubleshooting

### CUDA Out of Memory
1. Enable QLoRA: Set `quantization.enabled: true` in config
2. Reduce batch_size to 1-2
3. Increase gradient_accumulation_steps
4. Enable gradient_checkpointing (already on by default)

### Training Not Converging
1. Try different learning rates: 1e-4, 2e-4, 5e-4
2. Increase LoRA rank (r: 32 or 64)
3. Use DoRA method instead of standard LoRA
4. Check data quality and augmentation settings

### Model Loading Issues
1. Verify model_id matches Hugging Face repository
2. Check target_modules for your model architecture
3. Ensure sufficient disk space for model downloads
4. Clear Hugging Face cache: `rm -rf ~/.cache/huggingface`