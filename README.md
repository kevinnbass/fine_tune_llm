# LLM Fine-Tuning with LoRA

Fine-tuning GLM-4.5-Air (and Qwen2.5-7B) for bird flu classification using LoRA adapters.

## 🎯 Overview

This repository implements LoRA (Low-Rank Adaptation) fine-tuning for large language models:

- **Primary Model**: GLM-4.5-Air (ZHIPU-AI/glm-4-9b-chat)
- **Alternative**: Qwen2.5-7B (easy to switch)
- **Method**: Parameter-Efficient Fine-Tuning (PEFT) with LoRA
- **Task**: Bird flu content classification with JSON output

## 📁 Project Structure

```
birdflu-ensemble/
├── configs/
│   ├── labels.yaml          # Classification labels
│   └── llm_lora.yaml       # LoRA training config
├── data/                   # Training data
├── voters/llm/            # Core training code
│   ├── dataset.py         # Data preparation
│   ├── sft_lora.py        # LoRA training
│   └── infer.py           # Model inference
├── scripts/
│   ├── train_lora_sft.py  # Training script
│   └── infer_model.py     # Inference script
└── artifacts/models/      # Saved models
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

### Training

```bash
# Prepare your training data
make prepare-data

# Start LoRA fine-tuning
make train

# Monitor training in artifacts/runs/
```

### Inference

```bash
# Run inference with trained model
make infer

# Or directly:
python scripts/infer_model.py \
  --text "Your text to classify here" \
  --model-path "artifacts/models/your-model" \
  --metadata '{"source": "example"}'
```

## ⚙️ Configuration

### Model Selection

Edit `configs/llm_lora.yaml` to switch models:

```yaml
# GLM-4.5-Air (default)
model_id: ZHIPU-AI/glm-4-9b-chat
tokenizer_id: ZHIPU-AI/glm-4-9b-chat

# Or use Qwen2.5-7B (uncomment):
# model_id: Qwen/Qwen2.5-7B
# tokenizer_id: Qwen/Qwen2.5-7B
```

### LoRA Parameters

```yaml
lora:
  r: 16                    # LoRA rank
  lora_alpha: 32          # LoRA alpha (scaling)
  lora_dropout: 0.1       # Dropout rate
  target_modules:         # Modules to adapt
    - query_key_value     # GLM-4 modules
    - dense
    - dense_h_to_4h
    - dense_4h_to_h
```

### Training Hyperparameters

```yaml
training:
  learning_rate: 2e-4
  num_epochs: 3
  batch_size: 4
  gradient_accumulation_steps: 4
  max_length: 2048
  bf16: true              # Use bfloat16
```

## 🧠 Model Architecture

### GLM-4.5-Air Features
- **Size**: 9B parameters
- **Architecture**: GLM (General Language Model)
- **Strengths**: Multilingual, instruction following, chat format
- **LoRA Targets**: query_key_value, dense layers

### Qwen2.5-7B Alternative
- **Size**: 7B parameters  
- **Architecture**: Transformer with GQA
- **Strengths**: Strong performance, efficient training
- **LoRA Targets**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

## 📊 Training Process

### 1. Data Preparation
- Convert training data to instruction format
- Add system prompts for classification task
- Include JSON schema examples
- Handle abstention cases

### 2. LoRA Fine-tuning
- Initialize LoRA adapters on target modules
- Freeze base model parameters
- Train only LoRA weights (efficient!)
- Use gradient accumulation for larger effective batch size

### 3. Model Output
- Structured JSON responses
- Explicit abstention support
- Confidence and rationale included

## 🔧 Development

### Adding Training Data

1. Place data files in `data/raw/`
2. Update `scripts/prepare_data.py` 
3. Modify label definitions in `configs/labels.yaml`

### Hyperparameter Tuning

Key parameters to experiment with:
- **Learning rate**: 1e-4 to 5e-4
- **LoRA rank**: 8, 16, 32, 64
- **Batch size**: Adjust based on GPU memory
- **Epochs**: Monitor validation loss

### Multi-GPU Training

The training script supports multi-GPU:
- Uses `accelerate` for distributed training
- Automatic gradient accumulation
- Mixed precision (bfloat16)

## 📋 Requirements

- **Python**: ≥3.9
- **GPU**: 8GB+ VRAM for GLM-4 (24GB+ recommended)
- **Memory**: 16GB+ RAM
- **Storage**: 20GB+ for models and checkpoints

### Key Dependencies

- `torch>=2.0.0` - PyTorch framework
- `transformers>=4.35.0` - Hugging Face transformers
- `peft>=0.7.0` - Parameter-Efficient Fine-Tuning
- `trl>=0.7.0` - Transformer Reinforcement Learning
- `accelerate>=0.25.0` - Multi-GPU training

## 🧪 Testing

```bash
# Full test suite
make test

# Code formatting
make format
make lint

# Clean artifacts
make clean
```

## 📚 Usage Examples

### Basic Classification

```python
from scripts.infer_model import load_model, format_prompt, generate_response

# Load trained model
model, tokenizer = load_model(config, "artifacts/models/checkpoint-1000")

# Classify text
text = "This research paper discusses H5N1 transmission patterns..."
response = generate_response(model, tokenizer, format_prompt(text, {}, config))
```

### Batch Processing

```bash
# Process multiple texts
for text in texts:
    python scripts/infer_model.py --text "$text" --model-path artifacts/models/best
done
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Follow code quality: `make lint && make test`
4. Submit pull request

## 📄 License

[MIT License](LICENSE) - see LICENSE file for details.

---

**Focus**: LoRA fine-tuning for GLM-4.5-Air and Qwen2.5-7B
**Built with**: PyTorch, Transformers, PEFT, Accelerate