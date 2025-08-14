.PHONY: install test lint format clean train train-qlora train-dora train-adalora infer merge tune ui eval docker-build docker-run docker-stop

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

format:
	black .
	isort .

lint:
	flake8 .
	mypy .
	black --check .
	isort --check-only .

test:
	pytest tests/ -v --cov=. --cov-report=html

prepare-data:
	python scripts/prepare_data.py

train:
	python scripts/train_lora_sft.py

train-qlora:
	python scripts/train_lora_sft.py --quantization 4bit

train-dora:
	python scripts/train_lora_sft.py --lora-method dora

train-adalora:
	python scripts/train_lora_sft.py --lora-method adalora

infer:
	python scripts/infer_model.py

infer-structured:
	python scripts/infer_model.py --structured

merge:
	python scripts/merge_lora.py --base-model $(MODEL_ID) --lora-path artifacts/models/llm_lora/final --output-path artifacts/merged --verify

eval:
	python voters/llm/evaluate.py --model-path artifacts/models/llm_lora/final --test-data data/processed/val.json

tune:
	python scripts/tune_hyperparams.py --train-data data/processed/train.json --val-data data/processed/val.json

ui:
	python ui.py

clean:
	rm -rf artifacts/runs/*
	rm -rf __pycache__ .pytest_cache .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Docker commands
docker-build:
	docker build -t llm-finetuning:latest .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

docker-logs:
	docker-compose logs -f llm-finetuning

docker-shell:
	docker exec -it llm-finetuning-app bash

docker-clean:
	docker system prune -f
	docker image prune -f

help:
	@echo "Available commands:"
	@echo "  install-dev    - Install development dependencies"
	@echo "  prepare-data   - Prepare training data with augmentation"
	@echo "  train          - Train with LoRA"
	@echo "  train-qlora    - Train with QLoRA (4-bit quantization)"
	@echo "  train-dora     - Train with DoRA method"
	@echo "  train-adalora  - Train with AdaLoRA method"
	@echo "  infer          - Run inference"
	@echo "  infer-structured - Run inference with structured output"
	@echo "  eval           - Evaluate trained model"
	@echo "  merge          - Merge LoRA adapter with base model"
	@echo "  tune           - Run hyperparameter tuning"
	@echo "  ui             - Launch Gradio web interface"
	@echo "  format         - Format code"
	@echo "  lint           - Run linters"
	@echo "  test           - Run tests"
	@echo "  clean          - Clean artifacts"
	@echo ""
	@echo "Docker commands:"
	@echo "  docker-build   - Build Docker image"
	@echo "  docker-run     - Run with Docker Compose"
	@echo "  docker-stop    - Stop Docker containers"
	@echo "  docker-logs    - View container logs"
	@echo "  docker-shell   - Open shell in container"
	@echo "  docker-clean   - Clean Docker cache"