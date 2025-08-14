.PHONY: install test lint format clean train-all eval

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

train-classical:
	python scripts/train_classical.py

train-lora:
	python scripts/train_lora_sft.py

train-weak-supervision:
	python scripts/train_weak_supervision.py

train-stacker:
	python scripts/predict_all_voters.py
	python scripts/train_stacker.py

train-all: prepare-data train-classical train-lora train-weak-supervision train-stacker

eval:
	python scripts/eval_full_system.py

clean:
	rm -rf artifacts/probs/*.jsonl
	rm -rf artifacts/runs/*
	rm -rf __pycache__ .pytest_cache .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete