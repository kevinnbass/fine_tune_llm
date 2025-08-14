.PHONY: install test lint format clean train

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

infer:
	python scripts/infer_model.py

clean:
	rm -rf artifacts/runs/*
	rm -rf __pycache__ .pytest_cache .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete