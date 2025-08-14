.PHONY: install test lint format clean train-all eval-all

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

test-safety:
	python scripts/eval_full_system.py --safety-set-only

prepare-data:
	python scripts/prepare_data.py

train-weak-supervision:
	python scripts/run_lfs.py
	python scripts/train_weak_supervision.py

train-classical:
	python scripts/train_classical.py

train-lora:
	python scripts/train_lora_sft.py

train-stacker:
	python scripts/predict_all_voters.py
	python scripts/train_stacker.py

set-thresholds:
	python scripts/set_conformal_threshold.py

train-all: prepare-data train-weak-supervision train-classical train-lora train-stacker set-thresholds

eval-all:
	python scripts/eval_full_system.py

generate-disagreements:
	python scripts/generate_disagreements.py

serve:
	streamlit run miner/audit_ui/app.py

clean:
	rm -rf artifacts/probs/*.jsonl
	rm -rf artifacts/runs/*
	rm -rf __pycache__ .pytest_cache .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete