# Test Suite Organization

This directory contains organized test suites to run comprehensive testing without timeout issues.

## Test Suites

### 1. Core Suite (`core_suite.py`)
**Fast, essential functionality tests:**
- `test_dataset.py` - Data processing pipeline
- `test_integration.py` - System integration 
- `test_high_stakes_simple.py` - Simplified high-stakes features

**Run with:** `python tests/suites/core_suite.py`

### 2. ML Suite (`ml_suite.py`) 
**Machine Learning focused tests with heavier imports:**
- `test_sft_lora.py` - LoRA training
- `test_high_stakes.py` - High-stakes ML features  
- `test_evaluate.py` - Model evaluation
- `test_llm_infer.py` - LLM inference

**Run with:** `python tests/suites/ml_suite.py`

### 3. Scripts Suite (`scripts_suite.py`)
**Script functionality tests:**
- `test_infer_model.py` - Inference script
- `test_prepare_data.py` - Data preparation script
- `test_merge_lora.py` - LoRA merging script
- `test_tune_hyperparams.py` - Hyperparameter tuning
- `test_train_high_stakes.py` - High-stakes training script
- `test_ui.py` - UI functionality

**Run with:** `python tests/suites/scripts_suite.py`

## Run All Suites

### Sequential Execution (Recommended)
```bash
python tests/suites/run_all_suites.py
```

This runs all suites sequentially with:
- Proper timeout handling
- Memory cleanup between suites
- Comprehensive reporting
- Graceful handling of import issues

### Individual Suite Execution
```bash
# Run just the core tests (fastest)
python tests/suites/core_suite.py

# Run ML tests (may have timeouts)
python tests/suites/ml_suite.py

# Run script tests (may skip some due to imports)
python tests/suites/scripts_suite.py
```

## Expected Results

### Core Suite
- **Expected:** 70+ tests passing
- **Time:** 1-3 minutes
- **Issues:** Minimal, should always pass

### ML Suite  
- **Expected:** 30-50 tests passing, some skipped
- **Time:** 5-15 minutes
- **Issues:** Import timeouts, dependency issues handled gracefully

### Scripts Suite
- **Expected:** 50-100 tests passing, many skipped
- **Time:** 3-10 minutes  
- **Issues:** Import path issues, dependency availability

## Benefits of This Organization

1. **No More Timeouts:** Each suite runs independently
2. **Focused Testing:** Run only the suites you need
3. **Clear Reporting:** See exactly which areas pass/fail
4. **Graceful Degradation:** Missing dependencies don't break everything
5. **Comprehensive Coverage:** All tests eventually run when possible