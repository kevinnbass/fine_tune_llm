@echo off
REM Windows Test Runner for Fine-Tune LLM
REM Optimized for Windows environment with ML dependencies

echo ========================================
echo Windows Test Runner for Fine-Tune LLM
echo ========================================
echo.

REM Set environment variables for Windows
set PYTEST_TIMEOUT=300
set TOKENIZERS_PARALLELISM=false
set OMP_NUM_THREADS=1
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

REM Run tests based on argument
if "%1"=="" goto :quick
if "%1"=="quick" goto :quick
if "%1"=="core" goto :core
if "%1"=="ml" goto :ml
if "%1"=="all" goto :all
if "%1"=="help" goto :help

:help
echo Usage: test_windows.bat [option]
echo.
echo Options:
echo   quick  - Run quick tests only (default, no ML imports)
echo   core   - Run core functionality tests
echo   ml     - Run ML tests (may timeout)
echo   all    - Run all tests
echo   help   - Show this help message
echo.
goto :end

:quick
echo Running QUICK tests (no ML dependencies)...
echo ----------------------------------------
python -m pytest tests/ --quick -v --tb=short --maxfail=5
goto :end

:core
echo Running CORE functionality tests...
echo ----------------------------------------
python -m pytest tests/test_dataset.py tests/test_integration.py tests/test_high_stakes_simple.py tests/test_high_stakes.py -v --tb=short
goto :end

:ml
echo Running ML tests (this may take a while)...
echo ----------------------------------------
echo WARNING: ML tests may timeout on Windows due to heavy imports
echo.
python -m pytest tests/ --ml -v --tb=short --timeout=600
goto :end

:all
echo Running ALL tests...
echo ----------------------------------------
echo Step 1: Core tests
python -m pytest tests/test_dataset.py tests/test_integration.py tests/test_high_stakes_simple.py tests/test_high_stakes.py -v --tb=short
echo.
echo Step 2: ML tests (if available)
python -m pytest tests/test_sft_lora.py tests/test_evaluate.py tests/test_llm_infer.py -v --tb=short --timeout=600 2>nul
echo.
echo Step 3: Script tests
python -m pytest tests/test_infer_model.py tests/test_prepare_data.py tests/test_merge_lora.py tests/test_tune_hyperparams.py -v --tb=short --timeout=300 2>nul
goto :end

:end
echo.
echo ========================================
echo Test run completed
echo ========================================