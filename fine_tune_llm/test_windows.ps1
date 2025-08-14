# Windows PowerShell Test Runner for Fine-Tune LLM
# Provides more control and better error handling

param(
    [Parameter(Position=0)]
    [ValidateSet("quick", "core", "ml", "all", "fix")]
    [string]$TestType = "quick"
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Windows Test Runner for Fine-Tune LLM" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Set environment variables
$env:PYTEST_TIMEOUT = "300"
$env:TOKENIZERS_PARALLELISM = "false"
$env:OMP_NUM_THREADS = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:512"

# Function to run tests with error handling
function Run-Tests {
    param(
        [string]$TestCommand,
        [string]$TestName
    )
    
    Write-Host "Running $TestName..." -ForegroundColor Yellow
    Write-Host "----------------------------------------" -ForegroundColor Gray
    
    try {
        Invoke-Expression $TestCommand
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ $TestName passed successfully" -ForegroundColor Green
        } else {
            Write-Host "✗ $TestName had failures" -ForegroundColor Red
        }
    } catch {
        Write-Host "✗ $TestName encountered an error: $_" -ForegroundColor Red
    }
    
    Write-Host ""
}

switch ($TestType) {
    "quick" {
        Run-Tests -TestCommand "python -m pytest tests/ --quick -v --tb=short --maxfail=5" -TestName "QUICK tests (no ML)"
    }
    
    "core" {
        Run-Tests -TestCommand "python -m pytest tests/test_dataset.py tests/test_integration.py tests/test_high_stakes_simple.py tests/test_high_stakes.py -v --tb=short" -TestName "CORE functionality tests"
    }
    
    "ml" {
        Write-Host "WARNING: ML tests may timeout on Windows" -ForegroundColor Yellow
        Run-Tests -TestCommand "python -m pytest tests/ --ml -v --tb=short --timeout=600" -TestName "ML tests"
    }
    
    "all" {
        Write-Host "Running comprehensive test suite..." -ForegroundColor Cyan
        
        # Core tests
        Run-Tests -TestCommand "python -m pytest tests/test_dataset.py tests/test_integration.py tests/test_high_stakes_simple.py tests/test_high_stakes.py -v --tb=short" -TestName "Core tests"
        
        # Try ML tests with timeout protection
        $mlTests = @(
            "tests/test_sft_lora.py",
            "tests/test_evaluate.py", 
            "tests/test_llm_infer.py"
        )
        
        foreach ($test in $mlTests) {
            if (Test-Path $test) {
                Run-Tests -TestCommand "python -m pytest $test -v --tb=short --timeout=300 2>$null" -TestName "ML test: $test"
            }
        }
    }
    
    "fix" {
        Write-Host "Attempting to fix Windows environment issues..." -ForegroundColor Cyan
        
        # Install missing packages
        Write-Host "Installing test optimization packages..." -ForegroundColor Yellow
        & python -m pip install pytest-xdist pytest-timeout pytest-lazy-fixture --quiet 2>$null
        
        # Clear pytest cache
        Write-Host "Clearing pytest cache..." -ForegroundColor Yellow
        Remove-Item -Path ".pytest_cache" -Recurse -Force -ErrorAction SilentlyContinue
        
        # Clear Python cache
        Write-Host "Clearing Python cache..." -ForegroundColor Yellow
        Get-ChildItem -Path . -Include "__pycache__" -Recurse -Directory | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        
        Write-Host "✓ Environment fixes applied" -ForegroundColor Green
        Write-Host "Now try running: .\test_windows.ps1 core" -ForegroundColor Cyan
    }
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Test run completed" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan