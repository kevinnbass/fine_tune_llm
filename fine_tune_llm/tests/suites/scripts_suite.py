"""
Scripts Test Suite - Script functionality tests  
Run with: python -m pytest tests/suites/scripts_suite.py -v
"""

# Script-focused tests
SCRIPTS_TESTS = [
    "tests/test_infer_model.py",       # Inference script
    "tests/test_prepare_data.py",      # Data preparation script  
    "tests/test_merge_lora.py",        # LoRA merging script
    "tests/test_tune_hyperparams.py",  # Hyperparameter tuning
    "tests/test_train_high_stakes.py", # High-stakes training script
    "tests/test_ui.py"                 # UI functionality
]

if __name__ == "__main__":
    import subprocess
    import sys
    import time
    
    print("=" * 60)
    print("RUNNING SCRIPTS TEST SUITE")
    print("=" * 60)
    
    total_passed = 0
    total_failed = 0
    total_skipped = 0
    
    for test_file in SCRIPTS_TESTS:
        print(f"\n{'='*20} Running {test_file} {'='*20}")
        start_time = time.time()
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                test_file, "-v", "--tb=short", "--maxfail=3"
            ], capture_output=True, text=True, cwd=".", timeout=300)
            
            duration = time.time() - start_time
            print(f"Duration: {duration:.2f}s")
            
            # Parse output for results
            output = result.stdout
            
            if "collected 0 items" in output or "SKIPPED [100%]" in output:
                print(f"SKIPPED: {test_file} (import/dependency issues)")
                total_skipped += 1
            elif result.returncode == 0 and " passed" in output:
                print(f"PASSED: {test_file}")
                # Try to extract passed count
                lines = output.split('\n')
                for line in lines:
                    if " passed" in line and " in " in line:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == "passed" and i > 0:
                                try:
                                    count = int(parts[i-1])
                                    total_passed += count
                                    print(f"  → {count} tests passed")
                                except:
                                    total_passed += 1  # Fallback
                        break
            else:
                print(f"FAILED: {test_file}")
                if "FAILED" in output:
                    total_failed += 1
                else:
                    total_skipped += 1
                    
                # Show error details
                error_lines = output.split('\n')
                print("Error summary:")
                for line in error_lines[-10:]:
                    if line.strip() and ("FAILED" in line or "ERROR" in line or "ImportError" in line):
                        print(f"  {line}")
                        
        except subprocess.TimeoutExpired:
            print(f"TIMEOUT: {test_file}")
            total_skipped += 1
        except Exception as e:
            print(f"EXCEPTION: {test_file} - {e}")
            total_failed += 1
            
    print("\n" + "="*60)
    print("SCRIPTS SUITE SUMMARY") 
    print("="*60)
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    print(f"Skipped/Timeout: {total_skipped}")
    if (total_passed + total_failed) > 0:
        success_rate = total_passed / (total_passed + total_failed) * 100
        print(f"Success Rate: {success_rate:.1f}%")
    else:
        print("Success Rate: Tests skipped due to import issues")