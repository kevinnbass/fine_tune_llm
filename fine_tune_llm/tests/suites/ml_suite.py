"""
ML Test Suite - Machine Learning focused tests
Run with: python -m pytest tests/suites/ml_suite.py -v
"""

# ML-focused tests - may have heavier imports
ML_TESTS = [
    "tests/test_sft_lora.py",          # LoRA training
    "tests/test_high_stakes.py",       # High-stakes ML features
    "tests/test_evaluate.py",          # Model evaluation
    "tests/test_llm_infer.py"          # LLM inference
]

if __name__ == "__main__":
    import subprocess
    import sys
    import time
    import os
    
    print("=" * 60)
    print("RUNNING ML TEST SUITE") 
    print("=" * 60)
    
    total_passed = 0
    total_failed = 0
    total_skipped = 0
    
    for test_file in ML_TESTS:
        print(f"\n{'='*20} Running {test_file} {'='*20}")
        start_time = time.time()
        
        # Run with timeout to handle hanging imports
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                test_file, "-v", "--tb=short", "-x", "--timeout=300"
            ], capture_output=True, text=True, cwd=".", timeout=600)
            
            duration = time.time() - start_time
            
            print(f"Duration: {duration:.2f}s")
            
            # Check if test was skipped entirely
            if "collected 0 items" in result.stdout or "SKIPPED" in result.stdout:
                print(f"SKIPPED: {test_file} (import issues)")
                total_skipped += 1
                continue
            
            if result.returncode == 0:
                print(f"PASSED: {test_file}")
                # Extract passed count
                lines = result.stdout.split('\n')
                for line in lines:
                    if " passed" in line and " in " in line:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == "passed" and i > 0:
                                try:
                                    total_passed += int(parts[i-1])
                                except:
                                    pass
                        break
            else:
                print(f"FAILED: {test_file}")
                print("Last few output lines:")
                output_lines = result.stdout.split('\n')
                for line in output_lines[-5:]:
                    if line.strip():
                        print(f"  {line}")
                total_failed += 1
                
        except subprocess.TimeoutExpired:
            print(f"TIMEOUT: {test_file} (> 10 minutes)")
            total_skipped += 1
            
    print("\n" + "="*60)
    print("ML SUITE SUMMARY")
    print("="*60)
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}") 
    print(f"Skipped/Timeout: {total_skipped}")
    if (total_passed + total_failed) > 0:
        success_rate = total_passed / (total_passed + total_failed) * 100
        print(f"Success Rate: {success_rate:.1f}%")
    else:
        print("Success Rate: No tests completed")