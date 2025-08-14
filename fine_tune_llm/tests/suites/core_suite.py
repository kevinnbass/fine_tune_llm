"""
Core Test Suite - Essential functionality tests
Run with: python -m pytest tests/suites/core_suite.py -v
"""

# Core functionality tests - lightweight and essential
CORE_TESTS = [
    "tests/test_dataset.py",           # Data processing pipeline
    "tests/test_integration.py",       # System integration  
    "tests/test_high_stakes_simple.py" # Simplified high-stakes features
]

if __name__ == "__main__":
    import subprocess
    import sys
    import time
    
    print("=" * 60)
    print("RUNNING CORE TEST SUITE")
    print("=" * 60)
    
    total_passed = 0
    total_failed = 0
    total_skipped = 0
    
    for test_file in CORE_TESTS:
        print(f"\n{'='*20} Running {test_file} {'='*20}")
        start_time = time.time()
        
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            test_file, "-v", "--tb=short", "-x"
        ], capture_output=True, text=True, cwd=".")
        
        duration = time.time() - start_time
        
        print(f"Duration: {duration:.2f}s")
        print("STDOUT:", result.stdout.split('\n')[-3:])  # Last few lines
        
        # Parse results
        if "failed" in result.stdout:
            print(f"FAILED: {test_file}")
            failed_count = 1
        else:
            print(f"PASSED: {test_file}")
            failed_count = 0
            
        # Extract counts from pytest output
        lines = result.stdout.split('\n')
        for line in lines:
            if " passed" in line and " in " in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "passed":
                        total_passed += int(parts[i-1])
                    elif part == "failed":
                        total_failed += int(parts[i-1])  
                    elif part == "skipped":
                        total_skipped += int(parts[i-1])
                break
        
        if result.returncode != 0 and failed_count > 0:
            print(f"STDERR: {result.stderr}")
            
    print("\n" + "="*60)
    print("CORE SUITE SUMMARY")
    print("="*60)
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    print(f"Skipped: {total_skipped}")
    print(f"Success Rate: {total_passed/(total_passed+total_failed)*100:.1f}%" if (total_passed+total_failed) > 0 else "No tests run")