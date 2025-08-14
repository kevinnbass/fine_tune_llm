"""
Master Test Suite Runner - Runs all test suites sequentially
Run with: python tests/suites/run_all_suites.py
"""

import subprocess
import sys
import time
from pathlib import Path

def run_suite(suite_name, suite_file):
    """Run a specific test suite."""
    print(f"\n{'='*80}")
    print(f"STARTING {suite_name.upper()}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run([
            sys.executable, suite_file
        ], cwd=".", timeout=1800)  # 30 minute timeout per suite
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"SUCCESS: {suite_name} completed successfully in {duration:.1f}s")
            return True, duration
        else:
            print(f"FAILED: {suite_name} failed in {duration:.1f}s")
            return False, duration
            
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time  
        print(f"TIMEOUT: {suite_name} timed out after {duration:.1f}s")
        return False, duration
    except Exception as e:
        duration = time.time() - start_time
        print(f"EXCEPTION: {suite_name} exception: {e}")
        return False, duration

def main():
    """Run all test suites in sequence."""
    print("="*80)
    print("COMPREHENSIVE TEST SUITE EXECUTION")
    print("="*80)
    print("Running test suites in optimal order for dependency management...")
    
    # Test suites in order of complexity/dependencies
    suites = [
        ("Core Suite", "tests/suites/core_suite.py"),
        ("ML Suite", "tests/suites/ml_suite.py"), 
        ("Scripts Suite", "tests/suites/scripts_suite.py")
    ]
    
    results = []
    total_time = time.time()
    
    for suite_name, suite_file in suites:
        success, duration = run_suite(suite_name, suite_file)
        results.append((suite_name, success, duration))
        
        # Brief pause between suites to clear memory
        time.sleep(2)
    
    total_duration = time.time() - total_time
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL COMPREHENSIVE TEST RESULTS")
    print(f"{'='*80}")
    
    successful_suites = 0
    failed_suites = 0
    
    for suite_name, success, duration in results:
        status = "PASSED" if success else "FAILED"
        print(f"{status} {suite_name}: {duration:.1f}s")
        
        if success:
            successful_suites += 1
        else:
            failed_suites += 1
    
    print(f"\nSUMMARY:")
    print(f"   • Total Suites: {len(suites)}")
    print(f"   • Successful: {successful_suites}")
    print(f"   • Failed: {failed_suites}")
    print(f"   • Success Rate: {successful_suites/len(suites)*100:.1f}%")
    print(f"   • Total Time: {total_duration:.1f}s ({total_duration/60:.1f} minutes)")
    
    if successful_suites == len(suites):
        print(f"\nALL TEST SUITES PASSED! System is fully validated.")
        return 0
    else:
        print(f"\nWARNING: {failed_suites} suite(s) had issues. Review individual results above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)