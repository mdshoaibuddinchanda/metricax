#!/usr/bin/env python3
"""
Basic functionality test for MetricaX library.
This script tests core functionality to ensure the library works correctly.
"""

import sys
import traceback

def test_bayesian_module():
    """Test basic Bayesian module functionality."""
    try:
        import metricax.bayesian as mb
        
        # Test basic functions
        result1 = mb.beta_mean(2, 3)
        result2 = mb.beta_var(2, 3)
        result3 = mb.beta_pdf(0.5, 2, 3)
        
        print(f"✅ Bayesian module works:")
        print(f"  - beta_mean(2, 3) = {result1:.3f}")
        print(f"  - beta_var(2, 3) = {result2:.3f}")
        print(f"  - beta_pdf(0.5, 2, 3) = {result3:.3f}")
        
        return True
    except Exception as e:
        print(f"❌ Bayesian module failed: {e}")
        traceback.print_exc()
        return False

def test_info_theory_module():
    """Test basic Information Theory module functionality."""
    try:
        import metricax.info_theory as it
        
        # Test basic functions
        result1 = it.entropy([0.5, 0.5])
        result2 = it.kl_divergence([0.5, 0.5], [0.3, 0.7])
        result3 = it.cross_entropy([0.5, 0.5], [0.3, 0.7])
        
        print(f"✅ Information Theory module works:")
        print(f"  - entropy([0.5, 0.5]) = {result1:.3f}")
        print(f"  - kl_divergence([0.5, 0.5], [0.3, 0.7]) = {result2:.3f}")
        print(f"  - cross_entropy([0.5, 0.5], [0.3, 0.7]) = {result3:.3f}")
        
        return True
    except Exception as e:
        print(f"❌ Information Theory module failed: {e}")
        traceback.print_exc()
        return False

def test_examples():
    """Test that examples can be imported."""
    try:
        # Test Bayesian examples
        from metricax.bayesian.examples import ab_testing
        print("✅ Bayesian examples import successfully")
        
        # Test Info Theory examples
        from metricax.info_theory.examples import entropy_example
        print("✅ Information Theory examples import successfully")
        
        return True
    except Exception as e:
        print(f"❌ Examples failed to import: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all basic functionality tests."""
    print("🔬 MetricaX Basic Functionality Test")
    print("=" * 50)
    
    all_passed = True
    
    # Test core modules
    all_passed &= test_bayesian_module()
    print()
    all_passed &= test_info_theory_module()
    print()
    all_passed &= test_examples()
    print()
    
    # Summary
    if all_passed:
        print("🎉 All basic functionality tests PASSED!")
        print("✅ MetricaX library is working correctly")
        sys.exit(0)
    else:
        print("❌ Some tests FAILED!")
        print("🔧 Library needs fixes before publication")
        sys.exit(1)

if __name__ == "__main__":
    main()
