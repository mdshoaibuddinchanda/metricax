# Contributing to MetricaX 🤝

Thank you for your interest in contributing to MetricaX! This guide will help you add new modules, functions, or improvements to the library.

## 🏗️ **Adding New Mathematical Modules**

MetricaX uses a **self-contained module structure** that makes it easy to add new mathematical topics. Each module is completely independent with its own code, examples, and tests.

### **Step 1: Create Module Structure**

```bash
metricax/metricax/your_module/
├── __init__.py              # Export functions
├── core_functions.py        # Main mathematical functions
├── utils.py                 # Module-specific utilities (optional)
├── examples/                # Real-world applications
│   ├── __init__.py
│   ├── README.md
│   ├── example1.py
│   └── example2.py
└── tests/                   # Comprehensive test suite
    ├── __init__.py
    ├── test_core_functions.py
    └── test_utils.py
```

### **Step 2: Implement Core Functions**

Follow these standards for all functions:

```python
def your_function(param1: float, param2: int) -> float:
    """
    Brief description of what the function does.
    
    Detailed explanation of the mathematical concept,
    including formulas and use cases.
    
    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When and why this is raised
        
    Examples:
        >>> your_function(1.0, 2)
        3.0
        >>> your_function(0.5, 1)
        1.5
    """
    # Input validation
    if not math.isfinite(param1):
        raise ValueError("param1 must be finite")
    if param2 <= 0:
        raise ValueError("param2 must be positive")
    
    # Implementation with numerical stability
    result = param1 * param2  # Your math here
    
    return result
```

### **Step 3: Create Real-World Examples**

Each module should have 2-3 practical examples:

```python
#!/usr/bin/env python3
"""
Real-World Application: [Title]

Description of the practical scenario this solves.
"""

# Import from parent module
from .. import your_function, other_function

def practical_example():
    """
    Demonstrate real-world usage with business context.
    """
    print("🎯 [Example Title]")
    print("=" * 40)
    
    # Real data and scenario
    result = your_function(data, parameters)
    
    # Business interpretation
    print(f"Result: {result}")
    print("💡 Business Impact: [explanation]")
    
    return result

def run_example():
    """Main function to run the example"""
    return practical_example()

if __name__ == "__main__":
    run_example()
```

### **Step 4: Write Comprehensive Tests**

```python
import pytest
import math
from .. import your_function

class TestYourFunction:
    """Test cases for your_function"""
    
    def test_basic_functionality(self):
        """Test normal operation"""
        assert your_function(2.0, 3) == 6.0
        
    def test_edge_cases(self):
        """Test boundary conditions"""
        assert your_function(0.0, 1) == 0.0
        
    def test_invalid_input(self):
        """Test error handling"""
        with pytest.raises(ValueError):
            your_function(float('nan'), 1)
```

### **Step 5: Update Main Package**

Add your module to `metricax/__init__.py`:

```python
from . import bayesian
from . import your_module  # Add this line

__all__ = [
    "bayesian",
    "your_module",  # Add this line
]
```

## 📚 **Module Ideas for Contribution**

### **High-Impact Modules:**
- **`optimization/`** - Gradient descent, genetic algorithms, simulated annealing
- **`statistics/`** - Hypothesis testing, regression, ANOVA
- **`time_series/`** - ARIMA, exponential smoothing, trend analysis
- **`machine_learning/`** - Basic ML algorithms, cross-validation
- **`finance/`** - Options pricing, risk metrics, portfolio optimization

### **Specialized Modules:**
- **`signal_processing/`** - FFT, filtering, spectral analysis
- **`graph_theory/`** - Network analysis, shortest paths, centrality
- **`numerical/`** - Integration, differentiation, root finding
- **`probability/`** - Distribution fitting, random sampling

## 🎯 **Quality Standards**

### **Code Quality:**
- ✅ Type hints for all parameters and returns
- ✅ Comprehensive docstrings with examples
- ✅ Input validation and error handling
- ✅ Numerical stability considerations
- ✅ Pure Python (avoid heavy dependencies)

### **Documentation:**
- ✅ Real-world examples with business context
- ✅ Mathematical background explanation
- ✅ Clear usage instructions
- ✅ Performance considerations

### **Testing:**
- ✅ Unit tests for all functions
- ✅ Edge case testing
- ✅ Error condition testing
- ✅ Integration tests for examples

## 🚀 **Submission Process**

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-module`
3. **Implement following the structure above**
4. **Test thoroughly**: `python -m pytest metricax/your_module/tests/`
5. **Update documentation**
6. **Submit pull request** with:
   - Clear description of the module
   - Real-world use cases
   - Test results
   - Documentation updates

## 💡 **Tips for Success**

### **Mathematical Accuracy:**
- Reference authoritative sources (textbooks, papers)
- Include citations in docstrings
- Validate against known results
- Consider numerical precision

### **User Experience:**
- Focus on practical applications
- Provide clear error messages
- Include performance notes
- Make examples copy-pasteable

### **Professional Standards:**
- Follow existing code style
- Use consistent naming conventions
- Include comprehensive tests
- Document edge cases and limitations

## 🆘 **Getting Help**

- **Questions**: Open a GitHub issue with the "question" label
- **Discussions**: Use GitHub Discussions for design decisions
- **Examples**: Look at the `bayesian/` module as a reference
- **Testing**: Check existing test files for patterns

---

**Thank you for helping make MetricaX better! 🎉**
