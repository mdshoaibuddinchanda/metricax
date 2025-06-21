# Contributing to MetricaX

Welcome to the MetricaX contributor community! 🎉

We're building the world's premier mathematical toolkit for Python, and we'd love your help making it even better. Whether you're fixing bugs, adding features, improving documentation, or sharing ideas, every contribution matters.

## 🚀 Quick Start for Contributors

### 1. Set Up Development Environment

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/metricax.git
cd metricax

# Create virtual environment
python -m venv metricax_dev
source metricax_dev/bin/activate  # Linux/macOS
# metricax_dev\Scripts\activate   # Windows

# Install in development mode
pip install -e ".[dev]"

# Run tests to verify setup
pytest metricax/ -v
```

### 2. Make Your First Contribution

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make your changes
# ... edit files ...

# Run tests and linting
pytest metricax/ -v --cov=metricax
black metricax/
flake8 metricax/

# Commit and push
git add .
git commit -m "Add: your descriptive commit message"
git push origin feature/your-feature-name

# Create pull request on GitHub
```

## 🎯 Contribution Areas

### 🔥 High-Impact Contributions

1. **New Mathematical Modules**
   - `optimization/` - Gradient descent, genetic algorithms, simulated annealing
   - `statistics/` - Hypothesis testing, regression analysis, ANOVA
   - `time_series/` - ARIMA, exponential smoothing, trend analysis
   - `machine_learning/` - Core ML algorithms, cross-validation, metrics

2. **Function Enhancements**
   - Performance optimizations
   - Numerical stability improvements
   - Edge case handling
   - Better error messages

3. **Documentation & Examples**
   - Interactive Jupyter notebooks
   - Real-world use case examples
   - API documentation improvements
   - Tutorial content

### 🛠️ Technical Contributions

4. **Testing & Quality**
   - Unit test coverage expansion
   - Performance benchmarks
   - Integration tests
   - Property-based testing

5. **Infrastructure**
   - CI/CD pipeline improvements
   - Docker containerization
   - Package distribution
   - Performance monitoring

## 📋 Contribution Guidelines

### Code Style Standards

We follow strict coding standards to maintain professional quality:

```python
# ✅ Good: Clear, documented, validated function
def beta_pdf(x: float, alpha: float, beta: float) -> float:
    """
    Compute the probability density function of the Beta distribution.
    
    @formula: PDF(x; α, β) = (Γ(α + β) / (Γ(α) * Γ(β))) * x^(α-1) * (1-x)^(β-1)
    @source: Johnson, Kotz & Balakrishnan, Continuous Univariate Distributions
    
    Args:
        x: Value at which to evaluate PDF (must be in [0, 1])
        alpha: Shape parameter α > 0
        beta: Shape parameter β > 0
    
    Returns:
        Probability density at x
    
    Raises:
        ValueError: If parameters are invalid
    
    Examples:
        >>> beta_pdf(0.5, 2, 2)
        1.5
        >>> beta_pdf(0.3, 1, 1)
        1.0
    """
    # Input validation
    if not validate_prob(x):
        raise ValueError("x must be in [0, 1]")
    if alpha <= 0 or beta <= 0:
        raise ValueError("Alpha and beta must be positive")
    
    # Numerical computation with stability checks
    if x == 0 and alpha < 1:
        return float('inf')
    if x == 1 and beta < 1:
        return float('inf')
    
    # Main calculation
    log_pdf = ((alpha - 1) * math.log(x) + 
               (beta - 1) * math.log(1 - x) + 
               math.lgamma(alpha + beta) - 
               math.lgamma(alpha) - 
               math.lgamma(beta))
    
    return math.exp(log_pdf)

# ❌ Bad: Unclear, undocumented, unvalidated
def f(x, a, b):
    return (a-1)*math.log(x) + (b-1)*math.log(1-x)  # What does this do?
```

### Function Requirements

Every MetricaX function must include:

1. **Complete Docstring**
   - Mathematical formula with `@formula:` tag
   - Literature reference with `@source:` tag
   - Clear parameter descriptions with types
   - Return value specification
   - Usage examples
   - Error conditions

2. **Input Validation**
   - Type checking
   - Range validation
   - Edge case handling
   - Meaningful error messages

3. **Numerical Stability**
   - Avoid overflow/underflow
   - Use log-space when appropriate
   - Handle extreme values gracefully
   - Maintain precision

4. **Comprehensive Tests**
   - Unit tests for normal cases
   - Edge case testing
   - Error condition verification
   - Numerical precision validation

### Testing Standards

```python
# Example test structure
class TestBetaPDF:
    """Test beta_pdf function."""
    
    def test_uniform_distribution(self):
        """Test PDF of uniform Beta(1,1) distribution."""
        assert abs(beta_pdf(0.5, 1, 1) - 1.0) < 1e-10
        assert abs(beta_pdf(0.3, 1, 1) - 1.0) < 1e-10
    
    def test_known_values(self):
        """Test against known analytical values."""
        # Beta(2,2) at x=0.5 should equal 1.5
        assert abs(beta_pdf(0.5, 2, 2) - 1.5) < 1e-10
    
    def test_edge_cases(self):
        """Test boundary conditions."""
        # x = 0 with alpha < 1
        assert beta_pdf(0.0, 0.5, 2) == float('inf')
        
        # x = 1 with beta < 1  
        assert beta_pdf(1.0, 2, 0.5) == float('inf')
    
    def test_input_validation(self):
        """Test error handling."""
        with pytest.raises(ValueError):
            beta_pdf(-0.1, 1, 1)  # x out of range
        
        with pytest.raises(ValueError):
            beta_pdf(0.5, -1, 1)  # negative alpha
    
    def test_numerical_precision(self):
        """Test numerical stability."""
        # Very small parameters
        result = beta_pdf(0.001, 0.1, 0.1)
        assert math.isfinite(result)
        
        # Very large parameters
        result = beta_pdf(0.5, 100, 100)
        assert math.isfinite(result)
```

## 🏗️ Development Workflow

### Branch Naming Convention

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `test/description` - Test improvements
- `refactor/description` - Code refactoring

### Commit Message Format

```
Type: Brief description (50 chars max)

Detailed explanation of what and why (if needed).
Include references to issues: Fixes #123

Examples:
- Add: Shannon entropy function with numerical stability
- Fix: Handle edge case in beta_pdf when alpha < 1
- Docs: Add comprehensive examples for Bayesian module
- Test: Expand coverage for information theory functions
```

### Pull Request Process

1. **Before Submitting**
   ```bash
   # Run full test suite
   pytest metricax/ -v --cov=metricax --cov-report=html
   
   # Check code formatting
   black --check metricax/
   isort --check-only metricax/
   
   # Run linting
   flake8 metricax/
   mypy metricax/
   
   # Run security checks
   bandit -r metricax/
   ```

2. **PR Description Template**
   ```markdown
   ## Description
   Brief description of changes
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   - [ ] Performance improvement
   
   ## Testing
   - [ ] All existing tests pass
   - [ ] New tests added for new functionality
   - [ ] Manual testing completed
   
   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-review completed
   - [ ] Documentation updated
   - [ ] No breaking changes (or clearly documented)
   ```

3. **Review Process**
   - Automated CI checks must pass
   - Code review by maintainers
   - Documentation review
   - Performance impact assessment

## 🧪 Testing Guidelines

### Test Categories

1. **Unit Tests** - Individual function testing
2. **Integration Tests** - Module interaction testing
3. **Performance Tests** - Speed and memory benchmarks
4. **Property Tests** - Mathematical property verification

### Running Tests

```bash
# All tests
pytest metricax/ -v

# Specific module
pytest metricax/bayesian/tests/ -v

# With coverage
pytest metricax/ -v --cov=metricax --cov-report=html

# Performance benchmarks
pytest metricax/ --benchmark-only

# Parallel execution
pytest metricax/ -n auto
```

## 📚 Documentation Standards

### Module Documentation

Each module needs:
- `README.md` with overview and examples
- Complete API reference
- Interactive examples
- Mathematical background

### Function Documentation

```python
def function_name(param1: type, param2: type) -> return_type:
    """
    One-line summary of what the function does.
    
    Longer description explaining the mathematical concept,
    use cases, and any important implementation details.
    
    @formula: Mathematical formula in LaTeX-like notation
    @source: Academic reference or textbook citation
    
    Args:
        param1: Description with constraints and units
        param2: Description with valid ranges
    
    Returns:
        Description of return value with units/interpretation
    
    Raises:
        ValueError: When and why this error occurs
        TypeError: When and why this error occurs
    
    Examples:
        >>> function_name(1.0, 2.0)
        3.0
        >>> function_name(0.5, 1.5)
        2.0
    
    Note:
        Any important implementation notes or warnings.
    """
```

## 🎖️ Recognition

Contributors are recognized in multiple ways:

- **Contributors file** - All contributors listed
- **Release notes** - Major contributions highlighted  
- **Documentation credits** - Author attribution
- **Community recognition** - Social media shoutouts

## 🤝 Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inclusive environment:

- **Be respectful** - Treat everyone with kindness and professionalism
- **Be collaborative** - Work together to build something amazing
- **Be constructive** - Provide helpful feedback and suggestions
- **Be patient** - Remember that everyone is learning

### Getting Help

- **GitHub Issues** - Bug reports and feature requests
- **GitHub Discussions** - Questions and general discussion
- **Code Reviews** - Learning opportunity for everyone
- **Documentation** - Comprehensive guides and examples

## 🚀 Advanced Contributions

### Performance Optimization

```python
# Use profiling to identify bottlenecks
python -m cProfile -o profile.stats your_script.py
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(10)"

# Memory profiling
python -m memory_profiler your_script.py

# Benchmark comparisons
pytest metricax/ --benchmark-compare
```

### Adding New Modules

1. **Module Structure**
   ```
   metricax/new_module/
   ├── __init__.py          # Public API exports
   ├── README.md            # Module documentation
   ├── core_functions.py    # Main mathematical functions
   ├── utils.py            # Helper functions
   ├── examples/           # Usage examples
   │   ├── __init__.py
   │   └── basic_example.py
   └── tests/              # Test suite
       ├── __init__.py
       ├── test_core.py
       └── test_utils.py
   ```

2. **Integration Checklist**
   - [ ] Add to main `__init__.py`
   - [ ] Update documentation index
   - [ ] Add to CI/CD pipeline
   - [ ] Create example notebook
   - [ ] Update README.md

### Numerical Methods Best Practices

```python
# ✅ Good: Numerically stable implementation
def log_sum_exp(x):
    """Compute log(sum(exp(x))) in a numerically stable way."""
    x_max = max(x)
    return x_max + math.log(sum(math.exp(xi - x_max) for xi in x))

# ❌ Bad: Prone to overflow
def unstable_log_sum_exp(x):
    return math.log(sum(math.exp(xi) for xi in x))
```

---

**Ready to contribute?** Start by exploring our [good first issues](https://github.com/metricax/metricax/labels/good%20first%20issue) or join our [community discussions](https://github.com/metricax/metricax/discussions)!

Thank you for helping make MetricaX the best mathematical toolkit for Python! 🎯
