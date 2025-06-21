# Installation Guide

## Quick Installation

### From PyPI (Recommended)

```bash
pip install metricax
```

### From Source (Latest Features)

```bash
git clone https://github.com/metricax/metricax.git
cd metricax
pip install -e .
```

## System Requirements

### Python Version
- **Python 3.8+** (recommended: Python 3.10+)
- Compatible with Python 3.8, 3.9, 3.10, 3.11, 3.12

### Operating Systems
- ✅ **Linux** (Ubuntu 18.04+, CentOS 7+, etc.)
- ✅ **macOS** (10.14+)
- ✅ **Windows** (10+)

### Dependencies

MetricaX is designed to be **lightweight** with minimal dependencies:

#### Core Dependencies (Automatically Installed)
```
# No external dependencies - uses only Python standard library!
```

#### Optional Dependencies for Examples
```bash
# For running interactive examples and notebooks
pip install jupyter matplotlib seaborn pandas

# For development and testing
pip install pytest pytest-cov black flake8 mypy
```

## Installation Methods

### 1. Standard Installation
```bash
pip install metricax
```

### 2. Development Installation
```bash
git clone https://github.com/metricax/metricax.git
cd metricax
pip install -e ".[dev]"
```

### 3. With Optional Dependencies
```bash
pip install "metricax[examples]"  # Includes matplotlib, jupyter
pip install "metricax[dev]"       # Includes testing tools
pip install "metricax[all]"       # Includes everything
```

### 4. From Conda (Coming Soon)
```bash
conda install -c conda-forge metricax
```

## Verification

### Quick Test
```python
import metricax

# Test Bayesian module
import metricax.bayesian as mb
print(f"Beta mean: {mb.beta_mean(2, 3)}")  # Should print: 0.4

# Test Information Theory module  
import metricax.info_theory as it
print(f"Entropy: {it.entropy([0.5, 0.5])}")  # Should print: 1.0

print("✅ MetricaX installed successfully!")
```

### Run Test Suite
```bash
# Install with test dependencies
pip install "metricax[dev]"

# Run all tests
pytest metricax/ -v

# Run with coverage
pytest metricax/ -v --cov=metricax --cov-report=html
```

### Run Examples
```python
# Bayesian examples
from metricax.bayesian.examples import ab_testing
ab_testing.run_example()

# Information theory examples
from metricax.info_theory.examples import entropy_example
entropy_example.run_example()
```

## Troubleshooting

### Common Issues

#### ImportError: No module named 'metricax'
```bash
# Solution: Ensure proper installation
pip install --upgrade metricax

# Or for development:
pip install -e .
```

#### Permission Denied (Linux/macOS)
```bash
# Solution: Use user installation
pip install --user metricax

# Or use virtual environment (recommended)
python -m venv metricax_env
source metricax_env/bin/activate  # Linux/macOS
# metricax_env\Scripts\activate   # Windows
pip install metricax
```

#### Python Version Compatibility
```bash
# Check Python version
python --version

# Upgrade Python if needed (using pyenv)
pyenv install 3.10.0
pyenv global 3.10.0
```

### Performance Optimization

#### For Large-Scale Applications
```python
# MetricaX is optimized for performance
# No additional configuration needed

# For numerical stability with extreme values:
import metricax.bayesian.utils as utils
result = utils.safe_div(a, b, default=0.0)  # Handles division by zero
```

#### Memory Usage
```python
# MetricaX uses minimal memory
# All functions are stateless and memory-efficient

# For large datasets, process in batches:
def process_large_dataset(data, batch_size=1000):
    results = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        batch_result = metricax.info_theory.entropy(batch)
        results.append(batch_result)
    return results
```

## Virtual Environment Setup

### Using venv (Recommended)
```bash
# Create virtual environment
python -m venv metricax_env

# Activate (Linux/macOS)
source metricax_env/bin/activate

# Activate (Windows)
metricax_env\Scripts\activate

# Install MetricaX
pip install metricax

# Deactivate when done
deactivate
```

### Using conda
```bash
# Create conda environment
conda create -n metricax python=3.10

# Activate environment
conda activate metricax

# Install MetricaX
pip install metricax

# Deactivate when done
conda deactivate
```

## Docker Installation

### Using Official Docker Image (Coming Soon)
```bash
docker pull metricax/metricax:latest
docker run -it metricax/metricax:latest python
```

### Custom Dockerfile
```dockerfile
FROM python:3.10-slim

# Install MetricaX
RUN pip install metricax

# Copy your application
COPY . /app
WORKDIR /app

# Run your application
CMD ["python", "your_app.py"]
```

## IDE Integration

### VS Code
```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./metricax_env/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true
}
```

### PyCharm
1. File → Settings → Project → Python Interpreter
2. Add Interpreter → Existing Environment
3. Select `metricax_env/bin/python`

## Next Steps

After installation:

1. **📚 Read the [Quick Start Guide](../README.md#quick-start)**
2. **🎯 Try the [Examples](../examples/)**
3. **📖 Browse the [API Reference](../API_REFERENCE.md)**
4. **🔬 Run the [Interactive Notebooks](../notebooks/)**

## Support

- **📧 Issues**: [GitHub Issues](https://github.com/metricax/metricax/issues)
- **💬 Discussions**: [GitHub Discussions](https://github.com/metricax/metricax/discussions)
- **📖 Documentation**: [Full Documentation](https://metricax.readthedocs.io)

---

**Need help?** Open an issue on GitHub or check our [troubleshooting guide](troubleshooting.md).
