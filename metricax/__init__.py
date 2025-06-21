"""
MetricaX - Professional Mathematical and Statistical Toolkit for Python.

This package provides production-ready tools for mathematical analysis including:
- Bayesian statistics and inference
- Information theory and entropy measures
- Numerical utilities and validation
"""

__version__ = "0.1.0"
__author__ = "MetricaX Team"

# Import mathematical modules
from . import bayesian
from . import info_theory

__all__ = [
    "bayesian",
    "info_theory",
]
