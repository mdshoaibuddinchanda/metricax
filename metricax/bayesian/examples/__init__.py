"""
Bayesian Statistics Examples

Real-world applications demonstrating MetricaX Bayesian functions.

Available examples:
- ab_testing: A/B testing with conversion rate analysis
- spam_filter: Bayesian email classification
- data_updates: Online learning scenarios (quality control, sensor calibration, etc.)
"""

# Import example modules for easy access
try:
    from . import ab_testing
    from . import spam_filter  
    from . import data_updates
    
    __all__ = [
        "ab_testing",
        "spam_filter", 
        "data_updates",
    ]
except ImportError:
    # Handle cases where some examples might have missing dependencies
    __all__ = []
