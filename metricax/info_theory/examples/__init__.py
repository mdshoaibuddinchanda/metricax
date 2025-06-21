"""
Information Theory Examples

Real-world applications demonstrating MetricaX Information Theory functions.

Available examples:
- entropy_example: Feature selection, model comparison, and uncertainty quantification
- mutual_info_example: Dependency analysis and feature ranking
- coding_example: Optimal coding and compression analysis
"""

# Import example modules for easy access
try:
    from . import entropy_example
    from . import mutual_info_example
    from . import coding_example
    
    __all__ = [
        "entropy_example",
        "mutual_info_example", 
        "coding_example",
    ]
except ImportError:
    # Handle cases where some examples might have missing dependencies
    __all__ = []
