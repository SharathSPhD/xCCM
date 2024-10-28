# Save as: xCCM/__init__.py

from .modules.core import EnhancedCCM, CCMResult
from .modules.multivariate import MultivariateCCM
from .modules.network import CCMNetwork
from .modules.nonlinear import NonlinearityTests
from .utils.preprocessing import DataPreprocessor
from .utils.visualization import CCMVisualizer
from .utils.performance import CCMPerformance

__version__ = '0.2.0'

__all__ = [
    'EnhancedCCM',
    'CCMResult',
    'MultivariateCCM',
    'CCMNetwork',
    'NonlinearityTests',
    'DataPreprocessor',
    'CCMVisualizer',
    'CCMPerformance'
]