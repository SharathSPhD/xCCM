# Save as: tests/conftest.py

import pytest
import numpy as np
from typing import Tuple, Dict

@pytest.fixture
def sample_timeseries() -> Tuple[np.ndarray, np.ndarray]:
    """Generate sample coupled logistic maps."""
    n = 1000
    r = 3.8
    b = 0.4
    
    x = np.zeros(n)
    y = np.zeros(n)
    
    x[0] = 0.4
    y[0] = 0.2
    
    for t in range(1, n):
        x[t] = r * x[t-1] * (1 - x[t-1])
        y[t] = r * y[t-1] * (1 - y[t-1]) + b * x[t-1]
        
    return x, y

@pytest.fixture
def multivariate_timeseries() -> Dict[str, np.ndarray]:
    """Generate sample multivariate time series."""
    n = 1000
    t = np.linspace(0, 10*np.pi, n)
    
    return {
        'x': np.sin(t),
        'y': np.sin(t + np.pi/4),
        'z': np.sin(t + np.pi/2)
    }