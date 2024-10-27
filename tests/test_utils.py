# tests/test_utils.py
"""Tests for utility functions."""

import pytest
import numpy as np
from xCCM.utils.preprocessing import DataPreprocessor
from xCCM.utils.visualization import CCMVisualizer
from xCCM.utils.performance import CCMPerformance
import matplotlib.pyplot as plt
from xCCM.modules.core import EnhancedCCM

def test_data_preprocessing(sample_timeseries):
    """Test data preprocessing utilities."""
    x, _ = sample_timeseries
    
    # Test standardization
    standardized = DataPreprocessor.standardize(x)
    assert np.allclose(np.mean(standardized), 0, atol=1e-10)
    assert np.allclose(np.std(standardized), 1, atol=1e-10)
    
    # Test detrending
    detrended, trend = DataPreprocessor.detrend(x)
    assert len(detrended) == len(x)
    assert len(trend) == len(x)
    
    # Test outlier filtering
    filtered, mask = DataPreprocessor.filter_outliers(x)
    assert len(filtered) <= len(x)
    assert len(mask) == len(x)

def test_visualization(sample_timeseries, sample_ccm_results):
    """Test visualization utilities."""
    x, y = sample_timeseries
    viz = CCMVisualizer()
    
    # Test convergence plot
    lib_sizes = np.arange(10, 100, 10)
    correlations = np.random.rand(len(lib_sizes))
    fig = viz.plot_convergence(lib_sizes, correlations)
    assert isinstance(fig, plt.Figure)
    
    # Test network plot
    fig = viz.plot_causality_network(sample_ccm_results)
    assert isinstance(fig, plt.Figure)

def test_performance(sample_timeseries):
    """Test performance optimization utilities."""
    x, y = sample_timeseries
    perf = CCMPerformance(n_jobs=2)
    ccm = EnhancedCCM(embedding_dimension=3)
    
    # Test parallel computation
    lib_sizes = np.arange(10, 100, 10)
    results = perf.optimize_computation(ccm, x, y, lib_sizes)
    assert len(results) == len(lib_sizes)
    
    # Test memory-efficient computation
    results = perf.memory_efficient_ccm(ccm, x, y, max_lib_size=100)
    assert len(results) == 100

if __name__ == '__main__':
    pytest.main([__file__])