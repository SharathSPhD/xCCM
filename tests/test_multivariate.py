# tests/test_multivariate.py
"""Tests for multivariate CCM functionality."""

import pytest
import numpy as np
from xCCM.modules.multivariate import MultivariateCCM

def test_multivariate_initialization():
    """Test multivariate CCM initialization."""
    mccm = MultivariateCCM(embedding_dimension=3)
    assert mccm.embedding == 3
    assert mccm.tau == 1
    assert mccm.variables is None

def test_multivariate_embedding(multivariate_timeseries):
    """Test multivariate embedding creation."""
    mccm = MultivariateCCM(embedding_dimension=3)
    
    mv_embedding = mccm.create_multivariate_embedding(
        multivariate_timeseries,
        target_var='x'
    )
    
    expected_cols = (len(multivariate_timeseries) - 1) * 3
    assert mv_embedding.shape[1] == expected_cols

def test_multivariate_ccm(multivariate_timeseries):
    """Test multivariate CCM analysis."""
    mccm = MultivariateCCM(embedding_dimension=3)
    
    # Ensure time series have the same length
    min_length = min(len(ts) for ts in multivariate_timeseries.values())
    for key in multivariate_timeseries:
        multivariate_timeseries[key] = multivariate_timeseries[key][:min_length]
    
    results = mccm.run_multivariate_ccm(
        multivariate_timeseries,
        target_var='x',
        lib_sizes=[50, 100]
    )
    
    assert 'results' in results
    assert 'variables_used' in results
    assert 'target' in results
    assert results['target'] == 'x'
