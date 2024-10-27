# xCCM/tests/test_core.py

import numpy as np
import pytest
from typing import Tuple
import warnings

def generate_test_data(n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Generate coupled logistic map data for testing."""
    x = np.zeros(n_points)
    y = np.zeros(n_points)
    
    # Initial conditions
    x[0] = 0.4
    y[0] = 0.2
    
    # Parameters
    rx, ry = 3.8, 3.5
    bxy, byx = 0.02, 0.1
    
    # Generate coupled time series
    for t in range(n_points-1):
        x[t+1] = x[t]*(rx - rx*x[t] - bxy*y[t])
        y[t+1] = y[t]*(ry - ry*y[t] - byx*x[t])
        
    return x, y

def generate_delayed_data(delay: int = 2, n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Generate time series with known delay."""
    x = np.random.randn(n_points)
    y = np.zeros(n_points)
    y[delay:] = x[:-delay]
    return x, y

class TestCore:
    """Test suite for core CCM functionality."""
    
    @pytest.fixture
    def ccm(self):
        """Create CCM instance for testing."""
        from xCCM.modules.core import EnhancedCCM
        return EnhancedCCM(embedding_dimension=3)
    
    def test_initialization(self, ccm):
        """Test initialization and parameter validation."""
        assert ccm.embedding == 3
        assert ccm.tau == 1
        assert ccm.noise_threshold == 0.1
        assert ccm.n_neighbors == 4  # E + 1
        
        # Test invalid parameters
        from xCCM.modules.core import EnhancedCCM
        with pytest.raises(ValueError, match="Embedding dimension must be at least 2"):
            EnhancedCCM(embedding_dimension=1)
            
        with pytest.raises(ValueError, match="Time delay must be positive"):
            EnhancedCCM(embedding_dimension=3, tau=0)
            
        with pytest.raises(ValueError, match="Noise threshold must be positive"):
            EnhancedCCM(embedding_dimension=3, noise_threshold=0)
            
    def test_decompose_signal(self, ccm):
        """Test signal decomposition."""
        # Generate test signal
        t = np.linspace(0, 10*np.pi, 100)
        x = np.sin(t) + 0.1 * np.random.randn(100)
        
        # Test basic decomposition
        signal, noise = ccm.decompose_signal(x)
        assert len(signal) == len(x)
        assert len(noise) == len(x)
        assert np.allclose(x, signal + noise)
        
        # Test window length validation
        with pytest.raises(ValueError):
            ccm.decompose_signal(x[:3])  # Too short
            
        with pytest.raises(ValueError):
            ccm.decompose_signal(x, window_length=2)  # Invalid window
            
        # Test non-finite values
        x_noisy = x.copy()
        x_noisy[0] = np.nan
        with warnings.catch_warnings(record=True) as w:
            signal, noise = ccm.decompose_signal(x_noisy)
            assert len(w) >= 1
            assert "Non-finite values detected" in str(w[-1].message)
            
        # Test SNR warning
        x_very_noisy = x + np.random.randn(len(x))*10
        with warnings.catch_warnings(record=True) as w:
            signal, noise = ccm.decompose_signal(x_very_noisy)
            assert len(w) >= 1
            assert "Low SNR detected" in str(w[-1].message)
            
    def test_create_embedding(self, ccm):
        """Test embedding creation."""
        x = np.sin(np.linspace(0, 10*np.pi, 100))
        
        # Test basic embedding
        embedding = ccm.create_embedding(x)
        assert embedding.shape[1] == 3  # E dimensions
        assert embedding.shape[0] == len(x) - 2  # Correct length
        
        # Test short time series
        with pytest.raises(ValueError):
            ccm.create_embedding(x[:2])
            
        # Test with lag
        embedding_lag = ccm.create_embedding(x, predict_lag=2)
        assert embedding_lag.shape == embedding.shape
            
    def test_compute_weights(self, ccm):
        """Test weight computation."""
        target = np.zeros(3)
        neighbors = np.array([[1, 1, 1], [2, 2, 2]])
        distances = np.array([1.0, 2.0])
        
        # Test basic weight computation
        weights = ccm.compute_manifold_weights(target, neighbors, distances)
        assert len(weights) == len(distances)
        assert np.isclose(np.sum(weights), 1.0)
        assert weights[0] > weights[1]  # Closer points get higher weights
        
        # Test invalid inputs
        with pytest.raises(ValueError):
            ccm.compute_manifold_weights(target, np.array([]), np.array([]))
            
        with pytest.raises(ValueError):
            ccm.compute_manifold_weights(target, neighbors, distances, theta=0)
            
    def test_predict_weighted(self, ccm):
        """Test weighted prediction."""
        x = np.sin(np.linspace(0, 10*np.pi, 100))
        manifold = ccm.create_embedding(x)
        
        # Test insufficient library size
        with pytest.raises(ValueError):
            ccm.predict_weighted(manifold, 0, np.array([0]))
            
        # Test valid prediction
        lib_indices = np.arange(10)
        pred, weights = ccm.predict_weighted(manifold, 0, lib_indices)
        assert isinstance(pred, float)
        assert len(weights) == ccm.n_neighbors
        assert np.isclose(np.sum(weights), 1.0)
        
    def test_run_ccm(self, ccm):
        """Test full CCM analysis."""
        x, y = generate_test_data(300)  # Shorter series for speed
        
        # Test input validation
        with pytest.raises(ValueError):
            ccm.run_ccm(x[:-1], y, [10])  # Unequal lengths
            
        with pytest.raises(ValueError):
            ccm.run_ccm(x[:5], y[:5], [10])  # Too short
            
        # Test basic CCM
        lib_sizes = np.array([50, 100, 150])
        results = ccm.run_ccm(x, y, lib_sizes)
        
        assert len(results) == len(lib_sizes)
        assert all(isinstance(r.correlation, float) for r in results)
        assert all(-1 <= r.correlation <= 1 for r in results)
        assert all(len(r.predictions) > 0 for r in results)
        
        # Test convergence
        corrs = [r.correlation for r in results]
        assert corrs[-1] >= corrs[0]  # Should generally increase
        
        # Test with lag
        results_lag = ccm.run_ccm(x, y, lib_sizes, lag=2)
        assert len(results_lag) == len(lib_sizes)
        
    def test_time_delay_detection(self, ccm):
        """Test time delay detection."""
        x, y = generate_delayed_data(delay=2, n_points=300)
        lib_sizes = [100]
        
        # Test range of lags
        lags = range(-5, 5)
        cross_map_skills = []
        
        for lag in lags:
            result = ccm.run_ccm(x, y, lib_sizes, lag=lag)[0]
            cross_map_skills.append(result.correlation)
            
        # Find optimal lag
        optimal_lag = lags[np.argmax(cross_map_skills)]
        assert abs(optimal_lag + 2) <= 1  # Should be close to -2
        
        # Test different delays
        for delay in [1, 3, 4]:
            x, y = generate_delayed_data(delay=delay, n_points=300)
            cross_map_skills = []
            
            for lag in range(-5, 5):
                result = ccm.run_ccm(x, y, [100], lag=lag)[0]
                cross_map_skills.append(result.correlation)
                
            optimal_lag = np.argmax(cross_map_skills) - 5
            assert abs(optimal_lag + delay) <= 1

if __name__ == '__main__':
    pytest.main([__file__, '-v'])