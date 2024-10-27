# xCCM/tests/test_nonlinear.py

import numpy as np
import pytest
from xCCM.modules.nonlinear import NonlinearityTests, reconstruct

class TestNonlinearityTests:
    """Test suite for nonlinearity testing functionality."""
    
    @pytest.fixture
    def nlt(self):
        """Create NonlinearityTests instance."""
        return NonlinearityTests(n_surrogates=50)  # Reduced for speed
        
    @pytest.fixture
    def nonlinear_data(self):
        """Generate nonlinear time series (logistic map)."""
        n = 500  # Reduced length for speed
        r = 3.9  # Chaotic regime
        x = np.zeros(n)
        x[0] = 0.4
        
        for t in range(1, n):
            x[t] = r * x[t-1] * (1 - x[t-1])
            
        return x
        
    @pytest.fixture
    def linear_data(self):
        """Generate linear time series (AR process)."""
        n = 500  # Reduced length for speed
        ar = 0.7
        x = np.zeros(n)
        x[0] = np.random.randn()
        
        for t in range(1, n):
            x[t] = ar * x[t-1] + 0.1 * np.random.randn()
            
        return x

    def test_reconstruct(self):
        """Test time delay embedding reconstruction."""
        # Generate simple sine wave
        t = np.linspace(0, 4*np.pi, 200)
        x = np.sin(t)
        
        # Test basic reconstruction
        embedding = reconstruct(x, dimension=3, tau=2)
        assert embedding.shape[1] == 3  # Check dimensions
        assert embedding.shape[0] == len(x) - (3-1)*2  # Check length
        
        # Test error for short time series
        with pytest.raises(ValueError):
            reconstruct(x[:4], dimension=3, tau=2)
            
        # Test preservation of dynamics
        embedding_2d = reconstruct(x, dimension=2, tau=25)  # Quarter period
        # Check if points roughly form an ellipse
        corr = np.abs(np.corrcoef(embedding_2d[:, 0], embedding_2d[:, 1])[0, 1])
        assert corr > 0.3  # Relaxed threshold

    def test_surrogate_generation(self, nlt, nonlinear_data):
        """Test surrogate time series generation."""
        # Test IAAFT method
        surrogate = nlt.generate_surrogate(nonlinear_data, method='iaaft')
        assert len(surrogate) == len(nonlinear_data)
        assert not np.allclose(surrogate, nonlinear_data)
        assert np.allclose(np.sort(surrogate), np.sort(nonlinear_data))
        
        # Test random shuffle method
        surrogate = nlt.generate_surrogate(nonlinear_data, method='random_shuffle')
        assert len(surrogate) == len(nonlinear_data)
        assert not np.allclose(surrogate, nonlinear_data)
        assert np.allclose(np.sort(surrogate), np.sort(nonlinear_data))
        
        # Test invalid method
        with pytest.raises(ValueError):
            nlt.generate_surrogate(nonlinear_data, method='invalid')
            
    def test_time_reversal_asymmetry(self, nlt):
        """Test time reversal asymmetry statistic."""
        # Test on symmetric time series
        t = np.linspace(0, 4*np.pi, 200)
        x_sym = np.sin(t)
        tra_sym = nlt.time_reversal_asymmetry(x_sym)
        assert abs(tra_sym) < 1e-5  # Should be close to 0
        
        # Test on asymmetric time series
        x_asym = np.exp(-t/5) * np.sin(t)  # Decaying sine
        tra_asym = nlt.time_reversal_asymmetry(x_asym)
        assert abs(tra_asym) > 0.01  # Should be significantly non-zero
        
        # Test different lag values
        tra_lag1 = nlt.time_reversal_asymmetry(x_asym, lag=1)
        tra_lag2 = nlt.time_reversal_asymmetry(x_asym, lag=2)
        assert tra_lag1 != tra_lag2  # Should be different
        
        # Test short time series
        with pytest.raises(ValueError):
            nlt.time_reversal_asymmetry(x_sym[:3], lag=1)
        
    def test_nonlinearity_test(self, nlt, nonlinear_data, linear_data):
        """Test comprehensive nonlinearity testing."""
        # Test on nonlinear data
        is_nonlinear, pvalue, details = nlt.test_nonlinearity(
            nonlinear_data,
            embedding_dim=3  # Fixed dimension
        )
        assert isinstance(is_nonlinear, bool)
        assert 0 <= pvalue <= 1
        assert details['embedding_dimension'] > 0
        assert details['n_surrogates'] == nlt.n_surrogates
        assert is_nonlinear  # Should detect nonlinearity
        
        # Test on linear data
        is_nonlinear, pvalue, details = nlt.test_nonlinearity(
            linear_data,
            embedding_dim=3  # Fixed dimension
        )
        assert not is_nonlinear  # Should not detect nonlinearity
        
        # Test with specified vs estimated dimension
        _, _, details1 = nlt.test_nonlinearity(nonlinear_data, embedding_dim=5)
        _, _, details2 = nlt.test_nonlinearity(nonlinear_data)
        assert details1['embedding_dimension'] == 5
        assert details2['embedding_dimension'] > 0
        
    def test_embedding_dimension_estimation(self, nlt, nonlinear_data, linear_data):
        """Test embedding dimension estimation."""
        # Test on nonlinear data
        dim_nonlinear = nlt._estimate_embedding_dimension(
            nonlinear_data[:200]  # Using subset for speed
        )
        assert 2 <= dim_nonlinear <= 5  # Reasonable range for logistic map
        
        # Test on linear data
        dim_linear = nlt._estimate_embedding_dimension(
            linear_data[:200]  # Using subset for speed
        )
        assert 1 <= dim_linear <= 3  # Reasonable range for AR(1)
        
        # Test parameter sensitivity
        dim_low_thresh = nlt._estimate_embedding_dimension(
            nonlinear_data[:200], 
            threshold=0.05
        )
        dim_high_thresh = nlt._estimate_embedding_dimension(
            nonlinear_data[:200], 
            threshold=0.2
        )
        assert dim_low_thresh >= dim_high_thresh
        
        # Test max dimension limit
        dim_max = nlt._estimate_embedding_dimension(
            nonlinear_data[:200], 
            max_dim=3
        )
        assert dim_max <= 3
        
    def test_end_to_end(self, nlt):
        """Test complete workflow on known systems."""
        # Test on logistic map (known to be nonlinear)
        n = 300  # Reduced length for speed
        r = 3.9
        x = np.zeros(n)
        x[0] = 0.4
        for t in range(1, n):
            x[t] = r * x[t-1] * (1 - x[t-1])
            
        is_nonlinear, _, _ = nlt.test_nonlinearity(x, embedding_dim=3)
        assert is_nonlinear
        
        # Test on AR process (linear)
        ar = np.zeros(n)
        ar[0] = np.random.randn()
        for t in range(1, n):
            ar[t] = 0.7 * ar[t-1] + 0.1 * np.random.randn()
            
        is_nonlinear, _, _ = nlt.test_nonlinearity(ar, embedding_dim=3)
        assert not is_nonlinear
        
        # Test on sine wave (periodic)
        t = np.linspace(0, 8*np.pi, n)
        sine = np.sin(t)
        is_nonlinear, _, _ = nlt.test_nonlinearity(sine, embedding_dim=3)
        assert not is_nonlinear  # Sine wave is linear in phase space

if __name__ == '__main__':
    pytest.main([__file__, '-v'])