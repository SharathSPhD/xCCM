# xCCM/modules/nonlinear.py

import numpy as np
from typing import Tuple, List, Optional, Dict
from scipy.stats import norm
from scipy.signal import welch
from sklearn.neighbors import NearestNeighbors

def reconstruct(time_series: np.ndarray, dimension: int, tau: int = 1) -> np.ndarray:
    """
    Create time-delay embedding of a time series.
    
    Args:
        time_series: Input time series
        dimension: Embedding dimension
        tau: Time delay
        
    Returns:
        Embedded manifold
    """
    time_series = np.asarray(time_series)
    n = len(time_series)
    
    if n < dimension * tau:
        raise ValueError("Time series too short for embedding parameters")
        
    # Calculate number of vectors
    n_vectors = n - (dimension-1)*tau
    
    # Pre-allocate embedding array
    embedding = np.zeros((n_vectors, dimension))
    
    # Fill embedding array
    for i in range(dimension):
        embedding[:, i] = time_series[i*tau:i*tau + n_vectors]
            
    return embedding

class NonlinearityTests:
    """Tests for detecting nonlinearity in time series."""
    
    def __init__(self, n_surrogates: int = 100) -> None:
        """
        Initialize NonlinearityTests.
        
        Args:
            n_surrogates: Number of surrogate time series to generate
        """
        self.n_surrogates = n_surrogates
        
    def generate_surrogate(
        self,
        time_series: np.ndarray,
        method: str = 'iaaft'
    ) -> np.ndarray:
        """
        Generate surrogate time series.
        
        Args:
            time_series: Original time series
            method: Surrogate generation method ('iaaft' or 'random_shuffle')
            
        Returns:
            Surrogate time series
        """
        time_series = np.asarray(time_series)
        
        if method == 'iaaft':
            # Iterative Amplitude Adjusted Fourier Transform
            fourier = np.fft.rfft(time_series)
            amplitudes = np.abs(fourier)
            
            # Randomize phases
            random_phases = np.random.uniform(0, 2*np.pi, len(fourier))
            fourier_random = amplitudes * np.exp(1j * random_phases)
            
            surrogate = np.fft.irfft(fourier_random, len(time_series))
            
            # Match original distribution
            surrogate.sort()
            orig_sorted = np.sort(time_series)
            surrogate = np.interp(
                np.linspace(0, 1, len(time_series)),
                np.linspace(0, 1, len(orig_sorted)),
                orig_sorted
            )
            
        elif method == 'random_shuffle':
            surrogate = np.random.permutation(time_series)
            
        else:
            raise ValueError(f"Unknown surrogate method: {method}")
            
        return surrogate
    
    def time_reversal_asymmetry(
        self,
        time_series: np.ndarray,
        lag: int = 1
    ) -> float:
        """
        Compute time reversal asymmetry statistic.
        
        Args:
            time_series: Input time series
            lag: Time lag
            
        Returns:
            Time reversal asymmetry statistic
        """
        time_series = np.asarray(time_series)
        n = len(time_series)
        
        if n < 2*lag + 1:
            raise ValueError("Time series too short for given lag")
            
        # Compute forward and backward segments
        forward = time_series[lag:n-lag]
        backward = time_series[0:n-2*lag]
        middle = time_series[lag:n-lag]
        
        # Ensure all segments have same length
        min_len = min(len(forward), len(backward), len(middle))
        forward = forward[:min_len]
        backward = backward[:min_len]
        middle = middle[:min_len]
        
        # Compute statistic
        asym = np.mean(forward * forward * backward - 
                      forward * middle * middle)
        return asym
    
    def test_nonlinearity(
        self,
        time_series: np.ndarray,
        embedding_dim: Optional[int] = None,
        significance_level: float = 0.05
    ) -> Tuple[bool, float, Dict]:
        """
        Comprehensive nonlinearity test.
        
        Args:
            time_series: Input time series
            embedding_dim: Embedding dimension (if None, estimated)
            significance_level: Significance level for testing
            
        Returns:
            Tuple of (is_nonlinear, p_value, test_details)
        """
        time_series = np.asarray(time_series)
        
        # Estimate embedding dimension if not provided
        if embedding_dim is None:
            embedding_dim = self._estimate_embedding_dimension(time_series)
            
        # Compute test statistics
        tra_stat = self.time_reversal_asymmetry(time_series)
        
        # Generate surrogates and compute statistics
        surrogate_stats = []
        for _ in range(self.n_surrogates):
            surrogate = self.generate_surrogate(time_series)
            surrogate_stats.append(self.time_reversal_asymmetry(surrogate))
            
        # Compute p-value
        p_value = np.mean(np.abs(surrogate_stats) >= np.abs(tra_stat))
        
        # Test result
        is_nonlinear = p_value < significance_level
        
        details = {
            'embedding_dimension': embedding_dim,
            'time_reversal_asymmetry': tra_stat,
            'tra_pvalue': p_value,
            'n_surrogates': self.n_surrogates,
            'surrogate_stats': np.array(surrogate_stats)
        }
        
        return is_nonlinear, p_value, details
    
    def _estimate_embedding_dimension(
        self,
        time_series: np.ndarray,
        max_dim: int = 10,
        tolerance: float = 2.0,
        threshold: float = 0.1
    ) -> int:
        """
        Estimate embedding dimension using false nearest neighbors algorithm.
        
        Args:
            time_series: Input time series
            max_dim: Maximum dimension to test
            tolerance: Tolerance for considering a neighbor false
            threshold: Threshold for percentage of false neighbors
            
        Returns:
            Estimated embedding dimension
        """
        time_series = np.asarray(time_series)
        min_dim = 1
        n_points = len(time_series)
        fnn_percentages = []
        
        for dim in range(min_dim, max_dim + 1):
            # Create delay embedding
            embedding = reconstruct(time_series, dim, tau=1)
            
            # Make sure we can create next embedding
            if dim + 1 <= max_dim:
                next_embedding = reconstruct(time_series, dim + 1, tau=1)
            else:
                break
                
            # Find nearest neighbors
            nbrs = NearestNeighbors(
                n_neighbors=2,  # Include the point itself
                algorithm='auto'
            ).fit(embedding)
            distances, indices = nbrs.kneighbors(embedding)
            
            # Count false neighbors
            n_false = 0
            total = 0
            
            for i in range(len(embedding)):
                # Get nearest neighbor (excluding point itself)
                nn_index = indices[i, 1]
                current_dist = distances[i, 1]
                
                if current_dist == 0:
                    continue
                    
                # Check if we can compute next dimension distance
                if i + dim >= len(time_series) or nn_index + dim >= len(time_series):
                    continue
                    
                # Distance in next dimension
                next_dist = np.abs(
                    time_series[i + dim] - time_series[nn_index + dim]
                )
                
                # Check if false neighbor
                if (next_dist / current_dist) > tolerance:
                    n_false += 1
                total += 1
            
            # Calculate percentage of false neighbors
            fnn_percent = n_false / total if total > 0 else 1.0
            fnn_percentages.append(fnn_percent)
            
            # Check if suitable dimension found
            if fnn_percent < threshold:
                return dim
                
            # Check convergence
            if (dim > 1 and 
                abs(fnn_percentages[-1] - fnn_percentages[-2]) < threshold/2):
                return dim
        
        # If no clear dimension found, return minimum fnn
        return min_dim + np.argmin(fnn_percentages)