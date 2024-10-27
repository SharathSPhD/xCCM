# xCCM/modules/core.py

import numpy as np
from scipy.signal import savgol_filter
from sklearn.neighbors import NearestNeighbors
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import warnings

@dataclass
class CCMResult:
    """Data class for storing CCM analysis results."""
    correlation: float
    predictions: np.ndarray
    lib_size: int
    significance: Optional[float] = None
    surrogate_correlations: Optional[np.ndarray] = None

class EnhancedCCM:
    """Enhanced CCM with improved stability and validation."""
    
    def __init__(
        self, 
        embedding_dimension: int,
        tau: int = 1,
        noise_threshold: float = 0.1,
        n_neighbors: Optional[int] = None,
        random_state: Optional[int] = None
    ) -> None:
        """Initialize EnhancedCCM."""
        if embedding_dimension < 2:
            raise ValueError("Embedding dimension must be at least 2")
        if tau < 1:
            raise ValueError("Time delay must be positive")
        if noise_threshold <= 0:
            raise ValueError("Noise threshold must be positive")
            
        self.embedding = embedding_dimension
        self.tau = tau
        self.noise_threshold = noise_threshold
        self.n_neighbors = n_neighbors or embedding_dimension + 1
        
        if random_state is not None:
            np.random.seed(random_state)
            
    def decompose_signal(
        self, 
        time_series: np.ndarray,
        window_length: Optional[int] = None,
        polyorder: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Decompose time series with improved validation."""
        time_series = np.asarray(time_series)
        if len(time_series) < 4:
            raise ValueError("Time series too short (min length 4)")
            
        if window_length is None:
            window_length = 2 * self.embedding + 1
            
        if window_length % 2 == 0:
            window_length += 1
            
        if window_length < polyorder + 2:
            raise ValueError("Window length must be > polyorder + 1")
            
        if window_length >= len(time_series):
            raise ValueError("Window length must be < time series length")
            
        if np.any(~np.isfinite(time_series)):
            warnings.warn("Non-finite values detected and removed")
            time_series = np.copy(time_series)
            mask = np.isfinite(time_series)
            time_series[~mask] = np.interp(
                np.nonzero(~mask)[0],
                np.nonzero(mask)[0],
                time_series[mask]
            )
            
        signal = savgol_filter(time_series, window_length, polyorder)
        noise = time_series - signal
        
        signal_power = np.var(signal)
        noise_power = np.var(noise)
        snr = signal_power / (noise_power + 1e-10)
        
        if snr < self.noise_threshold:
            warnings.warn(f"Low SNR detected: {snr:.2f} < {self.noise_threshold}")
            
        return signal, noise
        
    def create_embedding(
        self,
        time_series: np.ndarray,
        predict_lag: int = 0
    ) -> np.ndarray:
        """Create time-delay embedding with improved validation."""
        time_series = np.asarray(time_series)
        ts_length = len(time_series)
        
        if ts_length < self.embedding * self.tau:
            raise ValueError("Time series too short for embedding parameters")
            
        # Calculate number of vectors
        n_vectors = ts_length - (self.embedding-1)*self.tau
        
        # Pre-allocate embedding array
        embedding = np.zeros((n_vectors, self.embedding))
        
        # Fill embedding array
        for i in range(self.embedding):
            embedding[:, i] = time_series[i*self.tau:i*self.tau + n_vectors]
            
        return embedding
        
    def compute_manifold_weights(
        self,
        target_point: np.ndarray,
        neighbors: np.ndarray,
        distances: np.ndarray,
        theta: float = 1.0
    ) -> np.ndarray:
        """Compute weights with improved numerical stability."""
        if len(distances) == 0:
            raise ValueError("No neighbors provided")
            
        if theta <= 0:
            raise ValueError("theta must be positive")
            
        mean_dist = np.mean(distances) + 1e-10
        weights = np.exp(-theta * distances / mean_dist)
        weights = weights / (np.sum(weights) + 1e-10)
        
        if np.any(~np.isfinite(weights)):
            raise ValueError("Invalid weights computed")
            
        return weights
        
    def predict_weighted(
        self,
        manifold: np.ndarray,
        target_idx: int,
        lib_indices: np.ndarray,
        lag: int = 0
    ) -> Tuple[float, np.ndarray]:
        """Make weighted prediction with improved validation."""
        manifold = np.asarray(manifold)
        if len(lib_indices) < self.n_neighbors:
            raise ValueError(
                f"Library size ({len(lib_indices)}) must be >= {self.n_neighbors}"
            )
            
        # Ensure manifold is 2D
        if manifold.ndim == 1:
            manifold = manifold.reshape(-1, 1)
            
        target_point = manifold[target_idx]
        if target_point.ndim == 1:
            target_point = target_point.reshape(1, -1)
            
        nbrs = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            metric='euclidean'
        )
        nbrs.fit(manifold[lib_indices])
        
        distances, indices = nbrs.kneighbors(target_point)
        neighbor_points = manifold[lib_indices][indices[0]]
        
        weights = self.compute_manifold_weights(
            target_point.ravel(),
            neighbor_points,
            distances[0]
        )
        
        prediction = np.sum(weights * neighbor_points[:, 0])
        return prediction, weights
        
    def run_ccm(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lib_sizes: Union[np.ndarray, List[int]],
        num_samples: int = 100,
        random_state: Optional[int] = None,
        lag: int = 0
    ) -> List[CCMResult]:
        """Run CCM analysis with improved validation and stability."""
        if random_state is not None:
            np.random.seed(random_state)
            
        # Convert inputs to numpy arrays
        x = np.asarray(x)
        y = np.asarray(y)
        
        if len(x) != len(y):
            raise ValueError("Time series must have same length")
            
        if len(x) < self.embedding * self.tau:
            raise ValueError("Time series too short for embedding parameters")
            
        # Signal decomposition
        x_signal, x_noise = self.decompose_signal(x)
        y_signal, y_noise = self.decompose_signal(y)
        
        # Create embeddings
        x_manifold = self.create_embedding(x_signal)
        y_manifold = self.create_embedding(y_signal)
        
        results = []
        
        for lib_size in lib_sizes:
            predictions = np.zeros(num_samples)
            max_lib_size = min(lib_size, len(x_manifold)-1)
            
            for i in range(num_samples):
                # Ensure valid library indices
                valid_indices = np.arange(len(x_manifold))
                if lag > 0:
                    valid_indices = valid_indices[:-lag]
                elif lag < 0:
                    valid_indices = valid_indices[-lag:]
                
                if len(valid_indices) < max_lib_size:
                    max_lib_size = len(valid_indices)
                    
                lib_indices = np.random.choice(
                    valid_indices,
                    size=max_lib_size,
                    replace=False
                )
                
                # Ensure valid target index
                target_idx = np.random.choice(valid_indices)
                
                # Make prediction
                pred, _ = self.predict_weighted(
                    x_manifold,
                    target_idx,
                    lib_indices,
                    lag=lag
                )
                predictions[i] = pred
                
            # Handle lagged correlation calculation
            if lag >= 0:
                y_subset = y[lib_indices[lag:]]
                pred_subset = predictions[:-lag] if lag > 0 else predictions
            else:
                y_subset = y[lib_indices[:lag]]
                pred_subset = predictions[-lag:]
                
            # Ensure valid subsetting
            min_len = min(len(y_subset), len(pred_subset))
            y_subset = y_subset[:min_len]
            pred_subset = pred_subset[:min_len]
            
            # Compute correlation
            correlation = np.corrcoef(pred_subset, y_subset)[0, 1]
            
            results.append(CCMResult(
                correlation=correlation,
                predictions=predictions,
                lib_size=lib_size
            ))
                
        return results