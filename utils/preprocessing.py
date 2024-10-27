# xCCM/utils/preprocessing.py
"""Data preprocessing utilities for CCM analysis."""

import numpy as np
from typing import Optional, Tuple
from scipy import stats
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    """Utilities for preprocessing time series data."""
    
    @staticmethod
    def standardize(time_series: np.ndarray) -> np.ndarray:
        """
        Standardize time series to zero mean and unit variance.

        Args:
            time_series: Input time series

        Returns:
            Standardized time series
        """
        return StandardScaler().fit_transform(
            time_series.reshape(-1, 1)
        ).ravel()
    
    @staticmethod
    def detrend(
        time_series: np.ndarray,
        order: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove polynomial trend from time series.

        Args:
            time_series: Input time series
            order: Order of polynomial to fit

        Returns:
            Tuple of (detrended_series, trend)
        """
        x = np.arange(len(time_series))
        coeffs = np.polyfit(x, time_series, order)
        trend = np.polyval(coeffs, x)
        return time_series - trend, trend
    
    @staticmethod
    def filter_outliers(
        time_series: np.ndarray,
        threshold: float = 3,
        method: str = 'zscore'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove outliers from time series.

        Args:
            time_series: Input time series
            threshold: Threshold for outlier detection
            method: Method to use ('zscore' or 'iqr')

        Returns:
            Tuple of (filtered_series, outlier_mask)

        Raises:
            ValueError: If unknown method is specified
        """
        if method == 'zscore':
            z_scores = np.abs(stats.zscore(time_series))
            mask = z_scores < threshold
        elif method == 'iqr':
            q1, q3 = np.percentile(time_series, [25, 75])
            iqr = q3 - q1
            mask = ~((time_series < (q1 - threshold * iqr)) | 
                    (time_series > (q3 + threshold * iqr)))
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
            
        return time_series[mask], mask
    
    @staticmethod
    def interpolate_missing(
        time_series: np.ndarray,
        mask: Optional[np.ndarray] = None,
        method: str = 'linear'
    ) -> np.ndarray:
        """
        Interpolate missing values in time series.

        Args:
            time_series: Input time series
            mask: Boolean mask of missing values
            method: Interpolation method

        Returns:
            Interpolated time series
        """
        if mask is None:
            mask = np.isnan(time_series)
            
        x = np.arange(len(time_series))
        
        if method == 'linear':
            f = interp1d(
                x[~mask],
                time_series[~mask],
                bounds_error=False,
                fill_value='extrapolate'
            )
        else:
            f = interp1d(
                x[~mask],
                time_series[~mask],
                kind=method,
                bounds_error=False
            )
            
        return f(x)
