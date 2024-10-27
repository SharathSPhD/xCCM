# xCCM/utils/performance.py
"""Performance optimization tools for CCM analysis."""

import numpy as np
import multiprocessing as mp
from typing import List, Optional
from ..modules.core import EnhancedCCM, CCMResult

class CCMPerformance:
    """Performance optimization tools for CCM analysis."""
    
    def __init__(self, n_jobs: int = -1) -> None:
        """
        Initialize with number of parallel jobs.

        Args:
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()
        
    def optimize_computation(
        self,
        ccm: EnhancedCCM,
        x: np.ndarray,
        y: np.ndarray,
        lib_sizes: np.ndarray
    ) -> List[CCMResult]:
        """
        Optimize CCM computation using parallel processing.

        Args:
            ccm: EnhancedCCM instance
            x: First time series
            y: Second time series
            lib_sizes: Library sizes to test

        Returns:
            List of CCMResult objects
        """
        chunks = np.array_split(lib_sizes, self.n_jobs)
        
        with mp.Pool(self.n_jobs) as pool:
            results = pool.starmap(
                ccm.run_ccm,
                [(x, y, chunk) for chunk in chunks]
            )
            
        return [result for sublist in results for result in sublist]
    
    def memory_efficient_ccm(
        self,
        ccm: EnhancedCCM,
        x: np.ndarray,
        y: np.ndarray,
        max_lib_size: int,
        chunk_size: int = 1000
    ) -> List[CCMResult]:
        """
        Memory-efficient CCM computation for large datasets.

        Args:
            ccm: EnhancedCCM instance
            x: First time series
            y: Second time series
            max_lib_size: Maximum library size
            chunk_size: Size of chunks for processing

        Returns:
            List of CCMResult objects
        """
        results = []
        
        for start in range(0, max_lib_size, chunk_size):
            end = min(start + chunk_size, max_lib_size)
            lib_sizes = np.arange(start + 1, end + 1)
            
            chunk_results = ccm.run_ccm(x, y, lib_sizes)
            results.extend(chunk_results)
            
        return results