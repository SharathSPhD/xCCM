# xCCM/modules/multivariate.py
"""Multivariate CCM implementation."""

import numpy as np
from typing import Dict, List, Optional
from .core import EnhancedCCM

class MultivariateCCM:
    """Multivariate CCM for analyzing multiple coupled time series."""
    
    def __init__(
        self, 
        embedding_dimension: int,
        tau: int = 1,
        variables: Optional[List[str]] = None
    ) -> None:
        """
        Initialize MultivariateCCM.

        Args:
            embedding_dimension: Dimension of the embedding
            tau: Time delay
            variables: List of variable names
        """
        self.embedding = embedding_dimension
        self.tau = tau
        self.variables = variables
        self.ccm = EnhancedCCM(embedding_dimension, tau)
        
    def create_multivariate_embedding(
        self,
        time_series_dict: Dict[str, np.ndarray],
        target_var: str
    ) -> np.ndarray:
        """
        Create multivariate embedding using multiple variables.

        Args:
            time_series_dict: Dictionary of time series
            target_var: Target variable name

        Returns:
            Multivariate embedding array
        """
        if self.variables is None:
            self.variables = list(time_series_dict.keys())
            
        embeddings = []
        for var in self.variables:
            if var != target_var:
                ts = time_series_dict[var]
                embedding = self.ccm.create_embedding(ts)
                embeddings.append(embedding)
                
        return np.hstack(embeddings)
    
    def run_multivariate_ccm(
        self,
        time_series_dict: Dict[str, np.ndarray],
        target_var: str,
        lib_sizes: List[int]
    ) -> Dict:
        """
        Run CCM analysis using multiple variables.

        Args:
            time_series_dict: Dictionary of time series
            target_var: Target variable name
            lib_sizes: Library sizes to test

        Returns:
            Dictionary containing results and metadata
        """
        mv_embedding = self.create_multivariate_embedding(
            time_series_dict, target_var)
        
        target_ts = time_series_dict[target_var]
        results = self.ccm.run_ccm(mv_embedding, target_ts, lib_sizes)
        
        return {
            'results': results,
            'variables_used': self.variables,
            'target': target_var
        }