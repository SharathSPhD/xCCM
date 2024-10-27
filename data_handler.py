# xCCM/data_handler.py

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import yaml
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DataStructureDetector:
    """Detect and validate CSV file structure."""
    
    @staticmethod
    def detect_header(file_path: str, n_rows: int = 5) -> bool:
        """
        Detect if CSV file has a header.
        
        Args:
            file_path: Path to CSV file
            n_rows: Number of rows to check
            
        Returns:
            Boolean indicating if file has header
        """
        df = pd.read_csv(file_path, nrows=n_rows)
        potential_header = pd.read_csv(file_path, nrows=0).columns
        
        # Check if potential header values are all strings
        if all(isinstance(x, str) for x in potential_header):
            # Check if header values appear in data
            data_values = df.values.flatten()
            header_in_data = any(
                str(h) in map(str, data_values) 
                for h in potential_header
            )
            return not header_in_data
        
        return False
    
    @staticmethod
    def detect_datetime_column(
        df: pd.DataFrame
    ) -> Optional[str]:
        """
        Detect datetime column in DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Name of datetime column if found, else None
        """
        for col in df.columns:
            try:
                pd.to_datetime(df[col])
                return col
            except (ValueError, TypeError):
                continue
        return None
    
    @staticmethod
    def detect_numeric_columns(
        df: pd.DataFrame
    ) -> List[str]:
        """
        Detect numeric columns in DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of numeric column names
        """
        return df.select_dtypes(
            include=[np.number]
        ).columns.tolist()

class DataPreprocessor:
    """Preprocess time series data."""
    
    @staticmethod
    def standardize(series: pd.Series) -> pd.Series:
        """Standardize series to zero mean and unit variance."""
        return (series - series.mean()) / series.std()
    
    @staticmethod
    def detrend(series: pd.Series) -> pd.Series:
        """Remove linear trend from series."""
        x = np.arange(len(series))
        z = np.polyfit(x, series, 1)
        p = np.poly1d(z)
        trend = p(x)
        return series - trend
    
    @staticmethod
    def interpolate_missing(series: pd.Series) -> pd.Series:
        """Interpolate missing values."""
        return series.interpolate(method='linear')
    
    @staticmethod
    def remove_outliers(
        series: pd.Series,
        threshold: float = 3
    ) -> pd.Series:
        """Remove outliers using z-score method."""
        z_scores = np.abs((series - series.mean()) / series.std())
        return series[z_scores < threshold]

class DataHandler:
    """Handle data loading and preprocessing."""
    
    def __init__(self, config_path: str):
        """
        Initialize DataHandler.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.detector = DataStructureDetector()
        self.preprocessor = DataPreprocessor()
        
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Load and prepare data according to configuration.
        
        Returns:
            Tuple of (processed_data, metadata)
        """
        # Load configuration
        data_config = self.config['data']
        file_path = data_config['file_path']
        
        # Detect file structure if not specified
        has_header = data_config.get(
            'has_header',
            self.detector.detect_header(file_path)
        )
        
        # Read data
        df = pd.read_csv(
            file_path,
            header=0 if has_header else None
        )
        
        # Detect datetime column
        datetime_col = data_config.get(
            'datetime_column',
            self.detector.detect_datetime_column(df)
        )
        if datetime_col:
            df[datetime_col] = pd.to_datetime(df[datetime_col])
            df.set_index(datetime_col, inplace=True)
        
        # Get variables for analysis
        if 'variables' in data_config:
            variables = data_config['variables']
        else:
            numeric_cols = self.detector.detect_numeric_columns(df)
            variables = [
                {'name': col, 'type': 'float', 'preprocess': ['standardize']}
                for col in numeric_cols
            ]
        
        # Preprocess each variable
        processed_data = pd.DataFrame(index=df.index)
        for var in variables:
            series = df[var['name']]
            
            # Apply preprocessing steps
            for step in var.get('preprocess', []):
                if hasattr(self.preprocessor, step):
                    series = getattr(self.preprocessor, step)(series)
                else:
                    logger.warning(f"Unknown preprocessing step: {step}")
            
            processed_data[var['name']] = series
        
        # Create metadata
        metadata = {
            'n_variables': len(variables),
            'variable_names': [v['name'] for v in variables],
            'datetime_index': datetime_col is not None,
            'n_observations': len(processed_data)
        }
        
        return processed_data, metadata