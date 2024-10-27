# docs/user_guide.md
```markdown
# Extended CCM User Guide

## Table of Contents
1. [Installation](#installation)
2. [Basic Usage](#basic-usage)
3. [Advanced Features](#advanced-features)
4. [Performance Optimization](#performance-optimization)
5. [API Reference](#api-reference)
6. [Examples](#examples)

## Installation

```bash
pip install extended-ccm
```

For development installation:
```bash
git clone https://github.com/yourusername/extended-ccm.git
cd extended-ccm
pip install -e ".[dev]"
```

## Basic Usage

### Simple CCM Analysis
```python
from extended_ccm import EnhancedCCM
import numpy as np

# Generate or load your time series
x = your_first_timeseries
y = your_second_timeseries

# Initialize CCM
ccm = EnhancedCCM(embedding_dimension=3)

# Run analysis
lib_sizes = np.arange(10, 500, 50)
results = ccm.run_ccm(x, y, lib_sizes)

# Access results
for r in results:
    print(f"Library size: {r.lib_size}, Correlation: {r.correlation:.3f}")
```

### Visualization
```python
from extended_ccm import CCMVisualizer

viz = CCMVisualizer()
viz.plot_convergence(lib_sizes, [r.correlation for r in results])
```

## Advanced Features

### Multivariate Analysis
```python
from extended_ccm import MultivariateCCM

# Initialize multivariate CCM
mccm = MultivariateCCM(embedding_dimension=3)

# Prepare data
data = {
    'x': x_timeseries,
    'y': y_timeseries,
    'z': z_timeseries
}

# Run analysis
results = mccm.run_multivariate_ccm(data, target_var='x', lib_sizes=lib_sizes)
```

### Network Analysis
```python
from extended_ccm import CCMNetwork

# Create network
network = CCMNetwork()
network.build_network(ccm_results)

# Analyze causality
drivers = network.identify_drivers()
loops = network.find_feedback_loops()
```

### Nonlinearity Testing
```python
from extended_ccm import NonlinearityTests

nlt = NonlinearityTests()
is_nonlinear, pvalue, details = nlt.test_nonlinearity(time_series)
```

## Performance Optimization

### Parallel Processing
```python
from extended_ccm import CCMPerformance

perf = CCMPerformance(n_jobs=4)  # Use 4 cores
results = perf.optimize_computation(ccm, x, y, lib_sizes)
```

### Memory Efficiency
```python
results = perf.memory_efficient_ccm(
    ccm,
    x,
    y,
    max_lib_size=5000,
    chunk_size=1000
)
```

## API Reference

### EnhancedCCM
- `embedding_dimension`: Dimension for state space reconstruction
- `tau`: Time delay
- `noise_threshold`: Threshold for SNR warnings
- `n_neighbors`: Number of neighbors for prediction

Methods:
- `decompose_signal(time_series, window_length, polyorder)`
- `predict_weighted(manifold, target_idx, lib_indices)`
- `run_ccm(x, y, lib_sizes, num_samples, random_state)`

### MultivariateCCM
Methods:
- `create_multivariate_embedding(time_series_dict, target_var)`
- `run_multivariate_ccm(time_series_dict, target_var, lib_sizes)`

### CCMNetwork
Methods:
- `build_network(ccm_results, significance_threshold)`
- `identify_drivers()`
- `find_feedback_loops()`
- `compute_network_statistics()`

### NonlinearityTests
Methods:
- `generate_surrogate(time_series, method)`
- `surrogate_test(time_series, statistic, surrogate_method)`
- `test_nonlinearity(time_series, embedding_dim, significance_level)`
```

# examples/real_world_examples/climate_analysis.py
```python
"""Example of CCM analysis on climate data."""

import numpy as np
import pandas as pd
from extended_ccm import EnhancedCCM, CCMVisualizer, DataPreprocessor

def analyze_climate_data(temp_data, co2_data):
    """
    Analyze causality between temperature and CO2 levels.
    
    Args:
        temp_data: Temperature time series
        co2_data: CO2 levels time series
    """
    # Preprocess data
    preprocessor = DataPreprocessor()
    temp_clean = preprocessor.standardize(temp_data)
    co2_clean = preprocessor.standardize(co2_data)
    
    # Initialize CCM
    ccm = EnhancedCCM(embedding_dimension=3)
    
    # Run bidirectional analysis
    lib_sizes = np.arange(100, len(temp_clean)//2, 50)
    
    temp_causes_co2 = ccm.run_ccm(temp_clean, co2_clean, lib_sizes)
    co2_causes_temp = ccm.run_ccm(co2_clean, temp_clean, lib_sizes)
    
    # Visualize results
    viz = CCMVisualizer()
    
    viz.plot_convergence(
        lib_sizes,
        [r.correlation for r in temp_causes_co2],
        title='Temperature → CO2'
    )
    
    viz.plot_convergence(
        lib_sizes,
        [r.correlation for r in co2_causes_temp],
        title='CO2 → Temperature'
    )
    
    return temp_causes_co2, co2_causes_temp

# Example usage
if __name__ == "__main__":
    # Load example data
    data = pd.read_csv('climate_data.csv')  # You would need this file
    temp_data = data['temperature'].values
    co2_data = data['co2'].values
    
    results = analyze_climate_data(temp_data, co2_data)
```

# examples/real_world_examples/ecological_analysis.py
```python
"""Example of CCM analysis on ecological data."""

import numpy as np
import pandas as pd
from extended_ccm import MultivariateCCM, CCMNetwork, CCMVisualizer

def analyze_species_interactions(species_data: dict):
    """
    Analyze causal relationships in species abundance data.
    
    Args:
        species_data: Dictionary of species abundance time series
    """
    # Initialize multivariate CCM
    mccm = MultivariateCCM(embedding_dimension=3)
    
    # Analyze all species pairs
    lib_sizes = np.arange(50, 500, 50)
    species = list(species_data.keys())
    ccm_results = {}
    
    for i, sp1 in enumerate(species):
        for sp2 in species[i+1:]:
            # Test both directions
            result1 = mccm.run_multivariate_ccm(
                species_data,
                target_var=sp2,
                lib_sizes=lib_sizes
            )
            result2 = mccm.run_multivariate_ccm(
                species_data,
                target_var=sp1,
                lib_sizes=lib_sizes
            )
            
            ccm_results[(sp1, sp2)] = (
                result1['results'][-1].correlation,
                result1['results'][-1].significance
            )
            ccm_results[(sp2, sp1)] = (
                result2['results'][-1].correlation,
                result2['results'][-1].significance
            )
    
    # Create interaction network
    network = CCMNetwork()
    network.build_network(ccm_results)
    
    # Analyze network properties
    drivers = network.identify_drivers()
    loops = network.find_feedback_loops()
    
    # Visualize results
    viz = CCMVisualizer()
    viz.plot_causality_network(ccm_results)
    
    return {
        'ccm_results': ccm_results,
        'drivers': drivers,
        'feedback_loops': loops
    }

if __name__ == "__main__":
    # Example data structure
    species_data = {
        'species_a': np.random.randn(1000),
        'species_b': np.random.randn(1000),
        'species_c': np.random.randn(1000)
    }
    
    results = analyze_species_interactions(species_data)
```

# examples/real_world_examples/financial_analysis.py
```python
"""Example of CCM analysis on financial data."""

import numpy as np
import pandas as pd
from extended_ccm import EnhancedCCM, NonlinearityTests, CCMVisualizer
from extended_ccm.utils.preprocessing import DataPreprocessor

def analyze_market_causality(
    price_data: dict,
    window_size: int = 500,
    step_size: int = 50
):
    """
    Analyze causality between different financial instruments.
    
    Args:
        price_data: Dictionary of price time series
        window_size: Size of rolling window
        step_size: Step size for rolling analysis
    """
    # Preprocess data
    preprocessor = DataPreprocessor()
    processed_data = {
        k: preprocessor.standardize(v) 
        for k, v in price_data.items()
    }
    
    # Initialize CCM
    ccm = EnhancedCCM(embedding_dimension=3)
    nlt = NonlinearityTests()
    
    # Check for nonlinearity
    nonlinearity_results = {}
    for name, series in processed_data.items():
        is_nonlinear, pvalue, details = nlt.test_nonlinearity(series)
        nonlinearity_results[name] = {
            'is_nonlinear': is_nonlinear,
            'pvalue': pvalue,
            'details': details
        }
    
    # Rolling CCM analysis
    rolling_results = []
    instruments = list(processed_data.keys())
    
    for i in range(0, len(next(iter(processed_data.values()))) - window_size, step_size):
        window_data = {
            k: v[i:i+window_size]
            for k, v in processed_data.items()
        }
        
        window_results = {}
        for inst1 in instruments:
            for inst2 in instruments:
                if inst1 != inst2:
                    lib_sizes = np.arange(50, window_size//2, 50)
                    results = ccm.run_ccm(
                        window_data[inst1],
                        window_data[inst2],
                        lib_sizes
                    )
                    window_results[(inst1, inst2)] = results[-1].correlation
                    
        rolling_results.append({
            'start_idx': i,
            'results': window_results
        })
    
    # Visualize results
    viz = CCMVisualizer()
    
    # Plot rolling causality strengths
    times = np.arange(len(rolling_results))
    for pair in rolling_results[0]['results'].keys():
        values = [r['results'][pair] for r in rolling_results]
        viz.plot_convergence(
            times,
            values,
            title=f'Rolling Causality: {pair[0]} → {pair[1]}'
        )
    
    return {
        'nonlinearity': nonlinearity_results,
        'rolling_causality': rolling_results
    }

if __name__ == "__main__":
    # Example usage with financial data
    price_data = {
        'stock': np.random.randn(2000).cumsum(),
        'bond': np.random.randn(2000).cumsum(),
        'commodity': np.random.randn(2000).cumsum()
    }
    
    results = analyze_market_causality(price_data)
```