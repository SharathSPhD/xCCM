# examples/basic_usage.py
"""Basic usage examples for Extended CCM package."""

import numpy as np
import matplotlib.pyplot as plt
from xCCM.modules.core import EnhancedCCM
from xCCM.utils.visualization import CCMVisualizer
from xCCM.utils.preprocessing import DataPreprocessor

def generate_coupled_logistic(
    n_points: int = 1000,
    r: float = 3.8,
    coupling: float = 0.4,
    noise_level: float = 0.05
) -> tuple:
    """
    Generate coupled logistic maps with optional noise.
    
    Args:
        n_points: Number of time points
        r: Growth parameter
        coupling: Coupling strength
        noise_level: Standard deviation of noise
    
    Returns:
        Tuple of (driver, response) time series
    """
    x = np.zeros(n_points)
    y = np.zeros(n_points)
    
    # Initialize
    x[0] = 0.4
    y[0] = 0.2
    
    # Generate coupled time series
    for t in range(1, n_points):
        x[t] = r * x[t-1] * (1 - x[t-1])
        y[t] = r * y[t-1] * (1 - y[t-1]) + coupling * x[t-1]
    
    # Add noise
    if noise_level > 0:
        x += np.random.normal(0, noise_level, n_points)
        y += np.random.normal(0, noise_level, n_points)
    
    return x, y

def basic_ccm_analysis():
    """Demonstrate basic CCM analysis workflow."""
    # Generate sample data
    print("Generating sample data...")
    x, y = generate_coupled_logistic()
    
    # Preprocess data
    print("Preprocessing data...")
    preprocessor = DataPreprocessor()
    x_clean = preprocessor.standardize(x)
    y_clean = preprocessor.standardize(y)
    
    # Initialize CCM
    print("Initializing CCM...")
    ccm = EnhancedCCM(embedding_dimension=3, tau=1)
    
    # Run CCM analysis
    print("Running CCM analysis...")
    lib_sizes = np.arange(10, 500, 50)
    
    # Test causality in both directions
    x_causes_y = ccm.run_ccm(x_clean, y_clean, lib_sizes)
    y_causes_x = ccm.run_ccm(y_clean, x_clean, lib_sizes)
    
    # Visualize results
    print("Visualizing results...")
    viz = CCMVisualizer()
    
    # Plot convergence for both directions
    fig1 = viz.plot_convergence(
        lib_sizes,
        [r.correlation for r in x_causes_y],
        title='X causes Y'
    )
    
    fig2 = viz.plot_convergence(
        lib_sizes,
        [r.correlation for r in y_causes_x],
        title='Y causes X'
    )
    
    plt.show()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Final X→Y correlation: {x_causes_y[-1].correlation:.3f}")
    print(f"Final Y→X correlation: {y_causes_x[-1].correlation:.3f}")

if __name__ == "__main__":
    basic_ccm_analysis()