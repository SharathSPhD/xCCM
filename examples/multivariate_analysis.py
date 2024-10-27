# examples/multivariate_analysis.py
"""Multivariate CCM analysis examples."""

import numpy as np
import matplotlib.pyplot as plt
from xCCM.modules.multivariate import MultivariateCCM
from xCCM.utils.visualization import CCMVisualizer
from xCCM.modules.network import CCMNetwork

def generate_multivariate_system(
    n_points: int = 1000,
    noise_level: float = 0.1
) -> dict:
    """
    Generate multivariate coupled system.
    
    Args:
        n_points: Number of time points
        noise_level: Noise level
    
    Returns:
        Dictionary of time series
    """
    t = np.linspace(0, 20*np.pi, n_points)
    
    # Generate coupled oscillators
    x = np.sin(t)
    y = np.sin(t + np.pi/4) + 0.3 * x
    z = np.sin(t + np.pi/2) + 0.2 * y
    
    # Add noise
    x += np.random.normal(0, noise_level, n_points)
    y += np.random.normal(0, noise_level, n_points)
    z += np.random.normal(0, noise_level, n_points)
    
    return {'x': x, 'y': y, 'z': z}

def multivariate_analysis():
    """Demonstrate multivariate CCM analysis."""
    # Generate data
    print("Generating multivariate data...")
    data = generate_multivariate_system()
    
    # Initialize multivariate CCM
    print("Initializing multivariate CCM...")
    mccm = MultivariateCCM(embedding_dimension=3)
    
    # Analyze all variable pairs
    print("Running multivariate analysis...")
    lib_sizes = np.arange(50, 500, 50)
    variables = list(data.keys())
    ccm_results = {}
    
    for i, var1 in enumerate(variables):
        for var2 in variables[i+1:]:
            # Test both directions
            result1 = mccm.run_multivariate_ccm(
                data,
                target_var=var2,
                lib_sizes=lib_sizes
            )
            result2 = mccm.run_multivariate_ccm(
                data,
                target_var=var1,
                lib_sizes=lib_sizes
            )
            
            # Store final correlations and p-values
            ccm_results[(var1, var2)] = (
                result1['results'][-1].correlation,
                result1['results'][-1].significance
            )
            ccm_results[(var2, var1)] = (
                result2['results'][-1].correlation,
                result2['results'][-1].significance
            )
    
    # Create and visualize network
    print("Creating causal network...")
    network = CCMNetwork()
    network.build_network(ccm_results)
    
    viz = CCMVisualizer()
    viz.plot_causality_network(ccm_results)
    
    # Analyze network properties
    print("\nNetwork Analysis:")
    drivers = network.identify_drivers()
    loops = network.find_feedback_loops()
    stats = network.compute_network_statistics()
    
    print("\nDriver variables:")
    for var, score in drivers.items():
        print(f"{var}: {score:.3f}")
        
    print("\nFeedback loops:")
    for loop in loops:
        print(" → ".join(loop))
        
    print("\nNetwork statistics:")
    for stat, value in stats.items():
        print(f"{stat}: {value:.3f}")
    
    plt.show()

if __name__ == "__main__":
    multivariate_analysis()