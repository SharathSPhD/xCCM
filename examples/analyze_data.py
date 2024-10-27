# examples/analyze_data.py

import yaml
from pathlib import Path
from xCCM.data_handler import DataHandler
from xCCM.modules.core import EnhancedCCM
from xCCM.modules.multivariate import MultivariateCCM
from xCCM.utils.visualization import CCMVisualizer
import numpy as np
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_analysis(config_path: str):
    """
    Run CCM analysis based on configuration.
    
    Args:
        config_path: Path to configuration file
    """
    # Load and prepare data
    handler = DataHandler(config_path)
    data, metadata = handler.load_and_prepare_data()
    
    logger.info(f"Loaded data with {metadata['n_variables']} variables")
    
    # Load analysis configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    analysis_config = config['analysis']
    output_config = config['output']
    
    # Create output directory
    output_dir = Path(output_config['results_dir'])
    output_dir.mkdir(exist_ok=True)
    
    # Initialize visualization
    viz = CCMVisualizer()
    
    # Determine analysis type based on number of variables
    if metadata['n_variables'] == 2:
        # Bivariate analysis
        logger.info("Running bivariate CCM analysis")
        results = run_bivariate_analysis(
            data,
            analysis_config,
            output_dir,
            viz
        )
    else:
        # Multivariate analysis
        logger.info("Running multivariate CCM analysis")
        results = run_multivariate_analysis(
            data,
            analysis_config,
            output_dir,
            viz
        )
    
    # Save results
    save_results(results, output_dir, output_config)

def run_bivariate_analysis(data, config, output_dir, viz):
    """Run bivariate CCM analysis."""
    ccm = EnhancedCCM(
        embedding_dimension=config['embedding_dimension'],
        tau=config['tau']
    )
    
    variables = data.columns
    lib_size_range = np.arange(*config['library_sizes'])
    
    # Analyze in both directions
    results = {}
    for i, var1 in enumerate(variables):
        for var2 in variables[i+1:]:
            # X causes Y
            xy_results = ccm.run_ccm(
                data[var1].values,
                data[var2].values,
                lib_size_range
            )
            
            # Y causes X
            yx_results = ccm.run_ccm(
                data[var2].values,
                data[var1].values,
                lib_size_range
            )
            
            results[f"{var1}->{var2}"] = xy_results
            results[f"{var2}->{var1}"] = yx_results
            
            # Plot results
            if config.get('save_figures', True):
                viz.plot_convergence(
                    lib_size_range,
                    [r.correlation for r in xy_results],
                    title=f'Causality: {var1} → {var2}'
                )
                plt.savefig(
                    output_dir / f"convergence_{var1}_to_{var2}.png"
                )
                plt.close()
    
    return results

def run_multivariate_analysis(data, config, output_dir, viz):
    """Run multivariate CCM analysis."""
    mccm = MultivariateCCM(
        embedding_dimension=config['embedding_dimension'],
        tau=config['tau']
    )
    
    lib_size_range = np.arange(*config['library_sizes'])
    
    # Analyze all variables
    results = {}
    for target in data.columns:
        result = mccm.run_multivariate_ccm(
            {col: data[col].values for col in data.columns},
            target,
            lib_size_range
        )
        results[target] = result
        
        if config.get('save_figures', True):
            viz.plot_convergence(
                lib_size_range,
                [r.correlation for r in result['results']],
                title=f'Multivariate CCM: Target = {target}'
            )
            plt.savefig(
                output_dir / f"multivariate_ccm_{target}.png"
            )
            plt.close()
    
    return results

def save_results(results, output_dir, config):
    """Save analysis results."""
    # Implementation depends on result structure
    pass

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run CCM analysis on CSV data'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    run_analysis(args.config)
