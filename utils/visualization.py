# xCCM/utils/visualization.py
"""Visualization tools for CCM analysis."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, Tuple, List
import networkx as nx
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap

class CCMVisualizer:
    """Enhanced visualization tools for CCM analysis."""
    
    def __init__(self, style: str = 'seaborn') -> None:
        """
        Initialize visualizer with specified style.

        Args:
            style: matplotlib style to use
        """
        plt.style.use(style)
        self.colors = sns.color_palette("husl", 8)
        
        # Create custom colormap for causality strength
        self.causality_cmap = LinearSegmentedColormap.from_list(
            'causality',
            ['#FFFFFF', '#0343DF'],  # White to blue
            N=256
        )
        
    def plot_convergence(
        self,
        lib_sizes: np.ndarray,
        correlations: np.ndarray,
        surrogate_correlations: Optional[np.ndarray] = None,
        confidence_level: float = 0.95,
        title: str = "CCM Convergence Plot",
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot CCM convergence with confidence intervals.

        Args:
            lib_sizes: Array of library sizes
            correlations: Array of correlation values
            surrogate_correlations: Optional array of surrogate correlations
            confidence_level: Confidence level for intervals
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot main convergence line
        ax.plot(lib_sizes, correlations, 'o-', 
                color=self.colors[0], label='CCM Correlation')
        
        # Add confidence intervals if surrogate data provided
        if surrogate_correlations is not None:
            percentiles = np.percentile(
                surrogate_correlations,
                [100 * (1 - confidence_level) / 2,
                 100 * (1 + confidence_level) / 2],
                axis=0
            )
            ax.fill_between(
                lib_sizes, 
                percentiles[0], 
                percentiles[1],
                alpha=0.2, 
                color=self.colors[0],
                label=f'{confidence_level*100}% Confidence Interval'
            )
        
        ax.set_xlabel('Library Size')
        ax.set_ylabel('Correlation')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_manifold(
        self,
        manifold: np.ndarray,
        colorvar: Optional[np.ndarray] = None,
        dims: Tuple[int, int] = (0, 1),
        title: str = "Manifold Reconstruction",
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot 2D projection of the manifold.

        Args:
            manifold: Embedded manifold
            colorvar: Optional array for point colors
            dims: Dimensions to plot
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        scatter = ax.scatter(
            manifold[:, dims[0]], 
            manifold[:, dims[1]],
            c=colorvar if colorvar is not None else 'blue',
            cmap='viridis',
            alpha=0.6
        )
        
        if colorvar is not None:
            plt.colorbar(scatter)
            
        ax.set_xlabel(f'Dimension {dims[0]}')
        ax.set_ylabel(f'Dimension {dims[1]}')
        ax.set_title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_causality_network(
        self,
        ccm_results: Dict[Tuple[str, str], Tuple[float, float]],
        significance_threshold: float = 0.05,
        node_layout: str = 'spring',
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot network of causal relationships.

        Args:
            ccm_results: Dictionary of CCM results
            significance_threshold: P-value threshold
            node_layout: Layout algorithm to use
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure object
        """
        G = nx.DiGraph()
        
        # Add edges with significant causal relationships
        for (source, target), (correlation, pvalue) in ccm_results.items():
            if pvalue < significance_threshold:
                G.add_edge(
                    source, 
                    target, 
                    weight=abs(correlation),
                    correlation=correlation
                )
        
        # Create layout
        if node_layout == 'spring':
            pos = nx.spring_layout(G)
        elif node_layout == 'circular':
            pos = nx.circular_layout(G)
        else:
            pos = nx.random_layout(G)
            
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Draw edges with varying thickness and color
        edges = G.edges(data=True)
        weights = [abs(d['correlation']) for (u, v, d) in edges]
        
        nx.draw_networkx_edges(
            G, 
            pos, 
            width=[w * 5 for w in weights],
            edge_color=weights,
            edge_cmap=self.causality_cmap,
            alpha=0.7
        )
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, 
            pos, 
            node_color='lightblue',
            node_size=1000,
            alpha=0.7
        )
        
        # Add labels
        nx.draw_networkx_labels(G, pos)
        
        plt.title("Causal Network (Edge width = strength)")
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_nonlinearity_test(
        self,
        original_stat: float,
        surrogate_stats: np.ndarray,
        title: str = "Nonlinearity Test Results",
        save_path: Optional[str] = None
    ) -> Figure:
        """
        Plot nonlinearity test results.

        Args:
            original_stat: Original test statistic
            surrogate_stats: Array of surrogate statistics
            title: Plot title
            save_path: Optional path to save figure

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot histogram of surrogate statistics
        sns.histplot(
            surrogate_stats,
            stat='density',
            alpha=0.6,
            label='Surrogate Distribution',
            ax=ax
        )
        
        # Add vertical line for original statistic
        ax.axvline(
            original_stat,
            color='red',
            linestyle='--',
            label='Original Statistic'
        )
        
        ax.set_xlabel('Test Statistic')
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig