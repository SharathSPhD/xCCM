# xCCM/modules/network.py
"""Network analysis tools for CCM results."""

import networkx as nx
import numpy as np
from typing import Dict, List, Tuple

class CCMNetwork:
    """Network analysis tools for CCM results."""
    
    def __init__(self) -> None:
        """Initialize CCMNetwork with empty directed graph."""
        self.graph = nx.DiGraph()
        
    def build_network(
        self,
        ccm_results: Dict[Tuple[str, str], Tuple[float, float]],
        significance_threshold: float = 0.05
    ) -> None:
        """
        Build network from CCM results.

        Args:
            ccm_results: Dictionary of (source, target) -> (correlation, p-value)
            significance_threshold: P-value threshold for including edges
        """
        self.graph.clear()
        
        for (source, target), (correlation, pvalue) in ccm_results.items():
            if pvalue < significance_threshold:
                self.graph.add_edge(
                    source, 
                    target,
                    weight=abs(correlation),
                    correlation=correlation,
                    pvalue=pvalue
                )
                
    def identify_drivers(self) -> Dict[str, float]:
        """
        Identify driving variables using network metrics.

        Returns:
            Dictionary of node -> driver score, sorted by score
        """
        # Compute various centrality measures
        out_degree = dict(self.graph.out_degree(weight='weight'))
        betweenness = nx.betweenness_centrality(self.graph, weight='weight')
        pagerank = nx.pagerank(self.graph, weight='weight')
        
        # Combine metrics
        drivers = {}
        for node in self.graph.nodes():
            score = (
                out_degree.get(node, 0) + 
                betweenness.get(node, 0) + 
                pagerank.get(node, 0)
            ) / 3
            drivers[node] = score
            
        return dict(sorted(
            drivers.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
    
    def find_feedback_loops(self) -> List[List[str]]:
        """
        Identify feedback loops in the causal network.

        Returns:
            List of cycles, sorted by length
        """
        cycles = list(nx.simple_cycles(self.graph))
        return sorted(cycles, key=len)
    
    def compute_network_statistics(self) -> Dict[str, float]:
        """
        Compute various network statistics.

        Returns:
            Dictionary of network statistics
        """
        stats = {
            'density': nx.density(self.graph),
            'transitivity': nx.transitivity(self.graph),
            'reciprocity': nx.reciprocity(self.graph),
            'avg_clustering': nx.average_clustering(self.graph)
        }
        
        # Add average shortest path length if graph is connected
        if nx.is_strongly_connected(self.graph):
            stats['avg_shortest_path'] = nx.average_shortest_path_length(
                self.graph
            )
        else:
            stats['avg_shortest_path'] = float('inf')
            
        return stats
    
    def get_strongest_connections(
        self,
        top_n: int = 5
    ) -> List[Tuple[str, str, float]]:
        """
        Get the strongest causal connections in the network.

        Args:
            top_n: Number of connections to return

        Returns:
            List of (source, target, weight) tuples
        """
        edges = [
            (u, v, d['weight']) 
            for u, v, d in self.graph.edges(data=True)
        ]
        return sorted(edges, key=lambda x: x[2], reverse=True)[:top_n]