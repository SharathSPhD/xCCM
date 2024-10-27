# tests/test_network.py
"""Tests for network analysis functionality."""

import pytest
import numpy as np
from xCCM.modules.network import CCMNetwork

def test_network_initialization():
    """Test network initialization."""
    network = CCMNetwork()
    assert len(network.graph.nodes()) == 0
    assert len(network.graph.edges()) == 0

def test_network_construction(sample_ccm_results):
    """Test network construction from CCM results."""
    network = CCMNetwork()
    network.build_network(sample_ccm_results)
    
    assert len(network.graph.nodes()) > 0
    assert len(network.graph.edges()) > 0

def test_driver_identification(sample_ccm_results):
    """Test driver variable identification."""
    network = CCMNetwork()
    network.build_network(sample_ccm_results)
    
    drivers = network.identify_drivers()
    assert isinstance(drivers, dict)
    assert len(drivers) == len(network.graph.nodes())
    assert all(isinstance(v, float) for v in drivers.values())

def test_feedback_loops(sample_ccm_results):
    """Test feedback loop detection."""
    network = CCMNetwork()
    network.build_network(sample_ccm_results)
    
    loops = network.find_feedback_loops()
    assert isinstance(loops, list)
    assert all(isinstance(loop, list) for loop in loops)