"""
Implementation of the MaxCut problem Hamiltonian.

The MaxCut problem seeks to partition vertices of a graph into two sets 
such that the sum of weights of edges between the two sets is maximized.
"""
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from .base import Hamiltonian
from ..utils.pauli_utils import create_zz_term

def create_maxcut_hamiltonian(graph: nx.Graph) -> Hamiltonian:
    """
    Create a Hamiltonian for the MaxCut problem.
    
    The MaxCut Hamiltonian for a weighted graph is:
    H = sum_{(i,j) in E} w_{ij} * (1 - Z_i Z_j) / 2
    
    Args:
        graph: NetworkX graph with weighted edges
        
    Returns:
        Hamiltonian for the MaxCut problem
    """
    # Get number of nodes in the graph
    n_nodes = graph.number_of_nodes()
    hamiltonian = Hamiltonian(n_nodes)
    hamiltonian.metadata["problem"] = "MaxCut"
    hamiltonian.metadata["graph"] = graph
    
    # Add constant term
    constant_term = 0.0
    
    # Process each edge
    for i, j, attr in graph.edges(data=True):
        # Get edge weight
        weight = attr.get('weight', 1.0)
        
        # Add constant term: w_{ij} / 2
        constant_term += weight / 2
        
        # Add interaction term: -w_{ij} * Z_i Z_j / 2
        coeff, term = create_zz_term(i, j, -weight / 2)
        hamiltonian.add_term(coeff, term)
    
    # Add the constant term if non-zero
    if constant_term != 0:
        hamiltonian.add_constant(constant_term)
    
    return hamiltonian

def get_maxcut_solution(bit_string: Union[str, List[int]], graph: nx.Graph) -> Dict:
    """
    Get the solution from a bit string.
    
    Args:
        bit_string: Bit string or list of 0s and 1s representing the solution
        graph: The original graph
        
    Returns:
        Dictionary with solution information
    """
    if isinstance(bit_string, str):
        assignment = [int(bit) for bit in bit_string]
    else:
        assignment = bit_string
    
    # Convert to dictionary for compatibility with test expectations
    assignment_dict = {i: bit for i, bit in enumerate(assignment)}
    
    # Partition the nodes
    set_0 = [i for i, bit in enumerate(assignment) if bit == 0]
    set_1 = [i for i, bit in enumerate(assignment) if bit == 1]
    
    # Calculate the cut value
    cut_value = 0.0
    for i, j, attr in graph.edges(data=True):
        weight = attr.get('weight', 1.0)
        if (i in set_0 and j in set_1) or (i in set_1 and j in set_0):
            cut_value += weight
    
    return {
        "assignment": assignment_dict,
        "partition": [set_0, set_1],
        "cut_value": cut_value,
    } 