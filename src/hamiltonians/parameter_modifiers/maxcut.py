
from typing import Callable, Dict
import networkx as nx
import numpy as np

def linear_edge_scaling(weight: float, scaling_factor: float, edge_param) -> float:
    """
    Linearly scale the edge weight by a factor, considering a specific edge parameter.
    
    Args:
        weight: Original edge weight
        scaling_factor: Factor to scale the weight by
        edge_param: Specific parameter of the edge to consider in the scaling
        
    Returns:
        Modified weight
    """
    return weight + scaling_factor * edge_param


def quadratic_edge_scaling(weight: float, scaling_factor: float, edge_param) -> float:
    """
    Linearly scale the edge weight by a factor, considering a specific edge parameter.

    Args:
        weight: Original edge weight
        scaling_factor: Factor to scale the weight by
        edge_param: Specific parameter of the edge to consider in the scaling

    Returns:
        Modified weight
    """
    return weight + (scaling_factor * edge_param) ** 2

def qubic_edge_scaling(weight: float, scaling_factor: float, edge_param) -> float:
    """
    Linearly scale the edge weight by a factor, considering a specific edge parameter.

    Args:
        weight: Original edge weight
        scaling_factor: Factor to scale the weight by
        edge_param: Specific parameter of the edge to consider in the scaling

    Returns:
        Modified weight
    """
    return weight + (scaling_factor * edge_param) ** 3

def edge_density_modifier(weight: float, scaling_factor: float, graph:nx.Graph) -> float:
    """
    Custom modifier that scales weights based on edge density and a scaling factor.
    
    Args:
        weight: Original edge weight
        scaling_factor: Factor to scale the weight by
        graph: NetworkX graph instance
        
    Returns:
        Modified weight
    """
    n_nodes = graph.number_of_nodes()
    max_edges = n_nodes * (n_nodes - 1) / 2
    edge_density = graph.number_of_edges() / max_edges

    return weight * (1.0 + scaling_factor * edge_density)

def degree_weighted_modifier(weight: float, scaling_factor: float, 
                             degree_exponent: float, graph) -> float:
    """
    Custom modifier that scales weights based on node degrees.
    
    Args:
        weight: Original edge weight
        scaling_factor: Factor to scale the weight by
        degree_exponent: Exponent to raise the average degree to
        graph: NetworkX graph instance
        
    Returns:
        Modified weight
    """
    # Calculate average degree
    avg_degree = sum(dict(graph.degree()).values()) / graph.number_of_nodes()
    
    # Scale weight based on average degree raised to the specified exponent
    return weight * (1.0 + scaling_factor * (avg_degree ** degree_exponent))

def weighted_sine_modifier(weight: float, factor: float, angle: float) -> float:
    """
    Apply a weighted sine transformation to the edge weight.
    
    Args:
        weight: Original edge weight
        factor: Factor to multiply by
        angle: Angle for the sine function
        
    Returns:
        Modified weight
    """
    return weight * np.sin(factor * angle)

def weight_emphasis_modifier(weight: float, emphasis_factor: float) -> float:
    """
    Modifier that emphasizes or de-emphasizes edge weights in the MaxCut problem.
    
    Args:
        weight: Original edge weight
        emphasis_factor: Emphasis factor where:
            - emphasis_factor > 1: Increases the influence of heavy-weighted edges
            - emphasis_factor = 1: No change (original weights)
            - 0 < emphasis_factor < 1: Reduces weight differences
            - emphasis_factor = 0: All weights become equal (unweighted MaxCut)
        
    Returns:
        Modified weight
    """
    if emphasis_factor == 1.0:
        return weight
        
    sign = -1 if weight < 0 else 1
    abs_weight = abs(weight)
    
    if emphasis_factor == 0:
        # Make all weights equal (preserves sign)
        return sign
    elif emphasis_factor > 1.0:
        # Emphasize weight differences by raising to a power
        return sign * (abs_weight ** emphasis_factor)
    else:
        # De-emphasize weight differences (bringing them closer to uniform)
        return sign * (abs_weight ** emphasis_factor)

def get_modifiers() -> Dict[str, Callable]:
    """
    Get a dictionary of all MaxCut modifier functions.
    
    Returns:
        Dictionary mapping modifier names to modifier functions
    """
    return {
        "linear_edge_scaling": linear_edge_scaling,
        "quadratic_edge_scaling": quadratic_edge_scaling,
        "qubic_edge_scaling": qubic_edge_scaling,
        "edge_density_scaling": edge_density_modifier,
        "degree_weighted": degree_weighted_modifier,
        "weighted_sine": weighted_sine_modifier,
        "weight_emphasis": weight_emphasis_modifier,
    } 