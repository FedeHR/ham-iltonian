"""
This module provides helper functions for creating and manipulating graphs for various
optimization problems.
"""
import networkx as nx
import numpy as np
from typing import List, Tuple, Optional

def create_random_weighted_graph(n_nodes: int,
                                 edge_probability: float = 0.5,
                                 init_weight_range: Tuple[float, float] = (0.1, 1),
                                 edge_param_range: Tuple[float, float] = (0.1, 1),
                                 seed: Optional[int] = None) -> nx.Graph:
    """
    Create a random weighted graph.
    
    Args:
        n_nodes: Number of nodes in the graph
        edge_probability: Probability of edge creation between any two nodes
        init_weight_range: Range of initial edge weights (min, max)
        edge_param_range: Range of additional per edge parameter weights (min, max)
        seed: Random seed for reproducibility
        
    Returns:
        Erdos-Renyi graph with random weighted edges
    """
    if seed is not None:
        np.random.seed(seed)
        
    G = nx.erdos_renyi_graph(n=n_nodes, p=edge_probability, seed=seed)
    
    # Add random initial weights and additional parameter to edges
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.uniform(init_weight_range[0], init_weight_range[1])
        G[v][u]['edge_param'] = np.random.uniform(edge_param_range[0], edge_param_range[1])
        
    return G

def create_complete_weighted_graph(n_nodes: int,
                                   init_weight_range: Tuple[float, float] = (0.1, 1.0),
                                   edge_param_range: Tuple[float, float] = (0.1, 1),
                                   seed: Optional[int] = None) -> nx.Graph:
    """
    Create a complete weighted graph where all nodes are connected.
    
    Args:
        n_nodes: Number of nodes in the graph
        init_weight_range: Range of initial edge weights (min, max)
        edge_param_range: Range of additional per edge parameter weights (min, max)
        seed: Random seed for reproducibility
        
    Returns:
        NetworkX complete graph with random weighted edges
    """
    if seed is not None:
        np.random.seed(seed)
        
    G = nx.complete_graph(n=n_nodes)
    
    # Add random initial weights and additional parameter to edges
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.uniform(init_weight_range[0], init_weight_range[1])
        G[v][u]['edge_param'] = np.random.uniform(edge_param_range[0], edge_param_range[1])
        
    return G

def create_grid_graph(rows: int,
                      cols: int,
                      init_weight_range: Tuple[float, float] = (0.1, 1.0),
                      edge_param_range: Tuple[float, float] = (0.1, 1),
                      seed: Optional[int] = None) -> nx.Graph:
    """
    Create a 2D grid graph with random weights.
    
    Args:
        rows: Number of rows in the grid
        cols: Number of columns in the grid
        init_weight_range: Range of edge weights (min, max)
        edge_param_range: Range of additional per edge parameter weights (min, max)
        seed: Random seed for reproducibility
        
    Returns:
        NetworkX grid graph with random weighted edges
    """
    if seed is not None:
        np.random.seed(seed)
        
    G = nx.grid_2d_graph(rows, cols)
    
    # Convert to normal integer labeling
    G = nx.convert_node_labels_to_integers(G)
    
    # Add random initial weights and additional parameter to edges
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.uniform(init_weight_range[0], init_weight_range[1])
        G[v][u]['edge_param'] = np.random.uniform(edge_param_range[0], edge_param_range[1])
        
    return G

def create_weighted_graph_from_distances(coordinates: List[Tuple[float, float]], 
                                       fully_connected: bool = True,
                                       distance_p: float = 2.0) -> nx.Graph:
    """
    Create a weighted graph where edge weights are distances between coordinates.
    
    Args:
        coordinates: List of (x,y) coordinates for each node
        fully_connected: Whether to create a fully connected graph (default: True)
        distance_p: The p-norm to use for distance calculation (default: 2.0 for Euclidean)
        
    Returns:
        NetworkX graph with edges weighted by distances
    """
    n_nodes = len(coordinates)
    G = nx.Graph()
    
    # Add nodes
    for i in range(n_nodes):
        G.add_node(i, pos=coordinates[i])
    
    # Add edges with distance weights
    for i in range(n_nodes):
        if fully_connected:
            # For fully connected graphs, connect to all other nodes
            for j in range(i+1, n_nodes):
                dist = np.linalg.norm(
                    np.array(coordinates[i]) - np.array(coordinates[j]), 
                    ord=distance_p
                )
                G.add_edge(i, j, weight=dist)
    
    return G 