"""
Utility functions for creating problem instances.

This module provides helper functions for creating different problem instances
for various optimization problems.
"""
import networkx as nx
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union

from ..utils.graph_utils import (
    create_random_weighted_graph, 
    create_complete_weighted_graph,
    create_grid_graph,
    create_weighted_graph_from_distances
)

def create_maxcut_instance(
    n_nodes: int = 5,
    edge_probability: float = 0.7,
    weight_range: Tuple[float, float] = (0.5, 2.0),
    graph_type: str = "random", 
    grid_dims: Optional[Tuple[int, int]] = None,
    name: str = "MaxCut Instance",
    seed: Optional[int] = None
):
    """
    Create a MaxCut problem instance.
    
    Args:
        n_nodes: Number of nodes in the graph
        edge_probability: Probability of edge creation between any two nodes (for random graphs)
        weight_range: Range of edge weights (min, max)
        graph_type: Type of graph ("random", "complete", "grid")
        grid_dims: Dimensions (rows, cols) for grid graph (ignores n_nodes)
        name: Name of the problem instance
        seed: Random seed for reproducibility
        
    Returns:
        MaxCutProblem instance
    """
    # Import inside function to avoid circular imports
    from .maxcut import MaxCutProblem
    
    if graph_type == "random":
        graph = create_random_weighted_graph(
            n_nodes=n_nodes, 
            edge_probability=edge_probability,
            weight_range=weight_range,
            seed=seed
        )
    elif graph_type == "complete":
        graph = create_complete_weighted_graph(
            n_nodes=n_nodes,
            weight_range=weight_range,
            seed=seed
        )
    elif graph_type == "grid":
        rows, cols = grid_dims if grid_dims else (int(np.sqrt(n_nodes)), int(np.sqrt(n_nodes)))
        graph = create_grid_graph(
            rows=rows,
            cols=cols,
            weight_range=weight_range,
            seed=seed
        )
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")
        
    return MaxCutProblem(graph, name=name)

def create_tsp_instance(
    n_cities: int = 5,
    coordinates: Optional[List[Tuple[float, float]]] = None,
    coordinate_range: Tuple[float, float] = (0, 100),
    name: str = "TSP Instance",
    seed: Optional[int] = None
):
    """
    Create a TSP problem instance.
    
    Args:
        n_cities: Number of cities (nodes)
        coordinates: Optional list of (x,y) coordinates for each city
        coordinate_range: Range for random coordinates if not provided
        name: Name of the problem instance
        seed: Random seed for reproducibility
        
    Returns:
        TSPProblem instance
    """
    # Import inside function to avoid circular imports
    from .tsp import TSPProblem
    
    if seed is not None:
        np.random.seed(seed)
        
    if coordinates is None:
        # Generate random coordinates
        coordinates = [
            (
                np.random.uniform(coordinate_range[0], coordinate_range[1]),
                np.random.uniform(coordinate_range[0], coordinate_range[1])
            )
            for _ in range(n_cities)
        ]
    
    # Ensure n_cities is correct based on provided coordinates
    n_cities = len(coordinates)
    
    # Create a fully connected graph with distances as weights
    graph = create_weighted_graph_from_distances(coordinates, fully_connected=True)
    
    # Convert the graph into a distance matrix
    distances = np.zeros((n_cities, n_cities))
    for i in range(n_cities):
        for j in range(n_cities):
            if i != j:
                if graph.has_edge(i, j):
                    distances[i, j] = graph[i][j]['weight']
                else:
                    # If edge doesn't exist in the graph, use a large distance
                    distances[i, j] = 1000.0
    
    # Create city positions array from coordinates
    city_positions = np.array(coordinates)
    
    # Create optional city names
    city_names = [f"City {i}" for i in range(n_cities)]
    
    return TSPProblem(distances=distances, city_names=city_names, city_positions=city_positions, name=name)

def create_knapsack_instance(
    n_items: int = 10,
    max_weight: float = 100.0,
    value_range: Tuple[float, float] = (10, 50),
    weight_range: Tuple[float, float] = (5, 30),
    name: str = "Knapsack Instance",
    seed: Optional[int] = None
):
    """
    Create a Knapsack problem instance.
    
    Args:
        n_items: Number of items
        max_weight: Maximum weight capacity of the knapsack
        value_range: Range of values for items (min, max)
        weight_range: Range of weights for items (min, max)
        name: Name of the problem instance
        seed: Random seed for reproducibility
        
    Returns:
        KnapsackProblem instance
    """
    # Import inside function to avoid circular imports
    from .knapsack import KnapsackProblem
    
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random values and weights
    values = np.random.uniform(value_range[0], value_range[1], n_items)
    weights = np.random.uniform(weight_range[0], weight_range[1], n_items)
    
    # Note: KnapsackProblem uses 'capacity' instead of 'max_weight'
    return KnapsackProblem(values=values, weights=weights, capacity=max_weight, name=name)

def create_portfolio_instance(
    n_assets: int = 5,
    returns: Optional[np.ndarray] = None,
    risk_matrix: Optional[np.ndarray] = None,
    return_range: Tuple[float, float] = (0.01, 0.15),
    risk_factor: float = 0.5,
    budget: int = 3,
    name: str = "Portfolio Instance",
    seed: Optional[int] = None
):
    """
    Create a Portfolio Optimization problem instance.
    
    Args:
        n_assets: Number of assets
        returns: Optional array of expected returns
        risk_matrix: Optional covariance matrix for risk
        return_range: Range for random returns if not provided
        risk_factor: Factor for random risk generation
        budget: Maximum number of assets to select (default: 3)
        name: Name of the problem instance
        seed: Random seed for reproducibility
        
    Returns:
        PortfolioProblem instance
    """
    # Import inside function to avoid circular imports
    from .portfolio import PortfolioProblem
    
    if seed is not None:
        np.random.seed(seed)
    
    if returns is None:
        # Generate random returns
        returns = np.random.uniform(return_range[0], return_range[1], n_assets)
    
    if risk_matrix is None:
        # Generate a random but valid covariance matrix
        # First generate a random correlation matrix
        A = np.random.randn(n_assets, n_assets)
        corr = A @ A.T  # This creates a positive semi-definite matrix
        
        # Ensure diagonal is 1 for correlation matrix
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)
        
        # Now create covariance matrix with random volatilities
        vols = np.random.uniform(0.05, 0.3, n_assets)
        risk_matrix = corr * np.outer(vols, vols) * risk_factor
    
    return PortfolioProblem(returns=returns, risk_matrix=risk_matrix, budget=budget, risk_factor=risk_factor, name=name)

def create_number_partitioning_instance(
    numbers: Optional[List[float]] = None,
    n_numbers: int = 10,
    number_range: Tuple[float, float] = (1, 100),
    name: str = "Number Partitioning Instance",
    seed: Optional[int] = None
):
    """
    Create a Number Partitioning problem instance.
    
    Args:
        numbers: Optional list of numbers to partition
        n_numbers: Number of numbers to generate if numbers is None
        number_range: Range for random numbers if numbers is None
        name: Name of the problem instance
        seed: Random seed for reproducibility
        
    Returns:
        NumberPartitioningProblem instance
    """
    # Import here to avoid circular imports
    # Since there's no NumberPartitioningProblem class, we'll raise an error
    # This will need to be implemented
    raise NotImplementedError("NumberPartitioningProblem is not yet implemented")