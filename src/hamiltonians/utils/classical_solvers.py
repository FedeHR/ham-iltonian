"""
Classical solvers for combinatorial optimization problems.

This module provides brute force solvers for small instances of
the optimization problems supported by the library.
"""
import numpy as np
import networkx as nx
from typing import Dict, List, Any, Tuple
import itertools

def solve_maxcut_brute_force(graph: nx.Graph) -> Dict[str, Any]:
    """
    Solve the MaxCut problem using brute force.
    
    Args:
        graph: NetworkX graph with weighted edges
        
    Returns:
        Dictionary containing:
            - 'bitstring': Optimal binary string representation
            - 'assignment': Dictionary mapping nodes to 0/1 values
            - 'partition': List of two sets containing the nodes in each partition
            - 'cut_value': Value of the maximum cut
    """
    n_nodes = graph.number_of_nodes()
    optimal_cut = -float('inf')
    optimal_bitstring = None
    optimal_assignment = None
    optimal_partition = None
    
    # Try all possible bitstrings
    for i in range(2**n_nodes):
        bitstring = np.binary_repr(i, width=n_nodes)
        assignment = [int(bit) for bit in bitstring]

        assignment_dict, cut_value, set_0, set_1 = evaluate_mc_solution(assignment, graph)

        if cut_value > optimal_cut:
            optimal_cut = cut_value
            optimal_bitstring = bitstring
            optimal_assignment = assignment_dict
            optimal_partition = [set_0, set_1]
    
    return {
        "bitstring": optimal_bitstring,
        "assignment": optimal_assignment,
        "partition": optimal_partition,
        "cut_value": optimal_cut
    }


def evaluate_mc_solution(assignment: List[int], graph: nx.Graph):
    assignment_dict = convert_mc_bitstring_to_dict(assignment)
    set_0, set_1 = extract_mc_partitions(assignment_dict)
    cut_value = calculate_mc_value(graph, set_0, set_1)
    return assignment_dict, cut_value, set_0, set_1


def calculate_mc_value(graph: nx.Graph, set_0: List[int], set_1: List[int]) -> float:
    cut_value = 0.0
    for u, v, attr in graph.edges(data=True):
        weight = attr.get('weight', 1.0)
        if (u in set_0 and v in set_1) or (u in set_1 and v in set_0):
            cut_value += weight
    return cut_value


def extract_mc_partitions(assignment: Dict[int, int]) -> Tuple[List[int], List[int]]:
    set_0 = [node for node, partition in assignment.items() if partition == 0]
    set_1 = [node for node, partition in assignment.items() if partition == 1]
    return set_0, set_1


def convert_mc_bitstring_to_dict(assignment: List[int]) -> Dict[int, int]:
    assignment_dict = {node: partition for node, partition in enumerate(assignment)}
    return assignment_dict


def solve_tsp_brute_force(distances: np.ndarray) -> Dict[str, Any]:
    """
    Solve the Traveling Salesman Problem using brute force.
    
    Args:
        distances: Distance matrix between cities (n x n)
        
    Returns:
        Dictionary containing:
            - 'bitstring': Optimal binary string representation
            - 'tour': List of cities in the optimal tour
            - 'total_distance': Total distance of the optimal tour
            - 'valid': True (always valid for brute force solution)
    """
    n_cities = distances.shape[0]
    optimal_distance = float('inf')
    optimal_tour = None
    
    # Generate all permutations of cities
    for perm in itertools.permutations(range(n_cities)):
        # Calculate tour distance
        tour_distance = 0
        for i in range(n_cities):
            from_city = perm[i]
            to_city = perm[(i + 1) % n_cities]
            tour_distance += distances[from_city, to_city]
        
        # Update if better solution found
        if tour_distance < optimal_distance:
            optimal_distance = tour_distance
            optimal_tour = perm
    
    # Convert the optimal tour to a binary string representation
    # For n cities, we need n^2 qubits: x_{i,p} means city i at position p
    bitstring = ['0'] * (n_cities * n_cities)
    for pos, city in enumerate(optimal_tour):
        # Set x_{city,pos} = 1
        idx = city * n_cities + pos
        bitstring[idx] = '1'
    
    optimal_bitstring = ''.join(bitstring)
    
    # Create solution dictionary directly
    solution = {
        'bitstring': optimal_bitstring,
        'tour': list(optimal_tour),
        'total_distance': optimal_distance,
        'valid': True
    }
    
    return solution

def solve_knapsack_brute_force(values: List[float], weights: List[float], capacity: float) -> Dict[str, Any]:
    """
    Solve the Knapsack Problem using brute force.
    
    Args:
        values: List of values for each item
        weights: List of weights for each item
        capacity: Maximum capacity of the knapsack
        
    Returns:
        Dictionary containing:
            - 'bitstring': Optimal binary string representation
            - 'selected_items': Indices of items in the optimal solution
            - 'total_value': Total value of the optimal solution
            - 'total_weight': Total weight of the optimal solution
            - 'valid': True (always valid for brute force solution)
    """
    n_items = len(values)
    optimal_value = -float('inf')
    optimal_bitstring = None
    
    # Try all possible item combinations
    for i in range(2**n_items):
        bitstring = np.binary_repr(i, width=n_items)
        
        # Calculate value and weight
        selected_items = [j for j, bit in enumerate(bitstring) if bit == '1']
        total_value = sum(values[j] for j in selected_items)
        total_weight = sum(weights[j] for j in selected_items)
        
        # Only consider valid solutions
        if total_weight <= capacity and total_value > optimal_value:
            optimal_value = total_value
            optimal_bitstring = bitstring
    
    # Get the full solution details
    selected_items = [i for i, bit in enumerate(optimal_bitstring) if bit == '1']
    total_value = sum(values[i] for i in selected_items)
    total_weight = sum(weights[i] for i in selected_items)
    
    return {
        'bitstring': optimal_bitstring,
        'selected_items': selected_items,
        'total_value': total_value,
        'total_weight': total_weight,
        'valid': True
    }

def solve_portfolio_brute_force(
    returns: List[float], 
    risk_matrix: np.ndarray, 
    budget: float, 
    risk_factor: float = 1.0
) -> Dict[str, Any]:
    """
    Solve the Portfolio Optimization Problem using brute force.
    
    Args:
        returns: List of expected returns for each asset
        risk_matrix: Covariance matrix representing risk
        budget: Maximum number of assets to select
        risk_factor: Weight for risk term (higher values mean more risk-averse)
        
    Returns:
        Dictionary containing:
            - 'bitstring': Optimal binary string representation
            - 'selected_assets': Indices of assets in the optimal solution
            - 'total_selected': Number of assets selected
            - 'expected_return': Expected return of the portfolio
            - 'risk': Risk of the portfolio
            - 'objective': Combined objective value (return - risk_factor * risk)
            - 'valid': True (always valid for brute force solution)
    """
    n_assets = len(returns)
    optimal_objective = -float('inf')
    optimal_bitstring = None
    optimal_solution = None
    
    # Try all possible asset combinations
    for i in range(2**n_assets):
        bitstring = np.binary_repr(i, width=n_assets)
        
        # Extract selected assets
        selected_assets = [j for j, bit in enumerate(bitstring) if bit == '1']
        total_selected = len(selected_assets)
        
        # Check if solution meets budget constraint
        if total_selected > budget:
            continue
            
        # Calculate expected return
        expected_return = sum(returns[j] for j in selected_assets)
        
        # Calculate risk (portfolio variance)
        risk = 0.0
        for i in selected_assets:
            for j in selected_assets:
                risk += risk_matrix[i, j]
        
        # Calculate objective: maximize return while minimizing risk
        objective = expected_return - risk_factor * risk
        
        # Update if better solution found
        if objective > optimal_objective:
            optimal_objective = objective
            optimal_bitstring = bitstring
            optimal_solution = {
                'bitstring': bitstring,
                'selected_assets': selected_assets,
                'total_selected': total_selected,
                'expected_return': expected_return,
                'risk': risk,
                'objective': objective,
                'valid': True
            }
    
    return optimal_solution

def solve_number_partitioning_brute_force(numbers: List[float]) -> Dict[str, Any]:
    """
    Solve the Number Partitioning problem using brute force.
    
    Args:
        numbers: List of numbers to partition
        
    Returns:
        Dictionary with solution details
    """
    n = len(numbers)
    best_solution = None
    best_difference = float('inf')
    
    # Brute force through all 2^n possible partitions
    # We only need 2^(n-1) because swapping A and B gives the same difference
    for i in range(2**(n-1)):
        # Convert i to a binary string of length n
        # We fix the first number to always be in set A (value 1)
        # This halves the search space
        bitstring = "1" + format(i, f'0{n-1}b')
        
        # Create the two subsets
        subset_a = [numbers[j] for j, bit in enumerate(bitstring) if bit == "1"]
        subset_b = [numbers[j] for j, bit in enumerate(bitstring) if bit == "0"]
        
        # Calculate sums and difference
        sum_a = sum(subset_a)
        sum_b = sum(subset_b)
        difference = abs(sum_a - sum_b)
        
        # Track the best solution found
        if difference < best_difference:
            best_difference = difference
            best_solution = {
                "subset_a": subset_a,
                "subset_b": subset_b,
                "sum_a": sum_a,
                "sum_b": sum_b,
                "difference": difference,
                "bitstring": bitstring,
                "valid": True,
                "quality": -difference  # Negative because we want to minimize difference
            }
    
    return best_solution 