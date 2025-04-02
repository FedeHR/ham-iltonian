"""
Implementation of the Knapsack Problem Hamiltonian.

The Knapsack Problem seeks to maximize the value of items in a knapsack without 
exceeding the capacity constraint.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from .base import Hamiltonian
from ..utils.pauli_utils import create_z_term, create_zz_term

def create_knapsack_hamiltonian(
    values: List[float], 
    weights: List[float], 
    capacity: float, 
    penalty: float = None
) -> Hamiltonian:
    """
    Create a Hamiltonian for the Knapsack Problem.
    
    Following the formulation from Lucas (2014), we use binary variables x_i to 
    represent whether item i is included in the knapsack. The Hamiltonian consists
    of two terms:
    
    H = H_A + H_B
    
    H_A: Penalty term for exceeding capacity
    H_B: Objective term to maximize total value
    
    Args:
        values: List of values for each item
        weights: List of weights for each item
        capacity: Maximum capacity of the knapsack
        penalty: Penalty coefficient for constraint (if None, automatically calculated)
        
    Returns:
        Hamiltonian for the Knapsack Problem
    """
    n_items = len(values)
    assert len(weights) == n_items, "Values and weights must have the same length"
    
    if penalty is None:
        # Set penalty to be larger than the max possible value
        penalty = sum(values) + 1.0
    
    hamiltonian = Hamiltonian(n_items)
    hamiltonian.metadata["problem"] = "Knapsack"
    hamiltonian.metadata["n_items"] = n_items
    hamiltonian.metadata["values"] = values
    hamiltonian.metadata["weights"] = weights
    hamiltonian.metadata["capacity"] = capacity
    
    # H_A: Penalty term for exceeding capacity
    # H_A = A * (sum_{i=0}^{n-1} w_i * x_i - C)^2, if sum_{i=0}^{n-1} w_i * x_i > C
    # H_A = 0, otherwise
    #
    # We can rewrite as:
    # H_A = A * [max(0, sum_{i=0}^{n-1} w_i * x_i - C)]^2
    #
    # Following Lucas (2014), we introduce slack variables y_j to rewrite this as:
    # sum_{i=0}^{n-1} w_i * x_i - C = sum_{j=0}^{K-1} 2^j * y_j
    #
    # where K is chosen such that 2^K > sum_{i=0}^{n-1} w_i - C
    
    # Calculate the max possible excess weight
    max_excess = sum(weights) - capacity
    if max_excess <= 0:
        # No need for slack variables, the capacity is sufficient for all items
        # Only need to maximize value
        for i in range(n_items):
            # Objective: minimize negative values to maximize values
            coeff, term = create_z_term(i, -values[i] / 2)
            hamiltonian.add_term(coeff, term)
            
        # Add constant to make objective function match actual value
        hamiltonian.add_constant(sum(values) / 2)
        
        return hamiltonian
    
    # Determine number of slack variables needed
    K = int(np.ceil(np.log2(max_excess + 1)))
    n_slack = K
    
    # Total qubits = n_items (original variables) + n_slack (slack variables)
    n_qubits = n_items + n_slack
    hamiltonian.num_qubits = n_qubits
    
    # Add the objective term: -sum_{i=0}^{n-1} v_i * x_i
    for i in range(n_items):
        # Objective: minimize negative values to maximize values
        coeff, term = create_z_term(i, -values[i] / 2)
        hamiltonian.add_term(coeff, term)
        
    # Add constant to make objective function match actual value
    hamiltonian.add_constant(sum(values) / 2)
    
    # Add the constraint term using the slack variables
    # We need to enforce:
    # sum_{i=0}^{n-1} w_i * x_i - C = sum_{j=0}^{K-1} 2^j * y_j
    #
    # This gives us:
    # (sum_{i=0}^{n-1} w_i * x_i - C - sum_{j=0}^{K-1} 2^j * y_j)^2
    
    # The full expansion would be quadratic in x_i and y_j, resulting in:
    # (sum_{i=0}^{n-1} w_i * x_i - C - sum_{j=0}^{K-1} 2^j * y_j)^2
    # = (sum_{i=0}^{n-1} w_i * x_i)^2 - 2C*sum_{i=0}^{n-1} w_i * x_i + C^2
    #   - 2*sum_{i=0}^{n-1} w_i * x_i * sum_{j=0}^{K-1} 2^j * y_j
    #   + 2C*sum_{j=0}^{K-1} 2^j * y_j + (sum_{j=0}^{K-1} 2^j * y_j)^2
    
    # (sum_{i=0}^{n-1} w_i * x_i)^2 expands to:
    # sum_{i=0}^{n-1} w_i^2 * x_i^2 + 2*sum_{i<j} w_i * w_j * x_i * x_j
    
    # For x_i^2, since x_i is binary, x_i^2 = x_i, so:
    for i in range(n_items):
        coeff, term = create_z_term(i, penalty * weights[i]**2 / 4)
        hamiltonian.add_term(coeff, term)
    
    # For x_i * x_j (i != j):
    for i in range(n_items):
        for j in range(i+1, n_items):
            coeff, term = create_zz_term(i, j, penalty * weights[i] * weights[j] / 4)
            hamiltonian.add_term(coeff, term)
    
    # -2C*sum_{i=0}^{n-1} w_i * x_i
    for i in range(n_items):
        coeff, term = create_z_term(i, -penalty * capacity * weights[i] / 2)
        hamiltonian.add_term(coeff, term)
    
    # -2*sum_{i=0}^{n-1} w_i * x_i * sum_{j=0}^{K-1} 2^j * y_j
    for i in range(n_items):
        for j in range(n_slack):
            # y_j is at index n_items + j
            slack_idx = n_items + j
            coeff, term = create_zz_term(i, slack_idx, -penalty * weights[i] * (2**j) / 2)
            hamiltonian.add_term(coeff, term)
    
    # 2C*sum_{j=0}^{K-1} 2^j * y_j
    for j in range(n_slack):
        slack_idx = n_items + j
        coeff, term = create_z_term(slack_idx, penalty * capacity * (2**j) / 2)
        hamiltonian.add_term(coeff, term)
    
    # (sum_{j=0}^{K-1} 2^j * y_j)^2 expands similarly to (sum_{i} w_i * x_i)^2
    for j in range(n_slack):
        slack_idx = n_items + j
        coeff, term = create_z_term(slack_idx, penalty * (2**j)**2 / 4)
        hamiltonian.add_term(coeff, term)
    
    for j in range(n_slack):
        for k in range(j+1, n_slack):
            slack_j = n_items + j
            slack_k = n_items + k
            coeff, term = create_zz_term(slack_j, slack_k, penalty * (2**j) * (2**k) / 4)
            hamiltonian.add_term(coeff, term)
    
    # Constant term C^2 from the expansion
    hamiltonian.add_constant(penalty * capacity**2 / 4)
    
    return hamiltonian

def get_knapsack_solution(bit_string: Union[str, List[int]], values: List[float], weights: List[float], capacity: float) -> Dict:
    """
    Get the Knapsack solution from a bit string.
    
    Args:
        bit_string: Bit string or list of 0s and 1s representing the solution
        values: List of values for each item
        weights: List of weights for each item
        capacity: Maximum capacity of the knapsack
        
    Returns:
        Dictionary with solution information
    """
    n_items = len(values)
    
    if isinstance(bit_string, str):
        assignment = [int(bit) for bit in bit_string]
    else:
        assignment = bit_string
    
    # Extract the item selection (first n_items bits)
    items_selected = assignment[:n_items]
    
    # Calculate total value and weight
    total_value = sum(v * x for v, x in zip(values, items_selected))
    total_weight = sum(w * x for w, x in zip(weights, items_selected))
    
    # Check if the solution is valid
    is_valid = total_weight <= capacity
    
    # Get indices of selected items
    selected_indices = [i for i, x in enumerate(items_selected) if x == 1]
    
    return {
        "selected_items": selected_indices,
        "total_value": total_value,
        "total_weight": total_weight,
        "valid": is_valid
    } 