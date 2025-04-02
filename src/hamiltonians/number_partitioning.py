"""
Implementation of the Number Partitioning Problem Hamiltonian.

The Number Partitioning Problem seeks to divide a set of numbers into two subsets
such that the difference between the sums of the subsets is minimized.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from .base import Hamiltonian
from ..utils.pauli_utils import create_z_term, create_zz_term

def create_number_partitioning_hamiltonian(numbers: List[float]) -> Hamiltonian:
    """
    Create a Hamiltonian for the Number Partitioning Problem.
    
    Following the formulation from Lucas (2014), we use binary variables s_i in {-1, 1}
    to represent which subset each number goes into. The Hamiltonian is:
    
    H = (sum_{i=1}^{n} a_i s_i)^2
    
    where a_i are the numbers to be partitioned.
    
    Args:
        numbers: List of numbers to partition
        
    Returns:
        Hamiltonian for the Number Partitioning Problem
    """
    n_numbers = len(numbers)
    hamiltonian = Hamiltonian(n_numbers)
    hamiltonian.metadata["problem"] = "NumberPartitioning"
    hamiltonian.metadata["numbers"] = numbers
    
    # Expanding (sum_{i=1}^{n} a_i s_i)^2:
    # = sum_{i=1}^{n} a_i^2 + 2 * sum_{i<j} a_i * a_j * s_i * s_j
    
    # Constant term: sum_{i=1}^{n} a_i^2
    constant_term = sum(a**2 for a in numbers)
    hamiltonian.add_constant(constant_term)
    
    # Interaction terms: 2 * sum_{i<j} a_i * a_j * s_i * s_j
    # Since s_i = Z_i in the Ising formulation, we directly add Z_i Z_j terms
    for i in range(n_numbers):
        for j in range(i+1, n_numbers):
            coeff, term = create_zz_term(i, j, 2 * numbers[i] * numbers[j])
            hamiltonian.add_term(coeff, term)
    
    return hamiltonian

def get_number_partitioning_solution(bit_string: Union[str, List[int]], numbers: List[float]) -> Dict:
    """
    Get the Number Partitioning solution from a bit string.
    
    Args:
        bit_string: Bit string or list of 0s and 1s representing the solution
        numbers: List of numbers to partition
        
    Returns:
        Dictionary with solution information
    """
    if isinstance(bit_string, str):
        assignment = [int(bit) for bit in bit_string]
    else:
        assignment = bit_string
    
    # Convert binary (0,1) to spin (-1,1) for partition assignment
    spin_assignment = [2 * bit - 1 for bit in assignment]
    
    # Create the two subsets
    subset_a = []
    subset_b = []
    for i, spin in enumerate(spin_assignment):
        if spin == 1:
            subset_a.append(numbers[i])
        else:
            subset_b.append(numbers[i])
    
    # Calculate sums and difference
    sum_a = sum(subset_a)
    sum_b = sum(subset_b)
    difference = abs(sum_a - sum_b)
    
    # Create a dict for assignment
    assignment_dict = {i: 1 if spin == 1 else 0 for i, spin in enumerate(spin_assignment)}
    
    return {
        "assignment": assignment_dict,
        "subset_a": subset_a,
        "subset_b": subset_b,
        "sum_a": sum_a,
        "sum_b": sum_b,
        "difference": difference
    } 