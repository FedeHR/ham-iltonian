"""
Number Partitioning Hamiltonian implementation.

This module provides functions for creating and analyzing Hamiltonians
for the Number Partitioning problem.
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from .base import Hamiltonian
from ..utils.pauli_utils import create_z_term, create_zz_term, pauli_term_to_pennylane

def create_number_partitioning_hamiltonian(numbers: List[float]) -> Hamiltonian:
    """
    Create a Hamiltonian for the Number Partitioning problem.
    
    The Hamiltonian is designed to minimize the squared difference between the
    subset sums, which is equivalent to minimizing (S_a - S_b)^2, where S_a and S_b
    are the sums of subsets A and B respectively.
    
    This can be expanded as (2*S_a - S_total)^2, where S_a is the sum of elements
    with value 1 in the bit string, and S_total is the sum of all numbers.
    
    Let x_i be the binary decision variable (0 or 1) for each element.
    Then the Hamiltonian is (2*sum(a_i * x_i) - sum(a_i))^2, 
    where a_i are the numbers.
    
    Args:
        numbers: List of numbers to partition
        
    Returns:
        Hamiltonian for the Number Partitioning problem
    """
    n = len(numbers)
    total_sum = sum(numbers)
    hamiltonian = Hamiltonian(n)
    hamiltonian.metadata["problem"] = "NumberPartitioning"
    hamiltonian.metadata["numbers"] = numbers
    
    # Expand (2*sum(a_i * x_i) - total_sum)^2
    # = 4*sum(a_i * x_i)^2 - 4*total_sum*sum(a_i * x_i) + total_sum^2
    # = 4*sum(a_i^2 * x_i^2) + 4*sum(a_i * a_j * x_i * x_j) - 4*total_sum*sum(a_i * x_i) + total_sum^2
    # Since x_i^2 = x_i for binary variables:
    # = 4*sum(a_i^2 * x_i) + 4*sum_i!=j(a_i * a_j * x_i * x_j) - 4*total_sum*sum(a_i * x_i) + total_sum^2
    
    # Constant term: total_sum^2
    hamiltonian.add_constant(total_sum**2)
    
    for i in range(n):
        # Linear terms: 4*a_i^2 - 4*total_sum*a_i
        coeff = 4 * (numbers[i]**2) - 4 * total_sum * numbers[i]
        coeff_z, term_z = create_z_term(i, coeff)
        hamiltonian.add_term(coeff_z, term_z)
        
    for i in range(n):
        for j in range(i+1, n):
            # Quadratic terms: 8*a_i*a_j
            coeff = 8 * numbers[i] * numbers[j]
            coeff_zz, term_zz = create_zz_term(i, j, coeff)
            hamiltonian.add_term(coeff_zz, term_zz)
    
    return hamiltonian

def get_number_partitioning_solution(bitstring: str, numbers: List[float]) -> Dict[str, Any]:
    """
    Calculate the solution quality for a given bitstring.
    
    Args:
        bitstring: Binary string representation of a solution
        numbers: List of numbers to partition
        
    Returns:
        Dictionary with solution details
    """
    # Extract the binary decisions from the bitstring
    if len(bitstring) != len(numbers):
        raise ValueError(f"Bitstring length ({len(bitstring)}) does not match number of elements ({len(numbers)})")
    
    # Create the subsets
    subset_a = [numbers[i] for i, bit in enumerate(bitstring) if bit == "1"]
    subset_b = [numbers[i] for i, bit in enumerate(bitstring) if bit == "0"]
    
    # Calculate the sums
    sum_a = sum(subset_a)
    sum_b = sum(subset_b)
    
    # Calculate the difference (this is what we want to minimize)
    difference = abs(sum_a - sum_b)
    
    # Return the solution
    return {
        "subset_a": subset_a,
        "subset_b": subset_b,
        "sum_a": sum_a,
        "sum_b": sum_b,
        "difference": difference,
        "bitstring": bitstring,
        "valid": True,  # All bitstrings are valid solutions for number partitioning
        "quality": -difference,  # Negative because we want to minimize difference
    } 