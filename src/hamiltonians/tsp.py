"""
Implementation of the Traveling Salesman Problem (TSP) Hamiltonian.

The TSP seeks to find the shortest route that visits each city exactly once
and returns to the starting city.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from .base import Hamiltonian
from ..utils.pauli_utils import create_z_term, create_zz_term

def create_tsp_hamiltonian(
    distances: np.ndarray, 
    A: float = None, 
    B: float = None,
    time_dependent: bool = False
) -> Hamiltonian:
    """
    Create a Hamiltonian for the Traveling Salesman Problem.
    
    Following the formulation from Lucas (2014), we use binary variables x_i,p
    where x_i,p = 1 if city i is visited at position p in the route.
    
    The Hamiltonian consists of several penalty terms:
    1. Each city must be visited exactly once
    2. Each position in the route must have exactly one city
    3. The total distance of the route is minimized
    
    Args:
        distances: Square matrix of distances between cities
        A: Coefficient for the constraint terms (default: automatically calculated)
        B: Coefficient for the distance term (default: 1.0)
        time_dependent: Whether to create a time-dependent Hamiltonian
        
    Returns:
        Hamiltonian for the TSP
    """
    n_cities = distances.shape[0]
    
    # Validate input
    if distances.shape[0] != distances.shape[1]:
        raise ValueError("Distance matrix must be square")
    
    # Set default values for coefficients
    if A is None:
        # Set A large enough to enforce constraints
        A = 2.0 * np.max(distances) * n_cities
    
    if B is None:
        B = 1.0
    
    # Create the Hamiltonian
    hamiltonian = Hamiltonian(n_cities * n_cities)
    
    # Add metadata
    hamiltonian.metadata["problem"] = "TSP"
    hamiltonian.metadata["n_cities"] = n_cities
    hamiltonian.metadata["distances"] = distances.tolist()
    
    # Constraint 1: Each city must be visited exactly once
    # H_A = A * sum_i(1 - sum_p x_i,p)^2
    for i in range(n_cities):
        # Add constant term from expanding (1 - sum_p x_i,p)^2
        hamiltonian.add_constant(A)
        
        # Add linear terms: -2A * sum_p x_i,p
        for p in range(n_cities):
            qubit_idx = i * n_cities + p
            coef, term = create_z_term(qubit_idx, -A)
            hamiltonian.add_term(coef, term)
        
        # Add quadratic terms: A * sum_p sum_q x_i,p x_i,q
        for p in range(n_cities):
            for q in range(p+1, n_cities):
                qubit_idx_p = i * n_cities + p
                qubit_idx_q = i * n_cities + q
                coef, term = create_zz_term(qubit_idx_p, qubit_idx_q, A/2)
                hamiltonian.add_term(coef, term)
    
    # Constraint 2: Each position must have exactly one city
    # H_B = A * sum_p(1 - sum_i x_i,p)^2
    for p in range(n_cities):
        # Add constant term from expanding (1 - sum_i x_i,p)^2
        hamiltonian.add_constant(A)
        
        # Add linear terms: -2A * sum_i x_i,p
        for i in range(n_cities):
            qubit_idx = i * n_cities + p
            coef, term = create_z_term(qubit_idx, -A)
            hamiltonian.add_term(coef, term)
        
        # Add quadratic terms: A * sum_i sum_j x_i,p x_j,p
        for i in range(n_cities):
            for j in range(i+1, n_cities):
                qubit_idx_i = i * n_cities + p
                qubit_idx_j = j * n_cities + p
                coef, term = create_zz_term(qubit_idx_i, qubit_idx_j, A/2)
                hamiltonian.add_term(coef, term)
    
    # Objective: Minimize the total distance
    # H_C = B * sum_i,j,p d_i,j * x_i,p * x_j,(p+1)
    if time_dependent:
        # For time-dependent distances, we'll use parameter functions
        # Define a function that adjusts distances based on time
        def time_modifier(base_coef, params):
            # Extract the time parameter (expected to be between 0 and 24)
            time = params.get('time', 12.0)  # Default to noon if not specified
            
            # Define how traffic changes throughout the day (sample model)
            # 1. Rush hours: 7-9 AM, 4-6 PM (increase distances by up to 50%)
            # 2. Night hours: 11 PM - 5 AM (decrease distances by up to 30%)
            # 3. Otherwise: normal
            
            if 7 <= time < 9:  # Morning rush hour
                factor = 1.0 + 0.5 * (1.0 - abs(time - 8) / 1.0)  # Peak at 8 AM
            elif 16 <= time < 18:  # Evening rush hour 
                factor = 1.0 + 0.5 * (1.0 - abs(time - 17) / 1.0)  # Peak at 5 PM
            elif time >= 23 or time < 5:  # Night hours
                if time >= 23:
                    night_time = time - 23
                else:
                    night_time = time + 1
                factor = 0.7 + 0.3 * (night_time / 6.0)  # Gradually increase from 11 PM to 5 AM
            else:
                # Normal hours: slight variations
                hour_factor = np.sin(time * np.pi / 12.0) * 0.1
                factor = 1.0 + hour_factor
            
            return base_coef * factor
        
        for i in range(n_cities):
            for j in range(n_cities):
                if i != j:
                    for p in range(n_cities):
                        # Consider the route wrapping around
                        p_next = (p + 1) % n_cities
                        
                        qubit_idx_i_p = i * n_cities + p
                        qubit_idx_j_p_next = j * n_cities + p_next
                        
                        # Use the base distance coefficient
                        base_coef = B * distances[i, j] / 2
                        
                        # Use a parametric term that depends on time
                        hamiltonian.add_parametric_term(
                            base_coef, 
                            f"Z{qubit_idx_i_p} Z{qubit_idx_j_p_next}", 
                            'time',
                            time_modifier
                        )
    else:
        # Standard TSP with fixed distances
        for i in range(n_cities):
            for j in range(n_cities):
                if i != j:
                    for p in range(n_cities):
                        # Consider the route wrapping around
                        p_next = (p + 1) % n_cities
                        
                        qubit_idx_i_p = i * n_cities + p
                        qubit_idx_j_p_next = j * n_cities + p_next
                        
                        coef, term = create_zz_term(qubit_idx_i_p, qubit_idx_j_p_next, B * distances[i, j] / 2)
                        hamiltonian.add_term(coef, term)
    
    return hamiltonian

def get_tsp_solution(
    bit_string: Union[str, List[int]], 
    n_cities: int,
    distances: np.ndarray,
    time_factor: float = 1.0  # Factor to adjust distances based on time
) -> Dict:
    """
    Get the TSP solution from a bit string.
    
    Args:
        bit_string: Bit string representation of the solution
        n_cities: Number of cities
        distances: Matrix of distances between cities
        time_factor: Factor to adjust distances based on time of day (1.0 = no change)
        
    Returns:
        Dictionary with solution information
    """
    if isinstance(bit_string, str):
        bits = [int(b) for b in bit_string]
    else:
        bits = bit_string
    
    # Reshape to a matrix where rows are cities and columns are positions
    # The assignment is a binary matrix x_i,p where x_i,p = 1 if city i is at position p
    assignment_matrix = np.zeros((n_cities, n_cities), dtype=int)
    for i in range(n_cities):
        for p in range(n_cities):
            if i * n_cities + p < len(bits):
                assignment_matrix[i, p] = bits[i * n_cities + p]
    
    # Check if the assignment is valid
    valid = True
    
    # Each city must be visited exactly once
    city_visits = np.sum(assignment_matrix, axis=1)
    if not np.all(city_visits == 1):
        valid = False
    
    # Each position must have exactly one city
    position_assignments = np.sum(assignment_matrix, axis=0)
    if not np.all(position_assignments == 1):
        valid = False
    
    # Extract the tour
    tour = []
    for p in range(n_cities):
        cities_at_p = np.where(assignment_matrix[:, p] == 1)[0]
        if len(cities_at_p) == 1:
            tour.append(int(cities_at_p[0]))
        else:
            valid = False
            # For invalid solutions, still try to extract a tour for analysis
            if p < len(tour):
                tour.append(p)
    
    # Calculate the total distance
    total_distance = 0
    if valid:
        for i in range(n_cities):
            # Consider the route wrapping around
            from_city = tour[i]
            to_city = tour[(i + 1) % n_cities]
            
            # Apply time-dependent factor to the distance
            total_distance += distances[from_city, to_city] * time_factor
    
    return {
        "tour": tour,
        "total_distance": total_distance,
        "valid": valid
    } 