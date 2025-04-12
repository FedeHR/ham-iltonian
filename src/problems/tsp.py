"""
Traveling Salesman Problem (TSP) implementation.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any

from problems.base import Problem
from hamiltonian import Hamiltonian
from utils.pauli_utils import create_z_term, create_zz_term
from utils.classical_solvers import solve_tsp_brute_force
from parameter_modifiers.tsp import get_modifiers

class TSPProblem(Problem):
    """
    Traveling Salesman Problem representation.
    
    The TSP seeks to find the shortest possible route that visits each city exactly once 
    and returns to the starting city.
    """
    
    def __init__(
        self, 
        distances: Optional[np.ndarray] = None, 
        positions: Optional[np.ndarray] = None,
        city_names: Optional[List[str]] = None,
        distance_metric: str = "euclidean",
        name: str = "TSP"
    ):
        """
        Initialize a TSP problem.
        
        Args:
            distances: Distance matrix between cities (n x n). Optional if positions is provided.
            positions: Array of city positions (n x 2 or n x 3). Optional if distances is provided.
            city_names: Optional list of city names
            distance_metric: Metric to use when calculating distances from positions. Options: 'euclidean', 'manhattan'
            name: Name of the problem instance
            
        Raises:
            ValueError: If neither distances nor positions are provided
        """
        super().__init__(name)
        
        # Check that at least one of distances or positions is provided
        if distances is None and positions is None:
            raise ValueError("At least one of distances or positions must be provided")
        
        # If positions are provided but not distances, calculate distances
        if distances is None and positions is not None:
            n_cities = positions.shape[0]
            distances = np.zeros((n_cities, n_cities))
            
            # Calculate distances based on the specified metric
            for i in range(n_cities):
                for j in range(n_cities):
                    if i != j:
                        if distance_metric == 'euclidean':
                            # Euclidean distance
                            distances[i, j] = np.sqrt(np.sum((positions[i] - positions[j])**2))
                        elif distance_metric == 'manhattan':
                            # Manhattan distance
                            distances[i, j] = np.sum(np.abs(positions[i] - positions[j]))
                        else:
                            # Default to Euclidean
                            distances[i, j] = np.sqrt(np.sum((positions[i] - positions[j])**2))
        
        self.distances = distances
        self.n_cities = distances.shape[0]
        
        # Validate distance matrix
        if distances.shape[0] != distances.shape[1]:
            raise ValueError("Distance matrix must be square")
        assert np.all(np.diag(distances) == 0), "Diagonal of distance matrix should be 0"
        
        self.metadata["n_cities"] = self.n_cities
        self.metadata["distance_range"] = [float(np.min(distances)), float(np.max(distances))]
        self.metadata["avg_distance"] = float(np.mean(distances))
        
        # Set city names if not provided
        if city_names is None:
            self.city_names = [f"City {i}" for i in range(self.n_cities)]
        else:
            if len(city_names) != self.n_cities:
                raise ValueError("Number of city names must match number of cities")
            self.city_names = city_names
        
        # Set city positions for visualization
        if positions is not None:
            if positions.shape[0] != self.n_cities:
                raise ValueError("Number of positions must match number of cities")
            self.city_positions = positions
        else:
            # Generate random positions in 2D plane if not provided
            self.city_positions = np.random.rand(self.n_cities, 2)
            
        # Set default coefficients
        self.A = 2.0 * np.max(self.distances) * self.n_cities  # Constraint coefficient
        self.B = 1.0  # Distance coefficient
        
        # Store original distances for parameter resets
        self.original_distances = self.distances.copy()
        
        # Register modifiers
        default_modifiers = get_modifiers()
        self.modifier_functions.update(default_modifiers)
        
        # Build the initial Hamiltonian
        self.build_hamiltonian()
    
    def _apply_modifier(self, modifier_name: str, *args) -> None:
        """
        Apply the modifier to the problem parameters.
        
        Args:
            modifier_name: Name of the modifier function to apply
            *args: Parameters for the modifier function
        """
        modifier_func = self.modifier_functions[modifier_name]
        
        # Apply modifier to all distances
        for i in range(self.n_cities):
            for j in range(self.n_cities):
                if i != j:  # Skip diagonal elements
                    self.distances[i, j] = modifier_func(self.original_distances[i, j], *args)
        
        # Update metadata
        self.metadata["distance_range"] = [float(np.min(self.distances)), float(np.max(self.distances))]
        self.metadata["avg_distance"] = float(np.mean(self.distances))
        
        # Rebuild A coefficient based on new max distance
        self.A = 2.0 * np.max(self.distances) * self.n_cities
    
    def build_hamiltonian(self) -> None:
        """
        Build the Hamiltonian for this TSP problem following Lucas (2014) formulation.
        
        The exact formulation from Lucas uses:
        H_A = A * sum_i(1 - sum_p x_i,p)^2    # Each city must be visited exactly once
        H_B = A * sum_p(1 - sum_i x_i,p)^2    # Each position must have exactly one city
        H_C = B * sum_i,j,p d_i,j * x_i,p * x_j,(p+1)mod n    # Distance term
        
        H = H_A + H_B + H_C
        """
        n_cities = self.n_cities
        
        # Create a new Hamiltonian or clear the existing one
        if self.hamiltonian is None:
            self.hamiltonian = Hamiltonian(n_cities * n_cities)
        else:
            self.hamiltonian.clear()
        
        # Add metadata
        self.hamiltonian.metadata = {
            "problem": "TSP",
            "n_cities": n_cities,
            "distances": self.distances.tolist()
        }
        
        # Constraint 1 (H_A): Each city must be visited exactly once
        # H_A = A * sum_i(1 - sum_p x_i,p)^2
        for i in range(n_cities):
            # Add constant term from expanding (1 - sum_p x_i,p)^2
            self.hamiltonian.add_constant(self.A)
            
            # Add linear terms: -2A * sum_p x_i,p
            for p in range(n_cities):
                qubit_idx = i * n_cities + p
                coef, term = create_z_term(qubit_idx, -2 * self.A / 2)  # Divide by 2 for Z operator convention
                self.hamiltonian.add_term(coef, term)
            
            # Add quadratic terms: A * sum_p sum_q x_i,p x_i,q
            for p in range(n_cities):
                for q in range(p+1, n_cities):
                    qubit_idx_p = i * n_cities + p
                    qubit_idx_q = i * n_cities + q
                    coef, term = create_zz_term(qubit_idx_p, qubit_idx_q, self.A / 4)  # Divide by 4 for ZZ operator convention
                    self.hamiltonian.add_term(coef, term)
        
        # Constraint 2 (H_B): Each position must have exactly one city
        # H_B = A * sum_p(1 - sum_i x_i,p)^2
        for p in range(n_cities):
            # Add constant term from expanding (1 - sum_i x_i,p)^2
            self.hamiltonian.add_constant(self.A)
            
            # Add linear terms: -2A * sum_i x_i,p
            for i in range(n_cities):
                qubit_idx = i * n_cities + p
                coef, term = create_z_term(qubit_idx, -2 * self.A / 2)  # Divide by 2 for Z operator convention
                self.hamiltonian.add_term(coef, term)
            
            # Add quadratic terms: A * sum_i sum_j x_i,p x_j,p
            for i in range(n_cities):
                for j in range(i+1, n_cities):
                    qubit_idx_i = i * n_cities + p
                    qubit_idx_j = j * n_cities + p
                    coef, term = create_zz_term(qubit_idx_i, qubit_idx_j, self.A / 4)  # Divide by 4 for ZZ operator convention
                    self.hamiltonian.add_term(coef, term)
        
        # Objective (H_C): Minimize the total distance
        # H_C = B * sum_i,j,p d_i,j * x_i,p * x_j,(p+1)mod n
        for i in range(n_cities):
            for j in range(n_cities):
                if i != j:
                    for p in range(n_cities):
                        # Consider the route wrapping around
                        p_next = (p + 1) % n_cities
                        
                        qubit_idx_i_p = i * n_cities + p
                        qubit_idx_j_p_next = j * n_cities + p_next
                        
                        coef, term = create_zz_term(qubit_idx_i_p, qubit_idx_j_p_next, self.B * self.distances[i, j] / 4)  # Divide by 4 for ZZ operator convention
                        self.hamiltonian.add_term(coef, term)
    
    def solve_classically(self, **kwargs) -> Dict[str, Any]:
        """
        Solve the TSP problem using classical methods.
        
        Returns:
            Dictionary with solution details
        """
        # Solve using brute force
        solution = solve_tsp_brute_force(self.distances)
        self.solutions["classical"] = solution
        return solution
    
    def evaluate_bitstring(self, bitstring: str) -> Dict[str, Any]:
        """
        Get the TSP solution from a bitstring.
        
        Args:
            bitstring: Binary string representation of the solution
            
        Returns:
            Dictionary with solution details
        """
        if isinstance(bitstring, str):
            bits = [int(b) for b in bitstring]
        else:
            bits = bitstring
        
        # Reshape to a matrix where rows are cities and columns are positions
        # The assignment is a binary matrix x_i,p where x_i,p = 1 if city i is at position p
        assignment_matrix = np.zeros((self.n_cities, self.n_cities), dtype=int)
        for i in range(self.n_cities):
            for p in range(self.n_cities):
                if i * self.n_cities + p < len(bits):
                    assignment_matrix[i, p] = bits[i * self.n_cities + p]
        
        # Check if the assignment is valid
        valid = True
        route = []
        
        # Each city must be visited exactly once
        city_counts = assignment_matrix.sum(axis=1)
        if not np.all(city_counts == 1):
            valid = False
        
        # Each position must have exactly one city
        position_counts = assignment_matrix.sum(axis=0)
        if not np.all(position_counts == 1):
            valid = False
        
        # If valid, determine the route order
        if valid:
            for p in range(self.n_cities):
                # Find which city is at position p
                city_at_p = np.where(assignment_matrix[:, p] == 1)[0]
                if len(city_at_p) == 1:
                    route.append(int(city_at_p[0]))
                else:
                    valid = False
                    break
        
        # Calculate route distance
        total_distance = 0.0
        if valid and len(route) == self.n_cities:
            for i in range(self.n_cities):
                from_city = route[i]
                to_city = route[(i + 1) % self.n_cities]
                total_distance += self.distances[from_city, to_city]
        else:
            # Invalid solution gets a large distance
            valid = False
            total_distance = float('inf')
        
        return {
            "bitstring": bitstring,
            "route": route,
            "assignment_matrix": assignment_matrix.tolist(),
            "distance": total_distance,
            "valid": valid,
            "city_names": [self.city_names[i] for i in route] if valid else [],
            "quality": -total_distance if valid else -float('inf')
        }
    
    def calculate_quality(self, solution: Dict[str, Any]) -> float:
        """
        Calculate the quality of a TSP solution.
        
        For TSP, the quality is the negative of the route distance, as we want to minimize this.
        
        Args:
            solution: Solution dictionary
            
        Returns:
            Quality metric (higher is better, so negative of the distance)
        """
        if solution["valid"]:
            return -solution["distance"]
        else:
            return -float('inf')
    
    def reset_parameters(self):
        """
        Reset all parameters to their original values.
        """
        self.distances = self.original_distances.copy()
        
        # Reset metadata
        self.metadata["distance_range"] = [float(np.min(self.distances)), float(np.max(self.distances))]
        self.metadata["avg_distance"] = float(np.mean(self.distances))
        
        # Reset A coefficient
        self.A = 2.0 * np.max(self.distances) * self.n_cities
        
        # Rebuild the Hamiltonian
        self.build_hamiltonian()
    
    def visualize_solution(self, solution: Dict[str, Any], filename: Optional[str] = None) -> None:
        """
        Visualize a TSP solution.
        
        Args:
            solution: Solution dictionary
            filename: Optional filename to save the visualization
        """
        if not solution or not solution["valid"]:
            print("Invalid solution, cannot visualize.")
            return
        
        route = solution["route"]
        
        plt.figure(figsize=(10, 8))
        
        # Draw cities
        plt.scatter(self.city_positions[:, 0], self.city_positions[:, 1], c='blue', s=100, zorder=2)
        
        # Add city labels
        for i, (x, y) in enumerate(self.city_positions):
            plt.text(x, y + 0.02, self.city_names[i], ha='center', va='bottom', fontsize=10)
        
        # Draw route with arrows
        for i in range(len(route)):
            from_idx = route[i]
            to_idx = route[(i + 1) % len(route)]
            
            from_pos = self.city_positions[from_idx]
            to_pos = self.city_positions[to_idx]
            
            # Calculate arrow parameters (shorter than full distance to avoid overlap with nodes)
            dx = to_pos[0] - from_pos[0]
            dy = to_pos[1] - from_pos[1]
            
            # Normalize the direction vector
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                udx, udy = dx/length, dy/length
            else:
                udx, udy = 0, 0
                
            # Shrink the arrow slightly to avoid overlapping with nodes
            shrink = 0.1
            start_x = from_pos[0] + shrink * udx
            start_y = from_pos[1] + shrink * udy
            end_x = to_pos[0] - shrink * udx
            end_y = to_pos[1] - shrink * udy
            
            # Draw the arrow
            plt.arrow(start_x, start_y, end_x - start_x, end_y - start_y, 
                      head_width=0.03, head_length=0.05, fc='red', ec='red', zorder=1,
                      length_includes_head=True)
            
            # Add a small number showing the order
            mid_x = (from_pos[0] + to_pos[0]) / 2
            mid_y = (from_pos[1] + to_pos[1]) / 2
            # Offset the position number slightly to avoid overlap with the line
            offset_x = -udy * 0.03  # Perpendicular to the line
            offset_y = udx * 0.03   # Perpendicular to the line
            plt.text(mid_x + offset_x, mid_y + offset_y, str(i+1), 
                     ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
        
        # Draw a legend for the route order
        plt.plot([], [], 'r-', label=f'Route (Total: {solution["distance"]:.2f})')
        
        plt.title(f"TSP Solution - Total Distance: {solution['distance']:.2f}")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust plot limits to include all city positions with some padding
        plt.xlim([min(self.city_positions[:, 0]) - 0.1, max(self.city_positions[:, 0]) + 0.1])
        plt.ylim([min(self.city_positions[:, 1]) - 0.1, max(self.city_positions[:, 1]) + 0.1])
        
        if filename:
            plt.savefig(filename)
            print(f"Visualization saved to {filename}")
        
        plt.show()
    
    def __str__(self) -> str:
        """
        Return a string representation of the TSP problem.
        
        Returns:
            String description of the problem
        """
        cities_str = ", ".join(self.city_names)
        
        # Create a small representation of the distance matrix
        distances_str = "Distance Matrix:\n"
        # Show at most 5x5 of the distance matrix
        n_show = min(5, self.n_cities)
        for i in range(n_show):
            row = " ".join([f"{self.distances[i, j]:.1f}" for j in range(n_show)])
            distances_str += f"  {row}\n"
        
        if self.n_cities > n_show:
            distances_str += "  ... (truncated)\n"
        
        return f"{self.name} Problem with {self.n_cities} cities\n" \
               f"Cities: {cities_str}\n" \
               f"Average Distance: {self.metadata['avg_distance']:.2f}\n" \
               f"Distance Range: [{self.metadata['distance_range'][0]:.2f}, {self.metadata['distance_range'][1]:.2f}]\n" \
               f"{distances_str}" 