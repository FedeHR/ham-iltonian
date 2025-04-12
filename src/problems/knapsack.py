"""
Knapsack Problem implementation.
"""
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any
import numpy as np

from problems.base import Problem
from hamiltonian import Hamiltonian
from utils.pauli_utils import create_z_term, create_zz_term
from utils.classical_solvers import solve_knapsack_brute_force
from parameter_modifiers.knapsack import get_modifiers

class KnapsackProblem(Problem):
    """
    Knapsack Problem representation.
    
    The Knapsack Problem seeks to maximize the value of items in a knapsack without 
    exceeding the capacity constraint.
    """
    
    def __init__(self, 
                 values: List[float], 
                 weights: List[float], 
                 capacity: float, 
                 name: str = "Knapsack"):
        """
        Initialize a Knapsack problem.
        
        Args:
            values: List of values for each item
            weights: List of weights for each item
            capacity: Maximum capacity of the knapsack
            name: Name of the problem instance
        """
        super().__init__(name)
        self.values = np.array(values, dtype=float).copy()
        self.weights = np.array(weights, dtype=float).copy()
        self.capacity = float(capacity)
        self.penalty = sum(self.values) + 1.0  # Default penalty
        
        assert len(values) == len(weights), "Values and weights must have the same length"
        
        # Store problem-specific metadata
        self.metadata["problem"] = "Knapsack"
        self.metadata["n_items"] = len(values)
        self.metadata["total_value"] = sum(values)
        self.metadata["total_weight"] = sum(weights)
        self.metadata["capacity"] = capacity
        
        # Original values to keep track of modifications
        self.original_values = self.values.copy()
        self.original_weights = self.weights.copy()
        self.original_capacity = self.capacity
        self.original_penalty = self.penalty
        
        # Register Knapsack-specific modifiers
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
        
        # Apply the modifier based on which parameter it affects
        if modifier_name == "scale_values" or modifier_name == "linear_value":
            # Apply to all values
            self.values = np.array([modifier_func(val, *args) for val in self.values])
            
        elif modifier_name == "scale_weights" or modifier_name == "linear_weight":
            # Apply to all weights
            self.weights = np.array([modifier_func(weight, *args) for weight in self.weights])
            
        elif modifier_name == "scale_capacity":
            # Apply to capacity
            self.capacity = modifier_func(self.capacity, *args)
            
        elif modifier_name == "adjust_penalty":
            # Apply to penalty
            self.penalty = modifier_func(self.penalty, *args)
            
        # Update metadata
        self.metadata["total_value"] = sum(self.values)
        self.metadata["total_weight"] = sum(self.weights)
        self.metadata["capacity"] = self.capacity
    
    def build_hamiltonian(self) -> None:
        """
        Build the Knapsack Hamiltonian based on Lucas (2014) formulation.
        
        The Hamiltonian is constructed to:
        1. Maximize the value of selected items
        2. Enforce the capacity constraint using slack variables
        
        Lucas's exact formulation uses:
        - A term to maximize total value: -∑_i v_i x_i
        - A penalty term to enforce: ∑_i w_i x_i - C = ∑_j 2^j y_j
          where y_j are binary slack variables representing excess weight
        
        Written as a minimization problem:
        H = -∑_i v_i x_i + A(∑_i w_i x_i - C - ∑_j 2^j y_j)^2
        """
        n_items = len(self.values)
        
        # Create or clear the Hamiltonian
        if self.hamiltonian is None:
            self.hamiltonian = Hamiltonian(n_items)
        else:
            self.hamiltonian.clear()
            self.hamiltonian.num_qubits = n_items  # Reset to initial value
        
        # Calculate the max possible excess weight
        max_excess = sum(self.weights) - self.capacity
        if max_excess <= 0:
            # No need for slack variables, the capacity is sufficient for all items
            # Only need to maximize value
            for i in range(n_items):
                # Objective: minimize negative values to maximize values
                coeff, term = create_z_term(i, -self.values[i] / 2)  # Divide by 2 for Z operator convention
                self.hamiltonian.add_term(coeff, term)
                
            # Add constant to make objective function match actual value
            self.hamiltonian.add_constant(sum(self.values) / 2)
            
            return
        
        # Determine number of slack variables needed (K is number of bits needed to represent max_excess)
        K = int(np.ceil(np.log2(max_excess + 1)))
        n_slack = K
        
        # Total qubits = n_items (original variables) + n_slack (slack variables)
        n_qubits = n_items + n_slack
        self.hamiltonian.num_qubits = n_qubits
        
        # Add the objective term: -∑_i v_i x_i
        for i in range(n_items):
            # Objective: minimize negative values to maximize values
            coeff, term = create_z_term(i, -self.values[i] / 2)  # Divide by 2 for Z operator convention
            self.hamiltonian.add_term(coeff, term)
            
        # Add constant to make objective function match actual value
        self.hamiltonian.add_constant(sum(self.values) / 2)
        
        # Add the constraint term using the slack variables
        # We need to enforce: ∑_i w_i x_i - C = ∑_j 2^j y_j
        # This is done by adding a penalty: A(∑_i w_i x_i - C - ∑_j 2^j y_j)^2 where A is a large coefficient
        
        # Expanding (∑_i w_i x_i - C - ∑_j 2^j y_j)^2:
        # = (∑_i w_i x_i)^2 - 2C(∑_i w_i x_i) - 2(∑_i w_i x_i)(∑_j 2^j y_j) + C^2 + 2C(∑_j 2^j y_j) + (∑_j 2^j y_j)^2
        
        # Term 1: (∑_i w_i x_i)^2 = ∑_i w_i^2 x_i^2 + ∑_i≠j w_i w_j x_i x_j
        # Since x_i is binary, x_i^2 = x_i
        for i in range(n_items):
            coeff, term = create_z_term(i, self.penalty * self.weights[i]**2 / 4)  # Divide by 4 for Z operator convention
            self.hamiltonian.add_term(coeff, term)
        
        # Cross-terms of (∑_i w_i x_i)^2
        for i in range(n_items):
            for j in range(i+1, n_items):
                coeff, term = create_zz_term(i, j, self.penalty * self.weights[i] * self.weights[j] / 4)  # Divide by 4 for ZZ operator convention
                self.hamiltonian.add_term(coeff, term)
        
        # Term 2: -2C(∑_i w_i x_i)
        for i in range(n_items):
            coeff, term = create_z_term(i, -self.penalty * self.capacity * self.weights[i] / 2)  # Divide by 2 for Z operator convention
            self.hamiltonian.add_term(coeff, term)
        
        # Term 3: -2(∑_i w_i x_i)(∑_j 2^j y_j)
        for i in range(n_items):
            for j in range(n_slack):
                # y_j is at index n_items + j
                slack_idx = n_items + j
                coeff, term = create_zz_term(i, slack_idx, -self.penalty * self.weights[i] * (2**j) / 4)  # Divide by 4 for ZZ operator convention
                self.hamiltonian.add_term(coeff, term)
        
        # Term 4: C^2 (constant)
        self.hamiltonian.add_constant(self.penalty * self.capacity**2 / 4)
        
        # Term 5: 2C(∑_j 2^j y_j)
        for j in range(n_slack):
            slack_idx = n_items + j
            coeff, term = create_z_term(slack_idx, self.penalty * self.capacity * (2**j) / 2)  # Divide by 2 for Z operator convention
            self.hamiltonian.add_term(coeff, term)
        
        # Term 6: (∑_j 2^j y_j)^2 = ∑_j (2^j)^2 y_j^2 + ∑_j≠k 2^j 2^k y_j y_k
        # Since y_j is binary, y_j^2 = y_j
        for j in range(n_slack):
            slack_idx = n_items + j
            coeff, term = create_z_term(slack_idx, self.penalty * (2**j)**2 / 4)  # Divide by 4 for Z operator convention
            self.hamiltonian.add_term(coeff, term)
        
        # Cross-terms of (∑_j 2^j y_j)^2
        for j in range(n_slack):
            for k in range(j+1, n_slack):
                slack_j = n_items + j
                slack_k = n_items + k
                coeff, term = create_zz_term(slack_j, slack_k, self.penalty * (2**j) * (2**k) / 4)  # Divide by 4 for ZZ operator convention
                self.hamiltonian.add_term(coeff, term)
    
    def solve_classically(self, **kwargs) -> Dict[str, Any]:
        """
        Solve the Knapsack problem using classical methods.
        
        Returns:
            Dictionary with solution details
        """
        solution = solve_knapsack_brute_force(self.values, self.weights, self.capacity)
        self.solutions["classical"] = solution
        return solution
    
    def evaluate_bitstring(self, bitstring: str) -> Dict[str, Any]:
        """
        Get the Knapsack solution from a bitstring.
        
        Args:
            bitstring: Binary string representation of the solution
            
        Returns:
            Dictionary with solution details
        """
        n_items = len(self.values)
        
        if isinstance(bitstring, str):
            assignment = [int(bit) for bit in bitstring]
        else:
            assignment = bitstring
            
        # Keep only the item selection part of the bitstring, discarding any slack variables
        if len(assignment) > n_items:
            assignment = assignment[:n_items]
            
        # Determine which items are included
        included_items = [i for i, bit in enumerate(assignment) if bit == 1]
        
        # Calculate total value and weight
        total_value = sum(self.values[i] for i in included_items)
        total_weight = sum(self.weights[i] for i in included_items)
        
        # Check if solution is valid (weight does not exceed capacity)
        valid = total_weight <= self.capacity
        
        return {
            "bitstring": bitstring,
            "assignment": assignment,
            "included_items": included_items,
            "total_value": total_value,
            "total_weight": total_weight,
            "capacity": self.capacity,
            "valid": valid,
            "quality": total_value if valid else 0,  # Invalid solutions get 0 quality
        }
    
    def calculate_quality(self, solution: Dict[str, Any]) -> float:
        """
        Calculate the quality of a Knapsack solution.
        
        Args:
            solution: Solution dictionary
            
        Returns:
            Quality metric (higher is better)
        """
        if solution["valid"]:
            return solution["total_value"]
        else:
            return 0  # Invalid solutions have zero quality
    
    def reset_parameters(self):
        """
        Reset all parameters to their original values.
        """
        self.values = self.original_values.copy()
        self.weights = self.original_weights.copy()
        self.capacity = self.original_capacity
        self.penalty = self.original_penalty
        
        # Update metadata
        self.metadata["total_value"] = sum(self.values)
        self.metadata["total_weight"] = sum(self.weights)
        self.metadata["capacity"] = self.capacity
        
        # Rebuild the Hamiltonian
        self.build_hamiltonian()
    
    def visualize_solution(self, solution: Dict[str, Any], filename: Optional[str] = None) -> None:
        """
        Visualize a Knapsack solution.
        
        Args:
            solution: Solution dictionary
            filename: Optional filename to save the visualization
        """
        if not solution:
            raise ValueError("No solution provided")
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract data for visualization
        n_items = len(self.values)
        included = [1 if i in solution["included_items"] else 0 for i in range(n_items)]
        
        # Sort items by value to weight ratio for better visualization
        value_weight_ratio = self.values / np.maximum(self.weights, 1e-10)
        sort_indices = np.argsort(value_weight_ratio)[::-1]  # Sort in descending order
        
        sorted_values = self.values[sort_indices]
        sorted_weights = self.weights[sort_indices]
        sorted_included = [included[i] for i in sort_indices]
        
        # Subplot 1: Item values and inclusion status
        x = np.arange(n_items)
        width = 0.35
        
        # Plot values
        ax1.bar(x - width/2, sorted_values, width, label='Value', color='lightblue')
        
        # Overlay showing which items are included
        inclusion_color = ['green' if inc else 'lightgray' for inc in sorted_included]
        ax1.bar(x + width/2, sorted_weights, width, label='Weight', color=inclusion_color, alpha=0.7)
        
        ax1.set_xlabel('Item (sorted by value/weight ratio)')
        ax1.set_ylabel('Value / Weight')
        ax1.set_title('Item Selection')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"{i}" for i in sort_indices])
        ax1.legend()
        
        # Add value/weight ratio as text
        for i, idx in enumerate(sort_indices):
            ax1.text(i, sorted_values[i] + 0.1, f"{value_weight_ratio[idx]:.2f}", 
                     ha='center', va='bottom', fontsize=8, rotation=90)
        
        # Subplot 2: Solution summary
        ax2.axis('off')
        solution_text = f"Solution Summary\n\n"
        solution_text += f"Total Value: {solution['total_value']:.2f}\n"
        solution_text += f"Total Weight: {solution['total_weight']:.2f}\n"
        solution_text += f"Capacity: {self.capacity:.2f}\n"
        solution_text += f"Weight Utilization: {solution['total_weight']/self.capacity*100:.1f}%\n\n"
        solution_text += f"Items Included: {len(solution['included_items'])}/{n_items}\n"
        solution_text += f"Valid Solution: {'Yes' if solution['valid'] else 'NO - EXCEEDS CAPACITY'}"
        
        ax2.text(0.5, 0.5, solution_text, ha='center', va='center', fontsize=12)
        
        # Draw a capacity indicator bar at the bottom
        capacity_bar = plt.axes([0.55, 0.15, 0.3, 0.05])
        capacity_bar.set_xlim(0, self.metadata["total_weight"])
        capacity_bar.set_ylim(0, 1)
        capacity_bar.axvspan(0, self.capacity, color='lightgreen', alpha=0.5)
        capacity_bar.axvspan(self.capacity, self.metadata["total_weight"], color='lightcoral', alpha=0.5)
        capacity_bar.axvline(solution["total_weight"], color='blue', linestyle='--', linewidth=2)
        capacity_bar.set_title('Capacity Usage', fontsize=10)
        capacity_bar.set_yticks([])
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename)
            print(f"Visualization saved to {filename}")
        else:
            plt.savefig("temp/knapsack_solution.png")
            print("Visualization saved to temp/knapsack_solution.png")
        plt.show()
    
    def __str__(self) -> str:
        """
        Return a string representation of the Knapsack problem.
        
        Returns:
            String description of the problem
        """
        items_str = []
        for i in range(len(self.values)):
            items_str.append(f"Item {i}: (value={self.values[i]:.1f}, weight={self.weights[i]:.1f})")
        
        return f"{self.name} Problem with {len(self.values)} items\n" \
               f"Capacity: {self.capacity:.1f}\n" \
               f"Total Value: {sum(self.values):.1f}\n" \
               f"Total Weight: {sum(self.weights):.1f}\n" \
               f"Items:\n  " + "\n  ".join(items_str) 