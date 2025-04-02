"""
Knapsack Problem implementation.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any

from .base import Problem
from ..hamiltonians.knapsack import create_knapsack_hamiltonian, get_knapsack_solution
from ..utils.classical_solvers import solve_knapsack_brute_force

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
        self.values = values
        self.weights = weights
        self.capacity = capacity
        
        assert len(values) == len(weights), "Values and weights must have the same length"
        
        self.metadata["n_items"] = len(values)
        self.metadata["total_value"] = sum(values)
        self.metadata["total_weight"] = sum(weights)
        self.metadata["capacity"] = capacity
    
    def create_hamiltonian(self, penalty: Optional[float] = None) -> Any:
        """
        Create the Hamiltonian for this Knapsack problem.
        
        Args:
            penalty: Penalty coefficient for exceeding the capacity
            
        Returns:
            Hamiltonian for the Knapsack problem
        """
        self._hamiltonian = create_knapsack_hamiltonian(
            self.values, self.weights, self.capacity, penalty)
        return self._hamiltonian
    
    def solve_classically(self, **kwargs) -> Dict[str, Any]:
        """
        Solve the Knapsack problem using classical methods.
        
        Returns:
            Dictionary with solution details
        """
        solution = solve_knapsack_brute_force(self.values, self.weights, self.capacity)
        self.add_solution("classical", solution)
        return solution
    
    def get_solution_from_bitstring(self, bitstring: str) -> Dict[str, Any]:
        """
        Get the Knapsack solution from a bitstring.
        
        Args:
            bitstring: Binary string representation of the solution
            
        Returns:
            Dictionary with solution details
        """
        return get_knapsack_solution(bitstring, self.values, self.weights, self.capacity)
    
    def calculate_quality(self, solution: Dict[str, Any]) -> float:
        """
        Calculate the quality of a Knapsack solution.
        
        For Knapsack, the quality is the total value if the solution is valid,
        otherwise it's a large negative number.
        
        Args:
            solution: Solution dictionary
            
        Returns:
            Total value if valid, large negative number otherwise
        """
        if solution["valid"]:
            return solution["total_value"]
        else:
            # Return a large negative number for invalid solutions
            return -float('inf')
    
    def visualize_solution(self, solution: Dict[str, Any], filename: Optional[str] = None) -> None:
        """
        Visualize a Knapsack solution.
        
        Args:
            solution: Solution dictionary
            filename: Optional filename to save the visualization
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Bar chart of all items
        item_names = [f"Item {i}" for i in range(len(self.values))]
        bar_colors = ['lightgray'] * len(self.values)
        for idx in solution['selected_items']:
            bar_colors[idx] = 'lightgreen'
        
        ax1.bar(item_names, self.values, color=bar_colors)
        ax1.set_ylabel('Value')
        ax1.set_title('Item Values (selected in green)')
        
        ax2.bar(item_names, self.weights, color=bar_colors)
        ax2.set_ylabel('Weight')
        ax2.set_title('Item Weights (selected in green)')
        
        # Add a line for the capacity
        ax2.axhline(y=self.capacity, color='r', linestyle='-', label=f'Capacity ({self.capacity})')
        ax2.legend()
        
        title = f"Knapsack Solution - Value: {solution['total_value']:.2f}, "
        title += f"Weight: {solution['total_weight']:.2f}/{self.capacity}"
        if not solution['valid']:
            title += " (INVALID!)"
        fig.suptitle(title)
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename)
            print(f"Visualization saved to {filename}")
        
        plt.show()
    
    def __str__(self) -> str:
        """
        Return a string representation of the Knapsack problem.
        
        Returns:
            String description of the problem
        """
        item_details = []
        for i, (v, w) in enumerate(zip(self.values, self.weights)):
            item_details.append(f"Item {i}: Value={v:.1f}, Weight={w:.1f}")
        
        items_str = "\n".join(item_details)
        return f"{self.name} Problem with {self.metadata['n_items']} items\n" \
               f"Capacity: {self.capacity}\n" \
               f"Total Value: {self.metadata['total_value']}\n" \
               f"Total Weight: {self.metadata['total_weight']}\n" \
               f"Items:\n{items_str}" 