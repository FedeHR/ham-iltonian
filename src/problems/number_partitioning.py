"""
Number Partitioning Problem implementation.

The Number Partitioning Problem seeks to divide a set of numbers into two subsets
such that the difference between the sums of the two subsets is minimized.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any

from .base import Problem
from ..hamiltonians.base import Hamiltonian

class NumberPartitioningProblem(Problem):
    """
    Number Partitioning Problem representation.
    
    The Number Partitioning Problem seeks to partition a set of numbers into two subsets
    such that the difference between the sums of the subsets is minimized.
    """
    
    def __init__(self, 
                 numbers: List[float], 
                 name: str = "Number Partitioning"):
        """
        Initialize a Number Partitioning problem.
        
        Args:
            numbers: List of numbers to partition
            name: Name of the problem instance
        """
        super().__init__(name)
        self.numbers = np.array(numbers)
        self.n_numbers = len(numbers)
        self.total_sum = sum(numbers)
        
        # Store problem metadata
        self.metadata["n_numbers"] = self.n_numbers
        self.metadata["total_sum"] = self.total_sum
        self.metadata["min_value"] = float(min(numbers))
        self.metadata["max_value"] = float(max(numbers))
        self.metadata["mean_value"] = float(np.mean(numbers))
    
    def create_hamiltonian(self) -> Hamiltonian:
        """
        Create the Hamiltonian for this Number Partitioning problem.
        
        The objective is to minimize (sum(x) - sum(1-x))^2, which is equivalent to 
        minimizing (2*sum(x) - total_sum)^2.
        
        Returns:
            Hamiltonian for the Number Partitioning problem
        """
        from ..hamiltonians.number_partitioning import create_number_partitioning_hamiltonian
        self._hamiltonian = create_number_partitioning_hamiltonian(self.numbers)
        return self._hamiltonian
    
    def solve_classically(self) -> Dict[str, Any]:
        """
        Solve the Number Partitioning problem using classical methods.
        
        Returns:
            Dictionary with solution details
        """
        from ..utils.classical_solvers import solve_number_partitioning_brute_force
        solution = solve_number_partitioning_brute_force(self.numbers)
        self.add_solution("classical", solution)
        return solution
    
    def get_solution_from_bitstring(self, bitstring: str) -> Dict[str, Any]:
        """
        Get the Number Partitioning solution from a bitstring.
        
        Args:
            bitstring: Binary string representation of the solution
            
        Returns:
            Dictionary with solution details
        """
        from ..hamiltonians.number_partitioning import get_number_partitioning_solution
        solution = get_number_partitioning_solution(bitstring, self.numbers)
        return solution
    
    def calculate_quality(self, solution: Dict[str, Any]) -> float:
        """
        Calculate the quality of a Number Partitioning solution.
        
        For Number Partitioning, the quality is the negative of the difference between the sums,
        as we want to minimize this difference.
        
        Args:
            solution: Solution dictionary
            
        Returns:
            Quality metric (higher is better, so negative of the difference)
        """
        return -solution["difference"]
    
    def visualize_solution(self, solution: Dict[str, Any], filename: Optional[str] = None) -> None:
        """
        Visualize a Number Partitioning solution.
        
        Args:
            solution: Solution dictionary
            filename: Optional filename to save the visualization
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Sort the numbers for better visualization
        numbers_sorted = sorted(enumerate(self.numbers), key=lambda x: x[1], reverse=True)
        indices = [i for i, _ in numbers_sorted]
        values = [v for _, v in numbers_sorted]
        
        # Create sets for subset A and B
        subset_a_indices = [i for i, bit in enumerate(solution["bitstring"]) if bit == "1"]
        subset_b_indices = [i for i, bit in enumerate(solution["bitstring"]) if bit == "0"]
        
        subset_a_mask = [i in subset_a_indices for i in range(self.n_numbers)]
        subset_b_mask = [i in subset_b_indices for i in range(self.n_numbers)]
        
        # Bar chart showing the numbers and their assignment
        bar_colors = ['lightblue' if i in subset_a_indices else 'lightgreen' for i in range(self.n_numbers)]
        x_pos = np.arange(self.n_numbers)
        ax1.bar(x_pos, self.numbers, color=bar_colors)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f"{i}" for i in range(self.n_numbers)])
        ax1.set_xlabel('Number Index')
        ax1.set_ylabel('Value')
        ax1.set_title(f'Number Partitioning - {"Valid" if solution["valid"] else "INVALID"} Solution')
        
        # Legend
        ax1.legend([
            plt.Rectangle((0, 0), 1, 1, color='lightblue'),
            plt.Rectangle((0, 0), 1, 1, color='lightgreen')
        ], ['Subset A', 'Subset B'])
        
        # Pie chart showing the subset sums
        ax2.pie([solution["sum_a"], solution["sum_b"]], 
                labels=[f'Subset A: {solution["sum_a"]:.2f}', f'Subset B: {solution["sum_b"]:.2f}'],
                colors=['lightblue', 'lightgreen'], 
                autopct='%1.1f%%', 
                startangle=90,
                explode=(0.1, 0) if solution["sum_a"] > solution["sum_b"] else (0, 0.1))
        ax2.set_title(f'Subset Sums - Difference: {solution["difference"]:.2f}')
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename)
            print(f"Visualization saved to {filename}")
            
        plt.close()
    
    def __str__(self) -> str:
        """
        Return a string representation of the Number Partitioning problem.
        
        Returns:
            String description of the problem
        """
        numbers_str = ", ".join([f"{n:.1f}" for n in self.numbers])
        return f"{self.name} Problem with {self.n_numbers} numbers\n" \
               f"Numbers: [{numbers_str}]\n" \
               f"Total Sum: {self.total_sum:.1f}\n" \
               f"Value Range: [{self.metadata['min_value']:.1f}, {self.metadata['max_value']:.1f}]" 