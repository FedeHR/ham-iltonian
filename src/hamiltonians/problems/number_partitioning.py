"""
Number Partitioning Problem implementation.

The Number Partitioning Problem seeks to divide a set of numbers into two subsets
such that the difference between the sums of the two subsets is minimized.

***** TODO IN CONSTRUCTION / BETA
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any

from hamiltonians.problems.base import Problem
from hamiltonians.hamiltonian.hamiltonian import Hamiltonian
from hamiltonians.utils.pauli_utils import create_z_term, create_zz_term
from hamiltonians.utils.classical_solvers import solve_number_partitioning_brute_force
from hamiltonians.parameter_modifiers.number_partitioning import get_modifiers

class NumberPartitioningProblem(Problem):
    """
    Number Partitioning Problem representation.
    
    The Number Partitioning Problem seeks to partition a set of numbers into two subsets
    such that the difference between the sums of the subsets is minimized.
    """
    
    def __init__(self,
                 numbers: List[float],
                 problem_type: str = "Number Partitioning"):
        """
        Initialize a Number Partitioning problem.
        
        Args:
            numbers: List of numbers to partition
            problem_type: Name of the problem instance
        """
        super().__init__(problem_type)
        self.numbers = np.array(numbers)
        self.n_numbers = len(numbers)
        self.total_sum = sum(numbers)
        
        # Store problem metadata
        self.metadata["n_numbers"] = self.n_numbers
        self.metadata["total_sum"] = self.total_sum
        self.metadata["min_value"] = float(min(numbers))
        self.metadata["max_value"] = float(max(numbers))
        self.metadata["mean_value"] = float(np.mean(numbers))
        
        # Store original values for parameter resets
        self.original_numbers = self.numbers.copy()
        
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
        
        # Apply modifier to the numbers
        self.numbers = np.array([modifier_func(num, *args) for num in self.numbers])
        
        # Update problem metadata
        self.total_sum = sum(self.numbers)
        self.metadata["total_sum"] = self.total_sum
        self.metadata["min_value"] = float(min(self.numbers))
        self.metadata["max_value"] = float(max(self.numbers))
        self.metadata["mean_value"] = float(np.mean(self.numbers))
    
    def build_hamiltonian(self) -> None:
        """
        Build the Hamiltonian for this Number Partitioning problem following Lucas (2014) formulation.
        
        The objective is to minimize (S_a - S_b)^2, which is equivalent to 
        minimizing (2*S_a - S_total)^2.
        
        The exact formulation from Lucas is:
        H = (sum_i a_i s_i - sum_i a_i (1-s_i))^2
          = (2*sum_i a_i s_i - sum_i a_i)^2
          = 4(sum_i a_i s_i)^2 - 4(sum_i a_i)(sum_i a_i s_i) + (sum_i a_i)^2
        
        Where s_i ∈ {0,1} binary variables are converted to Ising spins σ_i ∈ {-1,1} with s_i = (1+σ_i)/2
        """
        # Create a new Hamiltonian or clear the existing one
        if self.hamiltonian is None:
            self.hamiltonian = Hamiltonian(self.n_numbers)
        else:
            self.hamiltonian.clear()
        
        total_sum = sum(self.numbers)
        
        # Constant term: (sum_i a_i)^2
        self.hamiltonian.add_constant(total_sum**2)
        
        for i in range(self.n_numbers):
            # Linear terms: 4*a_i^2 - 4*total_sum*a_i
            # For the Z operator convention, the coefficient is divided by 2
            coeff = 4 * (self.numbers[i]**2) - 4 * total_sum * self.numbers[i]
            coeff_z, term_z = create_z_term(i, coeff/2)  # Divide by 2 for Z operator convention
            self.hamiltonian.add_term(coeff_z, term_z)
            
        for i in range(self.n_numbers):
            for j in range(i+1, self.n_numbers):
                # Quadratic terms: 8*a_i*a_j
                # For the ZZ operator convention, the coefficient is divided by 4
                coeff = 8 * self.numbers[i] * self.numbers[j]
                coeff_zz, term_zz = create_zz_term(i, j, coeff/4)  # Divide by 4 for ZZ operator convention
                self.hamiltonian.add_term(coeff_zz, term_zz)
    
    def solve_classically(self, **kwargs) -> Dict[str, Any]:
        """
        Solve the Number Partitioning problem using classical methods.
        
        Returns:
            Dictionary with solution details
        """
        solution = solve_number_partitioning_brute_force(self.numbers)
        self.solutions["classical"] = solution
        return solution
    
    def evaluate_bitstring(self, bitstring: str) -> Dict[str, Any]:
        """
        Get the Number Partitioning solution from a bitstring.
        
        Args:
            bitstring: Binary string representation of the solution
            
        Returns:
            Dictionary with solution details
        """
        # Extract the binary decisions from the bitstring
        if len(bitstring) != len(self.numbers):
            raise ValueError(f"Bitstring length ({len(bitstring)}) does not match number of elements ({len(self.numbers)})")
        
        # Create the subsets
        subset_a = [self.numbers[i] for i, bit in enumerate(bitstring) if bit == "1"]
        subset_b = [self.numbers[i] for i, bit in enumerate(bitstring) if bit == "0"]
        
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
    
    def reset_parameters(self):
        """
        Reset all parameters to their original values.
        """
        self.numbers = self.original_numbers.copy()
        self.total_sum = sum(self.numbers)
        
        # Update metadata
        self.metadata["total_sum"] = self.total_sum
        self.metadata["min_value"] = float(min(self.numbers))
        self.metadata["max_value"] = float(max(self.numbers))
        self.metadata["mean_value"] = float(np.mean(self.numbers))
        
        # Rebuild the Hamiltonian
        self.build_hamiltonian()
    
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
        
        plt.show()
    
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