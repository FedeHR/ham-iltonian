"""
Base class for combinatorial optimization problems.
"""
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from ..hamiltonians.base import Hamiltonian

class Problem(ABC):
    """
    Abstract base class for combinatorial optimization problems.
    
    This class provides a generic interface for representing, solving,
    and analyzing combinatorial optimization problems.
    """
    
    def __init__(self, name: str):
        """
        Initialize a problem instance.
        
        Args:
            name: Name of the problem instance
        """
        self.name = name
        self._hamiltonian = None
        self.solutions = {}
        self.metadata = {}
        self.parameters = {}
        
    @property
    def hamiltonian(self) -> Optional[Hamiltonian]:
        """
        Get the Hamiltonian representation of the problem.
        
        Returns:
            Hamiltonian, or None if not created yet
        """
        return self._hamiltonian
    
    @abstractmethod
    def create_hamiltonian(self, **kwargs) -> Hamiltonian:
        """
        Create the Hamiltonian for this problem.
        
        Args:
            **kwargs: Problem-specific parameters
            
        Returns:
            Hamiltonian representation of the problem
        """
        pass
    
    def create_parameterized_hamiltonian(self, param_values: Dict[str, Any] = None, **kwargs) -> Hamiltonian:
        """
        Create a parameterized Hamiltonian and evaluate it with specific parameter values.
        
        Args:
            param_values: Dictionary of parameter values to apply
            **kwargs: Problem-specific parameters
            
        Returns:
            Hamiltonian that's been evaluated with the given parameters
        """
        # Create the base Hamiltonian with parameters
        hamiltonian = self.create_hamiltonian(**kwargs)
        
        # If parameter values were provided, evaluate the Hamiltonian with them
        if param_values and hasattr(hamiltonian, 'evaluate_with_parameters'):
            hamiltonian = hamiltonian.evaluate_with_parameters(param_values)
            
        self._hamiltonian = hamiltonian
        return hamiltonian
    
    @abstractmethod
    def solve_classically(self, **kwargs) -> Dict[str, Any]:
        """
        Solve the problem using classical methods.
        
        Args:
            **kwargs: Problem-specific parameters
            
        Returns:
            Dictionary with solution details
        """
        pass
    
    def solve_with_parameters(self, param_values: Dict[str, Any], solution_name: str = None, **kwargs) -> Dict[str, Any]:
        """
        Solve the problem with specific parameter values.
        
        Args:
            param_values: Dictionary of parameter values to apply
            solution_name: Name to assign to the solution (default: based on param values)
            **kwargs: Problem-specific parameters
            
        Returns:
            Dictionary with solution details
        """
        # Store current parameter values
        self.parameters = param_values.copy()
        
        # Create the parameterized Hamiltonian
        self.create_parameterized_hamiltonian(param_values, **kwargs)
        
        # Solve the problem
        solution = self.solve_classically(**kwargs)
        
        # If solution_name wasn't provided, generate one from parameters
        if solution_name is None:
            param_str = "_".join([f"{k}_{v}" for k, v in param_values.items()])
            solution_name = f"param_{param_str}"
        
        # Add parameter values to the solution
        solution["parameters"] = param_values.copy()
        
        # Store the solution
        self.add_solution(solution_name, solution)
        
        return solution
    
    @abstractmethod
    def get_solution_from_bitstring(self, bitstring: str) -> Dict[str, Any]:
        """
        Get the solution from a bitstring.
        
        Args:
            bitstring: Binary string representation of the solution
            
        Returns:
            Dictionary with solution details
        """
        pass
    
    @abstractmethod
    def calculate_quality(self, solution: Dict[str, Any]) -> float:
        """
        Calculate the quality of a solution.
        
        Higher values indicate better solutions.
        
        Args:
            solution: Solution dictionary
            
        Returns:
            Quality metric (higher is better)
        """
        pass
    
    @abstractmethod
    def visualize_solution(self, solution: Dict[str, Any], filename: Optional[str] = None) -> None:
        """
        Visualize a solution.
        
        Args:
            solution: Solution dictionary
            filename: Optional filename to save the visualization
        """
        pass
    
    def visualize_parameter_effect(self, 
                                param_name: str, 
                                param_values: List[Any], 
                                metric_name: str = None,
                                filename: Optional[str] = None) -> None:
        """
        Visualize how a parameter affects solution quality.
        
        Args:
            param_name: Name of the parameter to vary
            param_values: List of values to try for the parameter
            metric_name: Name of the metric to plot (default: quality)
            filename: Optional filename to save the visualization
        """
        # Set default metric to quality if none provided
        if metric_name is None:
            metric_name = "quality"
            metric_func = self.calculate_quality
        else:
            # Try to find the metric in the solution dictionary
            def metric_func(solution):
                return solution.get(metric_name, float('nan'))
        
        # Solve the problem for each parameter value
        metrics = []
        used_values = []
        
        for value in param_values:
            # Create a parameter dictionary with this value
            params = {param_name: value}
            solution_name = f"{param_name}_{value}"
            
            # Check if we already have this solution
            existing_solution = self.get_solution(solution_name)
            if existing_solution:
                solution = existing_solution
            else:
                # Solve with this parameter value
                solution = self.solve_with_parameters(params, solution_name)
            
            # Extract the metric
            metric = metric_func(solution)
            
            if not np.isnan(metric):
                metrics.append(metric)
                used_values.append(value)
        
        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(used_values, metrics, 'o-', linewidth=2)
        plt.xlabel(param_name)
        plt.ylabel(metric_name)
        plt.title(f"Effect of {param_name} on {metric_name}")
        plt.grid(True)
        
        if filename:
            plt.savefig(filename)
            print(f"Visualization saved to {filename}")
        
        # Close the figure to avoid displaying in non-interactive environments
        plt.close()
    
    def add_solution(self, name: str, solution: Dict[str, Any]) -> None:
        """
        Add a solution to the problem's solution dictionary.
        
        Args:
            name: Name for the solution
            solution: Solution dictionary
        """
        self.solutions[name] = solution
    
    def get_solution(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a solution by name.
        
        Args:
            name: Name of the solution
            
        Returns:
            Solution dictionary, or None if not found
        """
        return self.solutions.get(name)
    
    def compare_solutions(self, names: List[str]) -> Dict[str, Any]:
        """
        Compare multiple solutions.
        
        Args:
            names: List of solution names to compare
            
        Returns:
            Dictionary mapping solution names to their relative performance
        """
        comparison = {}
        solutions = [self.get_solution(name) for name in names if self.get_solution(name)]
        if not solutions:
            return comparison
        
        # Find the best solution based on quality
        qualities = [self.calculate_quality(sol) for sol in solutions]
        best_idx = np.argmax(qualities)
        best_quality = qualities[best_idx]
        
        # Calculate relative performance
        for i, name in enumerate([n for n, s in zip(names, solutions) if s]):
            quality = qualities[i]
            comparison[name] = {
                'quality': quality,
                'ratio': quality / best_quality if best_quality != 0 else float('inf')
            }
        
        return comparison
    
    def print_info(self) -> None:
        """
        Print general information about the problem.
        This method prints the problem's string representation.
        """
        print(self)
    
    def print_hamiltonian(self, truncate: bool = False, max_length: int = 500) -> None:
        """
        Print the Hamiltonian for this problem.
        
        Args:
            truncate: Whether to truncate long Hamiltonians
            max_length: Maximum length to print if truncating
        """
        if self._hamiltonian is None:
            print("Hamiltonian has not been created yet. Call create_hamiltonian() first.")
            return
        
        print(f"\n{self.name} Hamiltonian:")
        hamiltonian_str = str(self._hamiltonian)
        
        if truncate and len(hamiltonian_str) > max_length:
            print(hamiltonian_str[:max_length] + "...")
        else:
            print(hamiltonian_str)
    
    def print_pennylane_hamiltonian(self) -> None:
        """
        Print the PennyLane Hamiltonian representation.
        """
        if self._hamiltonian is None:
            print("Hamiltonian has not been created yet. Call create_hamiltonian() first.")
            return
        
        print(f"\n{self.name} PennyLane Hamiltonian:")
        pennylane_ham = self._hamiltonian.to_pennylane()
        print(pennylane_ham)
    
    def print_solution(self, name: str = "classical") -> None:
        """
        Print a solution by name.
        
        Args:
            name: Name of the solution to print
        """
        solution = self.get_solution(name)
        if solution is None:
            print(f"No solution named '{name}' found.")
            return
        
        print(f"\n{self.name} Solution '{name}':")
        
        # Print parameters if they exist
        if "parameters" in solution:
            print("Parameters:")
            for param_name, param_value in solution["parameters"].items():
                print(f"  {param_name}: {param_value}")
            print()
        
        # Print quality
        quality = self.calculate_quality(solution)
        print(f"Quality: {quality:.4f}")
        
        # Print other solution details
        for key, value in solution.items():
            if key != "parameters":  # Already printed parameters
                print(f"{key}: {value}")
    
    def print_comparison(self, names: List[str] = None) -> None:
        """
        Print a comparison of multiple solutions.
        
        Args:
            names: List of solution names to compare (default: all solutions)
        """
        if names is None:
            names = list(self.solutions.keys())
        
        if not names:
            print("No solutions to compare.")
            return
        
        comparison = self.compare_solutions(names)
        if not comparison:
            print("No valid solutions found for comparison.")
            return
        
        print(f"\n{self.name} Solution Comparison:")
        print(f"{'Solution':<15} {'Quality':<10} {'Ratio':<10}")
        print("-" * 35)
        
        # Find the best solution based on ratio (should be 1.0)
        best_name = max(comparison.keys(), key=lambda n: comparison[n]['ratio'])
        
        for name in names:
            if name in comparison:
                quality = comparison[name]['quality']
                ratio = comparison[name]['ratio']
                star = "*" if name == best_name else " "
                print(f"{name:<15} {quality:<10.4f} {ratio:<10.4f} {star}")
    
    def test_bitstrings(self, bitstrings: List[str], name_prefix: str = "test") -> List[str]:
        """
        Test multiple bitstrings and add them as solutions.
        
        Args:
            bitstrings: List of bitstrings to test
            name_prefix: Prefix for solution names
            
        Returns:
            List of solution names that were added
        """
        solution_names = []
        
        for i, bitstring in enumerate(bitstrings):
            solution_name = f"{name_prefix}_{i}"
            solution = self.get_solution_from_bitstring(bitstring)
            
            # Add the bitstring to the solution for reference
            solution["bitstring"] = bitstring
            
            # Calculate the quality
            quality = self.calculate_quality(solution)
            solution["quality"] = quality
            
            # Add the solution
            self.add_solution(solution_name, solution)
            solution_names.append(solution_name)
            
            print(f"\nSolution {solution_name}:")
            print(f"Bitstring: {bitstring}")
            print(f"Quality: {quality:.4f}")
            
            # Print other relevant solution details
            for key, value in solution.items():
                if key not in ["bitstring", "quality"]:
                    print(f"{key}: {value}")
        
        return solution_names
    
    def __str__(self) -> str:
        """
        Return a string representation of the problem.
        
        Returns:
            String representation
        """
        metadata_str = ", ".join([f"{k}: {v}" for k, v in self.metadata.items()])
        return f"{self.name} Problem\nMetadata: {metadata_str}" 