
from typing import Dict, List, Optional, Any, Callable
from abc import ABC, abstractmethod
import numpy as np

from hamiltonians.hamiltonian.hamiltonian import Hamiltonian


class Problem(ABC):
    """
    Abstract base class for combinatorial optimization problems.
    
    Provides a generic interface for representing, solving, and analyzing COPs.
    """
    
    def __init__(self, problem_type: str):
        """
        Initialize a problem instance.
        
        Args:
            problem_type: Name of the problem instance
        """
        self.name = problem_type
        self.hamiltonian = None
        self.solutions = {}
        self.metadata = {}
        
        # Dictionary of parameter modifier functions
        self.modifier_functions = {
            # Default modifier functions
            # TODO think about clever general / default modifier functions
            "linear":
                lambda value_to_be_modified, param: value_to_be_modified + param,
            "quadratic":
                lambda value_to_be_modified, param: value_to_be_modified + param ** 2,
            "qubic":
                lambda value_to_be_modified, param: value_to_be_modified + param ** 3,
            "exponential":
                lambda value_to_be_modified, param: value_to_be_modified + np.exp(param),
        }
    
    def add_modifier_function(self, function_name: str, 
                             function: Callable[[float, Any], float]) -> None:
        """
        Add a custom modifier function to modify problem parameters.
        
        Args:
            function_name: Name of the modifier
            function: Function that takes a problem parameter to be modified and optionally additional arguments
                      for the calculation, returns the modified problem parameter
        """
        self.modifier_functions[function_name] = function

    def modify_parameters(self, modifier_name: str, **kwargs) -> None:
        """
        Modify the problem parameters using the specified modifier function.
        
        Args:
            modifier_name: Name of the modifier function to apply
            **kwargs: Parameters for the modifier function
        """
        if modifier_name not in self.modifier_functions:
            raise ValueError(f"Unknown modifier '{modifier_name}'. "
                             f"Available modifiers: {list(self.modifier_functions.keys())} "
                             f"or define your own modifier function using add_modifier_function.")
        
        # Apply the parameter-specific modification logic
        self._apply_modifier(modifier_name, **kwargs)
        self.build_hamiltonian()
    
    @abstractmethod
    def _apply_modifier(self, modifier_name: str, **kwargs) -> None:
        """
        Apply the modifier to the problem parameters.
        
        Args:
            modifier_name: Name of the modifier function to apply
            *args: Parameters for the modifier function
        """
        pass
    
    @abstractmethod
    def build_hamiltonian(self) -> None:
        """
        Create the Hamiltonian for this problem.
        """
        pass
    
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

    @abstractmethod
    def evaluate_bitstring(self, bitstring: str) -> Dict[str, Any]:
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
    
    def get_solution(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a solution by name.
        
        Args:
            name: Name of the solution
            
        Returns:
            Solution dictionary, or None if not found
        """
        return self.solutions.get(name)

    def get_modifier_functions(self) -> Dict[str, Callable]:
        """
        Get the available modifier functions.

        Returns:
            Dictionary of modifier functions
        """
        return self.modifier_functions
    
    def print_modifier_functions(self, verbose: bool = False) -> None:
        """
        Print the available modifier functions in a readable format.
        
        Args:
            verbose: If True, attempts to print the function code as well
        """
        print(f"\nAvailable modifier functions for {self.name}:")
        for name, func in self.modifier_functions.items():
            print(f"  * {name}")
            if verbose:
                import inspect
                try:
                    source = inspect.getsource(func)
                    print(f"\n    Code: {source.strip()}")
                    print("----------------------------------------\n")
                except (TypeError, OSError):
                    print(f"    Code: <Built-in function, source not available>")
    
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
    
    def print_hamiltonian(self, truncate: bool = False, max_length: int = 500) -> None:
        """
        Print the Hamiltonian for this problem.
        
        Args:
            truncate: Whether to truncate long Hamiltonians
            max_length: Maximum length to print if truncating
        """
        if self.hamiltonian is None:
            print("Hamiltonian has not been created yet. Call build_hamiltonian() first.")
            return
        
        print(f"\n{self.name} Hamiltonian:")
        hamiltonian_str = str(self.hamiltonian)
        
        if truncate and len(hamiltonian_str) > max_length:
            print(hamiltonian_str[:max_length] + "...")
        else:
            print(hamiltonian_str)
    
    def print_pennylane_hamiltonian(self) -> None:
        """
        Print the PennyLane Hamiltonian representation.
        """
        if self.hamiltonian is None:
            print("Hamiltonian has not been created yet. Call build_hamiltonian() first.")
            return
        
        print(f"\n{self.name} PennyLane Hamiltonian:")
        pennylane_ham = self.hamiltonian.to_pennylane()
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

    def get_hamiltonian(self) -> Hamiltonian:
        """
        Get the Hamiltonian for this problem.
        """
        return self.hamiltonian

    def get_n_qubits(self) -> int:
        """
        Get the number of qubits for this problem.
        """
        return self.hamiltonian.num_qubits

    def __str__(self) -> str:
        """
        Return a string representation of the problem.
        
        Returns:
            String representation
        """
        metadata_str = ", ".join([f"{k}: {v}" for k, v in self.metadata.items()])
        return f"{self.name} Problem\nMetadata: {metadata_str}" 