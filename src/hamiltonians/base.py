from typing import List, Optional, Callable, Union
import pennylane as qml
from utils.pauli_utils import pauli_terms_to_pennylane
import numpy as np

class Hamiltonian:
    """
    Base class for representing a Hamiltonian as a sum of Pauli terms.
    """
    
    def __init__(self, num_qubits: int = 0):
        self.num_qubits = num_qubits
        self.terms = []  # List of (coefficient, pauli_term) tuples
        self.constant = 0.0  # Constant energy offset to ensure the Hamiltonian matches the objective function value
        self.metadata = {}

        # Dictionary of parameter names to functions that modify coefficients
        self.modifier_functions = {
            # Default modifier functions
            "linear": lambda coefficient, params: coefficient + params[0],
            "quadratic": lambda coefficient, params: coefficient * (params[0] ** 2),
            "exponential": lambda coefficient, params: coefficient * np.exp(params[0]),
            "chaotic": lambda coefficient: coefficient * np.random.uniform(0, 1),
        }
        
    def _update_num_qubits(self, pauli_term: str) -> None:
        """
        Update the number of qubits if the term requires more
        """
        # Split by @ to handle multi-operator terms like "Z0@Z1" (interaction term between qubits 0 and 1)
        individual_paulis = pauli_term.split('@')
        
        for pauli_str in individual_paulis:
            # Extract the qubit index from each Pauli operator
            qubit_idx = int(pauli_str[1:])
            self.num_qubits = max(self.num_qubits, qubit_idx + 1)

    def add_term(self, coefficient: float, pauli_term: str) -> None:
        """
        Add a Pauli term to the Hamiltonian.
        
        Args:
            coefficient: Coefficient for the term
            pauli_term: Pauli term as a string (e.g., "Z0" for a single qubit Pauli Z operator, or "Z0@Z1" for an interaction term between qubits 0 and 1)
        """
        self.terms.append((coefficient, pauli_term))
        self._update_num_qubits(pauli_term)

    def add_modifier_function(self, function_name: str,
                              function: Callable[[float, Optional[Union[float, List[float]]]], float]) -> None:
        """
        Add a custom modifier function to modify the coefficients of the Hamiltonian.
        Args:
            function_name: Name of the modifier
            function: Function that takes a coefficient and optionally a parameter or a list of parameters and returns a modified coefficient
        """
        self.modifier_functions[function_name] = function


    def modify_coefficients(self, modifier_name: str, *args, new_hamiltonian=True) -> Optional['Hamiltonian']:
        """
        Modify the coefficients of the Hamiltonian based on a specified modifier function and parameters.

        Args:
            modifier_name: Name of the registered modifier function to apply
            args: Parameters for the modifier function
            new_hamiltonian: If True, return a new Hamiltonian with modified coefficients;
                if False, modify the current Hamiltonian in place

        Returns:
            A new Hamiltonian with modified coefficients if new_hamiltonian is True, None otherwise

        Raises:
            ValueError: If the modifier_name is not registered
        """
        if modifier_name not in self.modifier_functions:
            raise ValueError(f"Unknown modifier '{modifier_name}'. "
                             f"Available modifiers: {list(self.modifier_functions.keys())}")

        modifier_func = self.modifier_functions[modifier_name]
        modified_terms = [(modifier_func(coefficient, args), pauli_term) for coefficient, pauli_term in self.terms]

        if new_hamiltonian:
            new_hamiltonian = Hamiltonian(self.num_qubits)
            new_hamiltonian.terms = modified_terms
            new_hamiltonian.constant = self.constant
            new_hamiltonian.metadata = self.metadata.copy()
            return new_hamiltonian
        else:
            self.terms = modified_terms
        
    def to_pennylane(self) -> qml.Hamiltonian:
        """
        Convert this Hamiltonian to a PennyLane Hamiltonian.
        
        Returns:
            A PennyLane Hamiltonian object
        """
        coefficients = [coeff for coeff, _ in self.terms]
        pauli_terms = [[term] for _, term in self.terms]  # Wrap each term in a list to later handle multi-operator terms with a simple product operator
        
        if self.constant != 0:
            coefficients.append(self.constant)
            pauli_terms.append(["Identity"])
        
        return pauli_terms_to_pennylane(coefficients, pauli_terms)
    
    def add_constant(self, value: float) -> None:
        """
        Add a constant term to the Hamiltonian.
        
        Args:
            value: Constant value to add
        """
        self.constant += value
    
    def __str__(self) -> str:
        """
        Return a string representation of the Hamiltonian.
        
        Returns:
            String representation
        """
        terms_strs = []
        
        for coefficient, pauli_term in self.terms:
            # Format the coefficient
            if coefficient == 0:
                continue
                
            # Handle the sign
            if terms_strs and coefficient >= 0:
                term_str = f"+ {abs(coefficient):.2f}"
            else:
                term_str = f"- {abs(coefficient):.2f}"
            
            # Add the Pauli operator
            term_str += f" * {pauli_term}"
            terms_strs.append(term_str)
        
        # Add the constant term if non-zero
        if self.constant != 0:
            if terms_strs and self.constant > 0:
                terms_strs.append(f"+ {self.constant}")
            elif self.constant < 0:
                terms_strs.append(f"- {abs(self.constant)}")
            else:
                terms_strs.append(f"{self.constant}")
        
        # Join all terms
        if not terms_strs:
            return "0"
        
        return " ".join(terms_strs)
    
    def __add__(self, other: 'Hamiltonian') -> 'Hamiltonian':
        """
        Add two Hamiltonians.
        
        Args:
            other: Hamiltonian to add
            
        Returns:
            Sum of the two Hamiltonians
        """
        # Create a new Hamiltonian with the combined number of qubits
        result = Hamiltonian(max(self.num_qubits, other.num_qubits))
        
        # Add all terms from self
        for coefficient, pauli_term in self.terms:
            result.add_term(coefficient, pauli_term)
        
        # Add all terms from other
        for coefficient, pauli_term in other.terms:
            result.add_term(coefficient, pauli_term)
        
        # Add the constants
        result.constant = self.constant + other.constant
        
        # Copy all modifier functions
        result.modifier_functions = self.modifier_functions.copy()
        result.modifier_functions.update(other.modifier_functions)
        
        # Copy metadata
        result.metadata = {**self.metadata, **other.metadata}
        
        return result
    
    def __mul__(self, scalar: float) -> 'Hamiltonian':
        """
        Multiply Hamiltonian by a scalar.
        
        Args:
            scalar: Scalar to multiply by
            
        Returns:
            Scaled Hamiltonian
        """
        # Create a new Hamiltonian with the same number of qubits
        result = Hamiltonian(self.num_qubits)
        
        # Scale all terms
        for coefficient, pauli_term in self.terms:
            result.add_term(coefficient * scalar, pauli_term)
        
        # Scale the constant
        result.constant = self.constant * scalar
        
        # Copy modifier functions
        result.modifier_functions = self.modifier_functions.copy()
        
        # Copy metadata
        result.metadata = self.metadata.copy()
        
        return result
    
    def __rmul__(self, scalar: float) -> 'Hamiltonian':
        """
        Right multiply Hamiltonian by a scalar.
        
        Args:
            scalar: Scalar to multiply by
            
        Returns:
            Scaled Hamiltonian
        """
        return self.__mul__(scalar) 
        