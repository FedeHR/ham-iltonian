from typing import Dict, Any, Callable
import pennylane as qml
from ..utils.pauli_utils import pauli_term_to_pennylane

class Hamiltonian:
    """
    Base class for representing a Hamiltonian as a sum of Pauli terms.
    """
    
    def __init__(self, num_qubits: int = 0):
        self.num_qubits = num_qubits
        self.terms = []  # List of (coefficient, pauli_term) tuples
        self.constant = 0.0  # Constant energy offset to ensure the Hamiltonian matches the objective function value
        self.param_functions = {}  # Dictionary of parameter names to functions that modify coefficients
        self.metadata = {}
        
    def _update_num_qubits(self, pauli_term: str) -> None:
        """
        Update the number of qubits if the term requires more
        """
        for char in pauli_term:
            if char.isdigit():
                qubit_idx = int(char)
                self.num_qubits = max(self.num_qubits, qubit_idx + 1)

    def add_term(self, coefficient: float, pauli_term: str) -> None:
        """
        Add a Pauli term to the Hamiltonian.
        """
        self.terms.append((coefficient, pauli_term))
        self._update_num_qubits(pauli_term)
    
    def add_parametric_term(self, 
                          base_coefficient: float, 
                          pauli_term: str, 
                          param_name: str, 
                          param_function: Callable[[float, Dict[str, Any]], float]) -> None:
        """
        Add a parametric Pauli term to the Hamiltonian, where the coefficient depends on parameters.
        
        Args:
            base_coefficient: Base coefficient for the term before modification
            pauli_term: Pauli term string representation
            param_name: Name of the parameter
            param_function: Function that takes the base coefficient and parameter values dictionary,
                            and returns the modified coefficient
        """
        self.terms.append((base_coefficient, pauli_term))
        
        # Store the parameter function
        term_idx = len(self.terms) - 1
        if param_name not in self.param_functions:
            self.param_functions[param_name] = {}
        
        self.param_functions[param_name][term_idx] = param_function
        self._update_num_qubits(pauli_term)
    
    def evaluate_with_parameters(self, param_values: Dict[str, Any]) -> 'Hamiltonian':
        """
        Evaluate the Hamiltonian with specific parameter values.
        
        Args:
            param_values: Dictionary of parameter names to values
            
        Returns:
            A new Hamiltonian with coefficients computed based on parameters
        """
        new_hamiltonian = Hamiltonian(self.num_qubits)
        new_hamiltonian.constant = self.constant
        new_hamiltonian.metadata = self.metadata.copy()
        
        # Add all terms with possibly modified coefficients
        for idx, (coefficient, pauli_term) in enumerate(self.terms):
            new_coeff = coefficient
            
            # Check if this term depends on any parameters
            for param_name, term_dict in self.param_functions.items():
                if idx in term_dict and param_name in param_values:
                    param_func = term_dict[idx]
                    new_coeff = param_func(new_coeff, param_values)
            
            new_hamiltonian.add_term(new_coeff, pauli_term)
        return new_hamiltonian
        
    def to_pennylane(self) -> qml.Hamiltonian:
        coefficients = []
        pauli_terms = []
        
        # First process all non-identity terms
        for coefficient, pauli_term in self.terms:
            if ',' in pauli_term:
                # This is a ZZ term from the utility function
                # Need to process differently
                i, j = map(int, pauli_term.split(','))
                pauli_terms.append([f"Z{i}", f"Z{j}"])
            else:
                # Regular term
                pauli_terms.append([pauli_term])
            coefficients.append(coefficient)
        
        # Add the constant term if non-zero
        if self.constant != 0:
            coefficients.append(self.constant)
            pauli_terms.append([])  # Empty list for the identity term
        
        # Convert to PennyLane representation
        return pauli_term_to_pennylane(coefficients, pauli_terms)
    
    def add_constant(self, value: float) -> None:
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
            if terms_strs and coefficient > 0:
                term_str = f"+ {abs(coefficient)}"
            elif coefficient < 0:
                term_str = f"- {abs(coefficient)}"
            else:
                term_str = f"{coefficient}"
            
            # Add the Pauli operator if it's not just a constant
            if ',' in pauli_term:
                # ZZ term from utility function
                i, j = map(int, pauli_term.split(','))
                term_str += f" Z{i}Z{j}"
            else:
                term_str += f" {pauli_term}"
                
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
        if not isinstance(other, Hamiltonian):
            raise TypeError("Can only add Hamiltonians together")
        
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
        
        # Combine parameter functions
        for param_name, term_dict in self.param_functions.items():
            if param_name not in result.param_functions:
                result.param_functions[param_name] = {}
            for term_idx, param_func in term_dict.items():
                result.param_functions[param_name][term_idx] = param_func
        
        for param_name, term_dict in other.param_functions.items():
            if param_name not in result.param_functions:
                result.param_functions[param_name] = {}
            for term_idx, param_func in term_dict.items():
                # Adjust term indices from other
                adjusted_idx = term_idx + len(self.terms)
                result.param_functions[param_name][adjusted_idx] = param_func
        
        return result
    
    def __mul__(self, scalar: float) -> 'Hamiltonian':
        """
        Multiply Hamiltonian by a scalar.
        
        Args:
            scalar: Scalar to multiply by
            
        Returns:
            Scaled Hamiltonian
        """
        if not isinstance(scalar, (int, float)):
            raise TypeError("Can only multiply Hamiltonian by a scalar")
        
        # Create a new Hamiltonian with the same number of qubits
        result = Hamiltonian(self.num_qubits)
        
        # Scale all terms
        for coefficient, pauli_term in self.terms:
            result.add_term(coefficient * scalar, pauli_term)
        
        # Scale the constant
        result.constant = self.constant * scalar
        
        # Adjust parameter functions
        for param_name, term_dict in self.param_functions.items():
            if param_name not in result.param_functions:
                result.param_functions[param_name] = {}
            for term_idx, param_func in term_dict.items():
                # Create a new function that includes the scalar
                def scaled_func(coeff, params, orig_func=param_func, scale=scalar):
                    return orig_func(coeff / scale, params) * scale
                result.param_functions[param_name][term_idx] = scaled_func
        
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