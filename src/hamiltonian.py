import pennylane as qml
from utils.pauli_utils import pauli_terms_to_pennylane

class Hamiltonian:
    """
    Base class for representing a Hamiltonian as a sum of Pauli terms.
    """
    
    def __init__(self, num_qubits: int = 0):
        self.num_qubits = num_qubits
        self.terms = []  # List of (coefficient, pauli_term) tuples
        self.constant = 0.0  # Constant energy offset to ensure the Hamiltonian matches the objective function value
        
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
    
    def clear(self) -> None:
        """
        Clear all terms and reset the constant.
        Useful when rebuilding a Hamiltonian after parameter modifications.
        """
        self.terms = []
        self.constant = 0.0
    
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
        