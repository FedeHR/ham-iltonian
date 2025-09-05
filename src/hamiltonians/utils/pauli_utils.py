"""
Utility functions for working with Pauli operators.
"""
import pennylane as qml
from typing import List, Tuple

def create_z_term(qubit_index: int, coefficient: float = 1.0) -> Tuple[float, str]:
    """
    Create a representation of a Z term on a single qubit.
    """
    return coefficient, f"Z{qubit_index}"

def create_zz_term(qubit_i: int, qubit_j: int, coefficient: float = 1.0) -> Tuple[float, str]:
    """
    Create a representation of a ZZ interaction term between two qubits.    
    """
    return coefficient, f"Z{qubit_i}@Z{qubit_j}"

def pauli_terms_to_pennylane(coefficients: List[float], pauli_terms: List[List[str]]) -> qml.Hamiltonian:
    """
    Convert coefficients and Pauli terms to a PennyLane Hamiltonian.
    
    Args:
        coefficients: List of coefficients for each term
        pauli_terms: List of lists of Pauli string representations.
                     Each string follows the format "P{qubit_index}" where P is the Pauli type (X, Y, Z).
                     Multi-qubit operators use the "@" symbol for tensor products (e.g., "Z0@Z1").
                     
    Returns:
        A PennyLane Hamiltonian object
    """
    observables = []
    for term in pauli_terms:
        if term == ["Identity"]:
            observables.append(qml.Identity(0))
            continue
            
        # Process the terms - each term could be a single Pauli or @-separated Paulis
        pauli_ops = []
        for pauli_op in term:
            # Split by @ to handle multi-operator terms like "Z0@Z1"
            individual_paulis = pauli_op.split('@')
            
            # After splitting, individual_paulis will be a list of Pauli strings, e.g. ["Z0", "Z1"] for a Z0Z1 term
            for pauli_str in individual_paulis:
                pauli_type = pauli_str[0]
                wire = int(pauli_str[1:])
                
                if pauli_type == "X":
                    pauli_ops.append(qml.PauliX(wire))
                elif pauli_type == "Y":
                    pauli_ops.append(qml.PauliY(wire))
                elif pauli_type == "Z":
                    pauli_ops.append(qml.PauliZ(wire))
                else:
                    raise ValueError(f"Unsupported Pauli type: {pauli_type}")
        
        if len(pauli_ops) == 1:
            # Directly append single Pauli operators like Z0
            observables.append(pauli_ops[0])
        else:
            # Use the tensor product operator for multi-qubit terms like Z0Z1
            observables.append(qml.prod(*pauli_ops))
    
    return qml.Hamiltonian(coefficients, observables)

# TODO: Add function to convert Hamiltonian to Qiskit format