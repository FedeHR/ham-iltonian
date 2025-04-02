"""
Utility functions for working with Pauli operators.
"""
import pennylane as qml
from typing import List, Tuple

def create_z_term(qubit_index: int, coefficient: float = 1.0) -> Tuple[float, str]:
    """
    Create a Z term on a single qubit.
    """
    return coefficient, f"Z{qubit_index}"

def create_zz_term(qubit_i: int, qubit_j: int, coefficient: float = 1.0) -> Tuple[float, str]:
    """
    Create a ZZ interaction term between two qubits.    
    """
    return coefficient, f"{qubit_i},{qubit_j}"

def pauli_term_to_pennylane(coefficients: List[float], pauli_terms: List[List[str]]) -> qml.Hamiltonian:
    """
    Convert coefficients and Pauli terms to a PennyLane Hamiltonian.
    """
    observables = []
    for term in pauli_terms:
        if not term:  # Identity term
            observables.append(qml.Identity(0))
            continue
            
        pauli_ops = []
        for pauli_str in term:
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
            observables.append(pauli_ops[0])
        else:
            observables.append(qml.prod(*pauli_ops))
    
    return qml.Hamiltonian(coefficients, observables)
