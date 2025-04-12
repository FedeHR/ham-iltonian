"""
CO Hamiltonians - A library for creating Hamiltonians representing 
combinatorial optimization problems as combinations of Pauli operators.
"""

from .hamiltonian import Hamiltonian

from .problems import (
    Problem,
    MaxCutProblem,
    KnapsackProblem,
    TSPProblem,
    PortfolioProblem
)

from .utils import (
    pauli_utils,
    graph_utils
)

__version__ = "0.1.0" 