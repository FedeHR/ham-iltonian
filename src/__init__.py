"""
CO Hamiltonians - A library for creating Hamiltonians representing 
combinatorial optimization problems as combinations of Pauli operators.
"""

from .hamiltonians import (
    Hamiltonian,
    create_maxcut_hamiltonian, get_maxcut_solution,
    create_tsp_hamiltonian, get_tsp_solution,
    create_knapsack_hamiltonian, get_knapsack_solution,
    create_portfolio_hamiltonian, get_portfolio_solution
)

from .problems import (
    Problem,
    MaxCutProblem,
    KnapsackProblem,
    TSPProblem,
    PortfolioProblem
)

__version__ = "0.1.0" 