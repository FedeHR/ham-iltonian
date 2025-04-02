"""
CO Hamiltonians - A library for creating Hamiltonians representing 
combinatorial optimization problems as combinations of Pauli operators.
"""

from .base import Hamiltonian
from .maxcut import create_maxcut_hamiltonian, get_maxcut_solution
from .tsp import create_tsp_hamiltonian, get_tsp_solution
from .knapsack import create_knapsack_hamiltonian, get_knapsack_solution
from .portfolio import create_portfolio_hamiltonian, get_portfolio_solution

__version__ = "0.1.0" 