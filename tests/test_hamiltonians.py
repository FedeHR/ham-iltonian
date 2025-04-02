"""
Tests for the Hamiltonian implementations.
"""

import os
import sys
import pytest
import numpy as np
import networkx as nx

# Add the parent directory to path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.hamiltonians import (
    Hamiltonian,
    create_maxcut_hamiltonian,
    create_tsp_hamiltonian,
    create_knapsack_hamiltonian,
    create_portfolio_hamiltonian,
    get_maxcut_solution,
    get_tsp_solution,
    get_knapsack_solution,
    get_portfolio_solution
)

class TestHamiltonian:
    """Test the Hamiltonian base class."""
    
    def test_initialization(self):
        """Test initialization of the Hamiltonian class."""
        ham = Hamiltonian(num_qubits=2)
        assert ham.num_qubits == 2
        assert len(ham.terms) == 0
        assert ham.constant == 0.0
    
    def test_add_term(self):
        """Test adding terms to the Hamiltonian."""
        ham = Hamiltonian(num_qubits=2)
        ham.add_term(0.5, "ZI")
        ham.add_term(-1.0, "ZZ")
        
        assert len(ham.terms) == 2
        assert ham.terms[0] == (0.5, "ZI")
        assert ham.terms[1] == (-1.0, "ZZ")
    
    def test_add_constant(self):
        """Test adding a constant to the Hamiltonian."""
        ham = Hamiltonian(num_qubits=2)
        ham.add_constant(3.5)
        assert ham.constant == 3.5
        
        # Add another constant
        ham.add_constant(1.5)
        assert ham.constant == 5.0
    
    def test_addition(self):
        """Test addition of Hamiltonians."""
        ham1 = Hamiltonian(num_qubits=2)
        ham1.add_term(0.5, "ZI")
        ham1.add_constant(1.0)
        
        ham2 = Hamiltonian(num_qubits=2)
        ham2.add_term(-1.0, "ZZ")
        ham2.add_constant(2.0)
        
        result = ham1 + ham2
        assert result.num_qubits == 2
        assert len(result.terms) == 2
        assert result.terms[0] == (0.5, "ZI")
        assert result.terms[1] == (-1.0, "ZZ")
        assert result.constant == 3.0
    
    def test_scalar_multiplication(self):
        """Test multiplication of a Hamiltonian by a scalar."""
        ham = Hamiltonian(num_qubits=2)
        ham.add_term(0.5, "ZI")
        ham.add_term(-1.0, "ZZ")
        ham.add_constant(2.0)
        
        result = ham * 2.0
        assert result.num_qubits == 2
        assert len(result.terms) == 2
        assert result.terms[0] == (1.0, "ZI")
        assert result.terms[1] == (-2.0, "ZZ")
        assert result.constant == 4.0
        
        # Test right multiplication
        result = 2.0 * ham
        assert result.num_qubits == 2
        assert len(result.terms) == 2
        assert result.terms[0] == (1.0, "ZI")
        assert result.terms[1] == (-2.0, "ZZ")
        assert result.constant == 4.0


class TestMaxCut:
    """Test the MaxCut Hamiltonian implementation."""
    
    def test_simple_graph(self):
        """Test with a simple graph."""
        # Create a simple 3-node graph
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 1.0), (1, 2, 2.0), (0, 2, 0.5)])
        
        ham = create_maxcut_hamiltonian(G)
        assert ham.num_qubits == 3
        assert len(ham.terms) == 3  # Three ZZ terms for three edges
        
        # Check if there's a term with the right qubit indices and coefficient
        edge_weights = {(0, 1): 1.0, (1, 2): 2.0, (0, 2): 0.5}
        for coef, term in ham.terms:
            if "," in term:
                i, j = map(int, term.split(","))
                i, j = min(i, j), max(i, j)  # Sort the indices to match the edge weights dictionary
                assert (i, j) in edge_weights
                assert coef == -edge_weights[(i, j)] / 2
    
    def test_solution_retrieval(self):
        """Test retrieving a solution for MaxCut."""
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 1.0), (1, 2, 2.0)])
        
        # 010 means nodes 0 and 2 in one partition, node 1 in the other
        # This should give a cut value of 3.0 (sum of all edge weights)
        solution = get_maxcut_solution("010", G)
        assert solution['assignment'] == {0: 0, 1: 1, 2: 0}
        assert set(solution['partition'][0]) == {0, 2}
        assert set(solution['partition'][1]) == {1}
        assert solution['cut_value'] == 3.0


class TestTSP:
    """Test the TSP Hamiltonian implementation."""
    
    def test_small_tsp(self):
        """Test with a small TSP instance."""
        # Simple 3-city problem
        distances = np.array([
            [0, 1, 2],
            [1, 0, 3],
            [2, 3, 0]
        ])
        
        A = 5.0  # Penalty for constraint A
        B = 5.0  # Penalty for constraint B
        
        ham = create_tsp_hamiltonian(distances, A, B)
        assert ham.num_qubits == 9  # 3x3 binary variables
        assert len(ham.terms) > 0
        
        # The constant should be non-zero due to the reformulation
        assert ham.constant != 0
    
    def test_solution_retrieval(self):
        """Test retrieving a TSP solution."""
        distances = np.array([
            [0, 1, 2],
            [1, 0, 3],
            [2, 3, 0]
        ])
        
        # 100010001 represents the tour 0->1->2->0
        # City 0 at position 0
        # City 1 at position 1
        # City 2 at position 2
        bit_string = "100010001"
        
        solution = get_tsp_solution(bit_string, distances)
        assert solution['valid'] == True
        assert solution['tour'] == [0, 1, 2]
        assert solution['distance'] == 6.0  # 1 + 3 + 2 = 6


class TestKnapsack:
    """Test the Knapsack Hamiltonian implementation."""
    
    def test_small_knapsack(self):
        """Test with a small Knapsack instance."""
        values = [10, 20, 30]
        weights = [5, 10, 15]
        capacity = 20
        penalty = 50.0
        
        ham = create_knapsack_hamiltonian(values, weights, capacity, penalty)
        # Should have at least the original variables
        assert ham.num_qubits >= 3
        assert len(ham.terms) > 0
    
    def test_solution_retrieval(self):
        """Test retrieving a Knapsack solution."""
        values = [10, 20, 30]
        weights = [5, 10, 15]
        capacity = 20
        
        # 110 selects items 0 and 1
        bit_string = "110"
        
        solution = get_knapsack_solution(bit_string, values, weights, capacity)
        assert solution['valid'] == True
        assert solution['selected_items'] == [0, 1]
        assert solution['total_value'] == 30
        assert solution['total_weight'] == 15
        
        # 111 selects all items, exceeds capacity
        bit_string = "111"
        
        solution = get_knapsack_solution(bit_string, values, weights, capacity)
        assert solution['valid'] == False
        assert solution['total_weight'] == 30  # Exceeds capacity


class TestPortfolio:
    """Test the Portfolio Optimization Hamiltonian implementation."""
    
    def test_small_portfolio(self):
        """Test with a small Portfolio instance."""
        returns = np.array([0.1, 0.2, 0.15])
        risk_matrix = np.array([
            [0.05, 0.01, 0.02],
            [0.01, 0.06, 0.03],
            [0.02, 0.03, 0.04]
        ])
        budget = 2
        risk_factor = 1.0
        
        ham = create_portfolio_hamiltonian(returns, risk_matrix, budget, risk_factor)
        assert ham.num_qubits == 3  # One qubit per asset
        assert len(ham.terms) > 0
    
    def test_solution_retrieval(self):
        """Test retrieving a Portfolio solution."""
        returns = np.array([0.1, 0.2, 0.15])
        risk_matrix = np.array([
            [0.05, 0.01, 0.02],
            [0.01, 0.06, 0.03],
            [0.02, 0.03, 0.04]
        ])
        budget = 2
        
        # 110 selects assets 0 and 1
        bit_string = "110"
        
        solution = get_portfolio_solution(bit_string, returns, risk_matrix, budget)
        assert solution['valid'] == True
        assert solution['selected_assets'] == [0, 1]
        assert np.isclose(solution['expected_return'], 0.3)
        assert np.isclose(solution['risk'], 0.05 + 0.06 + 2 * 0.01)  # Diagonal + correlation
        
        # 111 selects all assets, exceeds budget
        bit_string = "111"
        
        solution = get_portfolio_solution(bit_string, returns, risk_matrix, budget)
        assert solution['valid'] == False 