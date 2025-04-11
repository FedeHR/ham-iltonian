"""
Implementation of the Ising Hamiltonian for the MaxCut problem.
"""
import networkx as nx
import numpy as np
from typing import List
from hamiltonians.base import Hamiltonian
from utils.pauli_utils import create_zz_term

class MaxCutHamiltonian(Hamiltonian):
    """
    Hamiltonian for the MaxCut problem with custom modifiers.
    """
    
    def __init__(self, graph: nx.Graph):
        """
        Initialize a MaxCut Hamiltonian from a graph.
        
        Args:
            graph: NetworkX graph with weighted edges
        """
        n_nodes = graph.number_of_nodes()
        super().__init__(n_nodes)
        
        self.metadata["problem"] = "MaxCut"
        self.metadata["graph"] = graph
        
        constant_term = 0.0
        
        for i, j, attr in graph.edges(data=True):
            weight = attr.get('weight', 1.0)
            
            # Add constant term: w_ij / 2
            constant_term += weight / 2
            
            # Add interaction term: -w_ij * Z_i Z_j / 2
            coeff, term = create_zz_term(i, j, -weight / 2)
            self.add_term(coeff, term)
        
        self.add_constant(constant_term)
        
        # Register MaxCut-specific modifiers
        self._register_maxcut_modifiers()
    
    def _register_maxcut_modifiers(self):
        """
        Register MaxCut-specific coefficient modifier functions.
        These modifiers are designed to work directly with graph weights,
        not the Hamiltonian coefficients.
        """
        # Add a modifier that scales based on edge density
        self.add_modifier_function("edge_density_scaling", self._edge_density_modifier)
        
        # Add a modifier that scales based on node degree
        self.add_modifier_function("degree_weighted", self._degree_weighted_modifier)
        
        # Add a weighted sine function modifier for periodic transformations
        self.add_modifier_function("weighted_sine", lambda weight, params: weight * np.sin(params[0] * params[1]))
        
        # Add a weight_emphasis modifier that adjusts the importance of weights
        self.add_modifier_function("weight_emphasis", self._weight_emphasis_modifier)
        
        # Add standard modifiers that work with weights directly
        self.add_modifier_function("linear", lambda weight, params: weight + params[0])
        self.add_modifier_function("quadratic", lambda weight, params: weight * (params[0] ** 2))
        self.add_modifier_function("exponential", lambda weight, params: weight * np.exp(params[0]))
    
    def _edge_density_modifier(self, weight: float, scaling_factor: float) -> float:
        """
        Custom modifier that scales weights based on edge density and a scaling factor.
        
        Args:
            weight: Original edge weight
            scaling_factor: Factor to scale the edge density effect
            
        Returns:
            Modified weight
        """
        graph = self.metadata.get("graph")
        if not graph:
            return weight
            
        n_nodes = graph.number_of_nodes()
        max_edges = n_nodes * (n_nodes - 1) / 2
        edge_density = graph.number_of_edges() / max_edges

        return weight * (1.0 + scaling_factor * edge_density)
    
    def _degree_weighted_modifier(self, weight: float, params: List[float]) -> float:
        """
        Custom modifier that scales weights based on node degrees.
        
        Args:
            weight: Original edge weight
            params: List containing [scaling_factor, degree_exponent] or single scaling_factor
            
        Returns:
            Modified weight
        """
        graph = self.metadata.get("graph")
        if not graph:
            return weight
            
        scaling_factor = params[0]
        degree_exponent = params[1] if len(params) > 1 else 1.0

        # Calculate average degree
        avg_degree = sum(dict(graph.degree()).values()) / graph.number_of_nodes()
        
        # Scale weight based on average degree raised to the specified exponent
        return weight * (1.0 + scaling_factor * (avg_degree ** degree_exponent))
    
    def _weight_emphasis_modifier(self, weight: float, emphasis_factor: float) -> float:
        """
        Modifier that emphasizes or de-emphasizes edge weights in the MaxCut problem.
        This can be useful to balance between uniform cuts (all edges equal importance)
        and weighted cuts (edge weights determine importance).
        
        Args:
            weight: Original edge weight
            emphasis_factor: Single parameter where:
                - emphasis_factor > 1: Increases the influence of heavy-weighted edges
                - emphasis_factor = 1: No change (original weights)
                - 0 < emphasis_factor < 1: Reduces weight differences, making the problem closer to unweighted MaxCut
                - emphasis_factor = 0: All weights become equal (unweighted MaxCut)
            
        Returns:
            Modified weight
        """
        if emphasis_factor == 1.0:
            return weight
            
        sign = -1 if weight < 0 else 1
        abs_weight = abs(weight)
        
        if emphasis_factor == 0:
            # Make all weights equal (preserves sign)
            return sign
        elif emphasis_factor > 1.0:
            # Emphasize weight differences by raising to a power
            return sign * (abs_weight ** emphasis_factor)
        else:
            # De-emphasize weight differences (bringing them closer to uniform)
            return sign * (abs_weight ** emphasis_factor)