"""
MaxCut Problem implementation.
"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any

from .base import Problem
from ..hamiltonians.maxcut import create_maxcut_hamiltonian, get_maxcut_solution
from ..utils.classical_solvers import solve_maxcut_brute_force

class MaxCutProblem(Problem):
    """
    MaxCut Problem representation.
    
    The MaxCut problem seeks to partition vertices of a graph into two sets
    such that the sum of weights of edges between the two sets is maximized.
    """
    
    def __init__(self, graph: nx.Graph, name: str = "MaxCut"):
        """
        Initialize a MaxCut problem.
        
        Args:
            graph: NetworkX graph with weighted edges
            name: Name of the problem instance
        """
        super().__init__(name)
        self.graph = graph
        self.metadata["n_nodes"] = graph.number_of_nodes()
        self.metadata["n_edges"] = graph.number_of_edges()
        
        # Store the edge weights for easy access
        self.edge_weights = {}
        for u, v, data in graph.edges(data=True):
            self.edge_weights[(u, v)] = data.get("weight", 1.0)
            self.edge_weights[(v, u)] = data.get("weight", 1.0)
    
    def create_hamiltonian(self, **kwargs) -> Any:
        """
        Create the Hamiltonian for this MaxCut problem.
        
        Returns:
            Hamiltonian for the MaxCut problem
        """
        self._hamiltonian = create_maxcut_hamiltonian(self.graph)
        return self._hamiltonian
    
    def solve_classically(self, **kwargs) -> Dict[str, Any]:
        """
        Solve the MaxCut problem using classical methods.
        
        Returns:
            Dictionary with solution details
        """
        solution = solve_maxcut_brute_force(self.graph)
        self.add_solution("classical", solution)
        return solution
    
    def get_solution_from_bitstring(self, bitstring: str) -> Dict[str, Any]:
        """
        Get the MaxCut solution from a bitstring.
        
        Args:
            bitstring: Binary string representation of the solution
            
        Returns:
            Dictionary with solution details
        """
        return get_maxcut_solution(bitstring, self.graph)
    
    def calculate_quality(self, solution: Dict[str, Any]) -> float:
        """
        Calculate the quality of a MaxCut solution.
        
        For MaxCut, the quality is the cut value.
        
        Args:
            solution: Solution dictionary
            
        Returns:
            Cut value (higher is better)
        """
        return solution["cut_value"]
    
    def visualize_solution(self, solution: Dict[str, Any], filename: Optional[str] = None) -> None:
        """
        Visualize a MaxCut solution.
        
        Args:
            solution: Solution dictionary
            filename: Optional filename to save the visualization
        """
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(self.graph, seed=42)
        
        # Define node colors based on the assignment
        node_colors = ['skyblue' if solution['assignment'][i] == 0 else 'lightcoral' 
                      for i in range(self.graph.number_of_nodes())]
        
        # Draw the graph
        nx.draw(self.graph, pos, with_labels=True, node_color=node_colors, 
                font_weight='bold', node_size=700)
        
        # Draw edge weights
        edge_labels = {(u, v): f"{d['weight']:.1f}" for u, v, d in self.graph.edges(data=True)}
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels)
        
        # Highlight cut edges
        cut_edges = [(u, v) for u, v, d in self.graph.edges(data=True) 
                     if solution['assignment'][u] != solution['assignment'][v]]
        nx.draw_networkx_edges(self.graph, pos, edgelist=cut_edges, width=3, edge_color='red')
        
        plt.title(f'MaxCut Solution - Cut Value: {solution["cut_value"]:.2f}')
        
        if filename:
            plt.savefig(filename)
            print(f"Visualization saved to {filename}")
        else:
            plt.savefig("temp/maxcut_solution.png")
            print("Visualization saved to temp/maxcut_solution.png")
        
        plt.show()
    
    def __str__(self) -> str:
        """
        Return a string representation of the MaxCut problem.
        
        Returns:
            String description of the problem
        """
        edge_str = ", ".join([f"({u},{v}): {w:.1f}" for (u, v), w in self.edge_weights.items() 
                             if u < v])  # Only include each edge once
        return f"{self.name} Problem with {self.metadata['n_nodes']} nodes and " \
               f"{self.metadata['n_edges']} edges\nEdges: {edge_str}" 