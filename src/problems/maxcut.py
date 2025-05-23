"""
The MaxCut problem seeks to partition vertices of a graph into two sets 
such that the sum of weights of edges between the two sets is maximized.
"""
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Optional, Any, List, Union

from problems.base import Problem
from hamiltonian import Hamiltonian
from utils.pauli_utils import create_zz_term
from utils.classical_solvers import solve_maxcut_brute_force, evaluate_mc_solution
from parameter_modifiers.maxcut import get_modifiers

class MaxCutProblem(Problem):
    """
    MaxCut Problem representation.
    """
    def __init__(self, graph: nx.Graph, name: str = "MaxCut"):
        """
        Initialize a MaxCut problem.
        
        Args:
            graph: NetworkX graph with weighted edges (for unweighted MaxCut, set all edge weights to 1)
            name: Name of the problem instance
        """
        super().__init__(name)
        self.graph = graph
        self.original_graph = graph.copy()
        self.node_positions = nx.spring_layout(self.graph, seed=42)  # Store persistent positions

        default_maxcut_modifiers = get_modifiers()
        self.modifier_functions.update(default_maxcut_modifiers)
    
    def _apply_modifier(self, modifier_name: str, *args) -> None:
        """
        Apply the modifier to the graph weights.
        
        Args:
            modifier_name: Name of the modifier function to apply
            *args: Parameters for the modifier function
        """
        modifier_func = self.modifier_functions[modifier_name]

        for u, v, data in self.graph.edges(data=True):
            original_weight = data.get("weight")
            modified_weight = modifier_func(original_weight, *args)
            self.graph[u][v]["weight"] = modified_weight

    def build_hamiltonian(self) -> None:
        """
        Build the MaxCut Hamiltonian from the current graph.
        """
        # Create a new Hamiltonian or clear the existing one
        if self.hamiltonian is None:
            self.hamiltonian = Hamiltonian(self.graph.number_of_nodes())
        else:
            self.hamiltonian.clear()
        
        constant_term = 0.0
        
        # Build the Hamiltonian terms
        for i, j, attr in self.graph.edges(data=True):
            weight = attr.get('weight', 1.0)
            
            # Add constant term: w_ij / 2
            constant_term += weight / 2
            
            # Add interaction term: -w_ij * Z_i Z_j / 2
            coefficient, term = create_zz_term(i, j, -weight / 2)
            self.hamiltonian.add_term(coefficient, term)
        
        self.hamiltonian.add_constant(constant_term)
    
    def solve_classically(self) -> Dict[str, Any]:
        """
        Solve the MaxCut problem using classical methods.
        
        Returns:
            Dictionary with solution details
        """
        solution = solve_maxcut_brute_force(self.graph)
        self.solutions["brute_force"] = solution
        return solution
    
    def evaluate_bitstring(self, bitstring: Union[str, List[int]]) -> Dict[str, Any]:
        """
        Get the MaxCut solution from a bitstring.
        
        Args:
            bitstring: Binary string or list of 0s and 1s representing the solution
            
        Returns:
            Dictionary with solution details
        """
        if isinstance(bitstring, str):
            assignment = [int(bit) for bit in bitstring]
        else:
            assignment = bitstring
        
        assignment_dict, cut_value, set_0, set_1 = evaluate_mc_solution(assignment, self.graph)
        
        return {
            "bitstring": bitstring,
            "assignment": assignment_dict,
            "partition": [set_0, set_1],
            "cut_value": cut_value,
        }
    
    def calculate_quality(self, solution: Dict[str, Any]) -> float:
        return solution["cut_value"]
    
    def reset_parameters(self):
        """
        Reset the graph, resetting all parameters to their original values.
        """
        # Reset the graph weights to original values
        self.graph = self.original_graph.copy()
        self.build_hamiltonian()
    
    def visualize_graph(self, filename: Optional[str] = None) -> None:
        """
        Visualize the graph.
        """
        plt.figure(figsize=(8, 6))
        nx.draw(self.graph, with_labels=True, pos=self.node_positions,
                node_color='tab:blue', node_size=700, width=2, font_color='white')

        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in self.graph.edges(data=True)}
        nx.draw_networkx_edge_labels(self.graph, self.node_positions, edge_labels=edge_labels)

        if filename:
            plt.savefig(filename)
            print(f"Visualization saved to {filename}")
        plt.show()
    
    def visualize_solution(self, solution: Dict[str, Any], filename: Optional[str] = None) -> None:
        """
        Visualize a MaxCut solution.
        
        Args:
            solution: Solution dictionary
            filename: Optional filename to save the visualization
        """
        if not solution:
            raise ValueError("First run solve_classically() to get a solution.")
        
        plt.figure(figsize=(8, 6))
        
        # Draw nodes
        node_colors = ['tab:blue' if solution['assignment'][i] == 0 else 'tab:red' for i in range(self.graph.number_of_nodes())]
        nx.draw_networkx_nodes(self.graph, self.node_positions, node_color=node_colors, node_size=700)
        nx.draw_networkx_labels(self.graph, self.node_positions, font_color='white')

        # Draw edges
        cut_edges = [(u, v) for u, v in self.graph.edges() if solution['assignment'][u] != solution['assignment'][v]]
        nx.draw_networkx_edges(self.graph, self.node_positions, edgelist=cut_edges, edge_color='red', style='dashed', width=2)
        uncut_edges = [(u, v) for u, v in self.graph.edges() if solution['assignment'][u] == solution['assignment'][v]]
        nx.draw_networkx_edges(self.graph, self.node_positions, edgelist=uncut_edges, edge_color='black', style='solid', width=2)

        # Draw edge weights
        edge_weights = {(u, v): f"{d['weight']:.2f}" for u, v, d in self.graph.edges(data=True)}
        nx.draw_networkx_edge_labels(self.graph, self.node_positions, edge_labels=edge_weights)

        plt.axis('off')
        plt.title(f'MaxCut Solution - Cut Value: {solution["cut_value"]:.2f}')
        
        if filename:
            plt.savefig(filename)
            print(f"Visualization saved to {filename}")

        plt.show()
    
    def __str__(self) -> str:
        """
        Return a string representation of the MaxCut problem.
        
        Returns:
            String description of the problem
        """
        edge_str = ", ".join([f"({u},{v}): {d['weight']:.1f}" for u, v, d in self.graph.edges(data=True)])
        return f"{self.name} Problem with {self.graph.number_of_nodes()} nodes and " \
               f"{self.graph.number_of_edges()} edges\nEdges: {edge_str}" 