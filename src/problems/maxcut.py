"""
The MaxCut problem seeks to partition vertices of a graph into two sets 
such that the sum of weights of edges between the two sets is maximized.
"""
# TODO: Question for Tobias: should the modifier be at problem level or Hamiltonian level?
#  more intuitive at problem level, but technically the Hamiltonian should be parametrized
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Optional, Any, List, Union

from problems.base import Problem
from hamiltonians.maxcut import MaxCutHamiltonian
from utils.classical_solvers import solve_maxcut_brute_force

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
        self.node_positions = nx.spring_layout(self.graph, seed=42)  # Store persistent positions
        
        # Store the original edge weights for easy access
        self.original_weights = {}
        for u, v, data in graph.edges(data=True):
            self.original_weights[(u, v)] = data.get("weight", 1.0)
            self.original_weights[(v, u)] = data.get("weight", 1.0)
    
    def create_hamiltonian(self) -> Any:
        """
        Create the Hamiltonian for this MaxCut problem.
        
        Returns:
            Hamiltonian for the MaxCut problem
        """
        self.hamiltonian = MaxCutHamiltonian(self.graph)
        return self.hamiltonian
    
    def modify_hamiltonian(self, modifier_name: str, *args) -> None:
        """
        Modify the graph weights directly based on the modifier and then update the Hamiltonian.
        
        Args:
            modifier_name: Name of the modifier function to apply
            *args: Parameters for the modifier function
        """
        if not self.hamiltonian:
            raise ValueError("Hamiltonian has not been created yet. Call create_hamiltonian() first.")
            
        # Get the modifier function from the Hamiltonian
        if modifier_name not in self.hamiltonian.modifier_functions:
            raise ValueError(f"Unknown modifier '{modifier_name}'. "
                             f"Available modifiers: {list(self.hamiltonian.modifier_functions.keys())}")
        
        modifier_func = self.hamiltonian.modifier_functions[modifier_name]
        
        # Apply the modifier directly to the graph weights
        modified_graph = self.graph.copy()
        for u, v, data in self.graph.edges(data=True):
            original_weight = data.get("weight", 1.0)
            # Apply the modifier to the weight directly
            modified_weight = modifier_func(original_weight, args)
            modified_graph[u][v]["weight"] = modified_weight
        
        # Update the graph with modified weights
        self.graph = modified_graph
        
        # Recreate the Hamiltonian based on the modified graph
        self.hamiltonian = MaxCutHamiltonian(self.graph)
    
    def update_graph_from_hamiltonian(self) -> None:
        """
        Update the graph weights based on the current Hamiltonian coefficients.
        This is called automatically when the Hamiltonian is modified.
        """
            
        # Update edge weights based on Hamiltonian coefficients
        for coeff, term in self.hamiltonian.terms:
            # Extract edge indices from the term (e.g., "Z0@Z1" refers to edge (0,1))
            parts = term.split('@')
            u = int(parts[0][1:])  # Remove the 'Z' prefix
            v = int(parts[1][1:])
            
            # The Hamiltonian term coefficient is -w_ij/2, so we multiply by -2 to get weight
            weight = -2 * coeff
            self.graph[u][v]['weight'] = weight
    
    def solve_classically(self) -> Dict[str, Any]:
        """
        Solve the MaxCut problem using classical methods.
        
        Returns:
            Dictionary with solution details
        """
        solution = solve_maxcut_brute_force(self.graph)
        self.add_solution("brute_force", solution)
        return solution
    
    def get_solution_from_bitstring(self, bitstring: Union[str, List[int]]) -> Dict[str, Any]:
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
        
        # Convert to dictionary for compatibility with test expectations
        assignment_dict = {i: bit for i, bit in enumerate(assignment)}
        
        # Partition the nodes
        set_0 = [i for i, bit in enumerate(assignment) if bit == 0]
        set_1 = [i for i, bit in enumerate(assignment) if bit == 1]
        
        # Calculate the cut value
        cut_value = 0.0
        for i, j, attr in self.graph.edges(data=True):
            weight = attr.get('weight', 1.0)
            if (i in set_0 and j in set_1) or (i in set_1 and j in set_0):
                cut_value += weight
        
        return {
            "bitstring": bitstring,
            "assignment": assignment_dict,
            "partition": [set_0, set_1],
            "cut_value": cut_value,
        }
    
    def calculate_quality(self, solution: Dict[str, Any]) -> float:
        return solution["cut_value"]
    
    # TODO: interesting, NxGraph provides the option to convert graph to LaTeX!
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