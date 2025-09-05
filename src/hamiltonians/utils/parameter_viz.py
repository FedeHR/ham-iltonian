"""
Parameter effect visualization utilities for combinatorial optimization problems.
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import List, Dict, Any, Callable, Optional, Tuple
import seaborn as sns
from tqdm import tqdm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import inspect

def _get_modifier_param_name(problem, modifier_name: str) -> Optional[str]:
    """
    Inspect a modifier function to find the primary parameter name.
    """
    if modifier_name not in problem.modifier_functions:
        raise ValueError(f"Modifier '{modifier_name}' not found.")
        
    modifier_func = problem.modifier_functions[modifier_name]
    sig = inspect.signature(modifier_func)
    
    # Find the first parameter that is not 'weight', 'graph', or 'edge_param'
    for param in sig.parameters.values():
        if param.name not in ['weight', 'graph', 'edge_param']:
            return param.name
            
    return None

def compare_graph_weights(problem, param_name: str, param_values: List[Any], graph: Optional[nx.Graph] = None,
                          title: str = "Weight Comparison", figsize: Tuple[int, int] = (12, 8),
                          filename: Optional[str] = None) -> None:
    """
    Visualize how edge weights change with different parameter values.
    
    Args:
        problem: Problem instance (e.g., MaxCutProblem)
        param_name: Name of the parameter modifier to apply
        param_values: List of parameter values to visualize
        graph: Optional graph to visualize
        title: Plot title
        figsize: Figure size
        filename: Optional filename to save visualization
    """
    n_params = len(param_values)
    
    # Store original graph to restore later
    original_graph = problem.graph.copy()
    
    plt.figure(figsize=figsize)
    
    # Calculate subplot grid dimensions
    cols = min(3, n_params)
    rows = (n_params + cols - 1) // cols
    
    # Create a colormap for edge weights
    all_weights = []
    
    # First pass to gather all weights for consistent color scaling
    for i, value in enumerate(param_values):
        problem.reset_parameters()
        kwarg_name = _get_modifier_param_name(problem, param_name)
        if kwarg_name:
            problem.modify_parameters(param_name, **{kwarg_name: value})
        else:
            # Fallback for modifiers with no extra params, though less common for this function
            problem.modify_parameters(param_name)
        all_weights.extend([data['weight'] for _, _, data in problem.graph.edges(data=True)])
    
    vmin, vmax = min(all_weights), max(all_weights)
    
    # Second pass to create visualizations
    for i, value in enumerate(param_values):
        problem.reset_parameters()
        kwarg_name = _get_modifier_param_name(problem, param_name)
        if kwarg_name:
            problem.modify_parameters(param_name, **{kwarg_name: value})
        else:
            problem.modify_parameters(param_name)
        
        plt.subplot(rows, cols, i+1)
        
        # Create position layout (use the same layout for all graphs)
        pos = nx.spring_layout(problem.graph, seed=42) if i == 0 else pos
        
        # Draw nodes
        nx.draw_networkx_nodes(problem.graph, pos, node_size=300, node_color='skyblue')
        nx.draw_networkx_labels(problem.graph, pos, font_size=10)
        
        # Draw edges with colors based on weights
        edges = problem.graph.edges(data=True)
        weights = [data['weight'] for _, _, data in edges]
        norm = Normalize(vmin=vmin, vmax=vmax)
        
        # Draw edges with width proportional to weight
        edge_list = list(problem.graph.edges())
        nx.draw_networkx_edges(
            problem.graph, pos, 
            edgelist=edge_list,
            width=[max(0.5, abs(w) * 2) for w in weights],
            edge_color=weights,
            edge_cmap=plt.cm.coolwarm,
            edge_vmin=vmin, edge_vmax=vmax
        )
        
        plt.title(f"{param_name} = {value:.2f}")
        plt.axis('off')
    
    # Add colorbar
    plt.subplots_adjust(right=0.85)
    cax = plt.axes([0.9, 0.15, 0.02, 0.7])
    sm = ScalarMappable(cmap=plt.cm.coolwarm, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, cax=cax, label='Edge Weight')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    
    if filename:
        plt.savefig(filename)
        print(f"Visualization saved to {filename}")
    
    plt.show()
    
    # Restore original graph
    problem.graph = original_graph.copy()
    problem.build_hamiltonian()

def solution_evolution(problem, param_name: str, param_values: List[float],
                       title: str = "Solution Evolution", figsize: Tuple[int, int] = (12, 8),
                       filename: Optional[str] = None) -> None:
    """
    Visualize how the optimal solution evolves with different parameter values.
    
    Args:
        problem: Problem instance (e.g., MaxCutProblem)
        param_name: Name of the parameter modifier to apply
        param_values: List of parameter values to visualize
        title: Plot title
        figsize: Figure size
        filename: Optional filename to save visualization
    """
    n_params = len(param_values)
    
    # Store original graph to restore later
    original_graph = problem.graph.copy()
    
    plt.figure(figsize=figsize)
    
    # Calculate subplot grid dimensions
    cols = min(3, n_params)
    rows = (n_params + cols - 1) // cols
    
    # Use same node positions for all visualizations
    pos = problem.node_positions
    
    for i, value in enumerate(param_values):
        # Reset parameters and apply modifier
        problem.reset_parameters()
        kwarg_name = _get_modifier_param_name(problem, param_name)
        if kwarg_name:
            problem.modify_parameters(param_name, **{kwarg_name: value})
        else:
            problem.modify_parameters(param_name)
        
        # Solve the problem
        solution = problem.solve_classically()
        
        # Create subplot
        plt.subplot(rows, cols, i+1)
        
        # Draw nodes with partition colors
        node_colors = ['tab:blue' if solution['assignment'][n] == 0 else 'tab:red' 
                      for n in range(problem.graph.number_of_nodes())]
        nx.draw_networkx_nodes(problem.graph, pos, node_color=node_colors, node_size=300)
        nx.draw_networkx_labels(problem.graph, pos, font_size=10)
        
        # Draw edges
        cut_edges = [(u, v) for u, v in problem.graph.edges() 
                    if solution['assignment'][u] != solution['assignment'][v]]
        uncut_edges = [(u, v) for u, v in problem.graph.edges() 
                      if solution['assignment'][u] == solution['assignment'][v]]
        
        # Draw cut edges in red/dashed
        nx.draw_networkx_edges(problem.graph, pos, edgelist=cut_edges, 
                              edge_color='red', style='dashed', width=2)
        
        # Draw uncut edges in black/solid
        nx.draw_networkx_edges(problem.graph, pos, edgelist=uncut_edges, 
                              edge_color='black', style='solid', width=1)
        
        # Add edge weight labels
        edge_labels = {(u, v): f"{d['weight']:.1f}" for u, v, d in problem.graph.edges(data=True)}
        nx.draw_networkx_edge_labels(problem.graph, pos, edge_labels=edge_labels, font_size=8)
        
        plt.title(f"{param_name} = {value:.2f}\nCut Value: {solution['cut_value']:.2f}")
        plt.axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename)
        print(f"Visualization saved to {filename}")
    
    plt.show()
    
    # Restore original graph
    problem.graph = original_graph.copy()
    problem.build_hamiltonian()

def parameter_sensitivity_plot(problem, param_name: str, param_range: List[float],
                              title: str = "Parameter Sensitivity", figsize: Tuple[int, int] = (10, 6),
                              filename: Optional[str] = None) -> None:
    """
    Plot how the solution quality changes with parameter values.
    
    Args:
        problem: Problem instance (e.g., MaxCutProblem)
        param_name: Name of the parameter modifier to apply
        param_range: List of parameter values to test
        title: Plot title
        figsize: Figure size
        filename: Optional filename to save visualization
    """
    # Store original graph to restore later
    original_graph = problem.graph.copy()
    
    # Lists to store results
    cut_values = []
    
    # Calculate solution quality for each parameter value
    for value in tqdm(param_range, desc=f"Testing {param_name} values"):
        problem.reset_parameters()
        kwarg_name = _get_modifier_param_name(problem, param_name)
        if kwarg_name:
            problem.modify_parameters(param_name, **{kwarg_name: value})
        else:
            problem.modify_parameters(param_name)
        solution = problem.solve_classically()
        cut_values.append(solution['cut_value'])
    
    # Create the plot
    plt.figure(figsize=figsize)
    plt.plot(param_range, cut_values, 'o-', linewidth=2, markersize=8)
    
    # Add markers for min and max cut values
    max_idx = np.argmax(cut_values)
    min_idx = np.argmin(cut_values)
    
    plt.scatter(param_range[max_idx], cut_values[max_idx], color='green', s=100, 
               label=f'Max: {cut_values[max_idx]:.2f} at {param_range[max_idx]:.2f}', zorder=5)
    plt.scatter(param_range[min_idx], cut_values[min_idx], color='red', s=100, 
               label=f'Min: {cut_values[min_idx]:.2f} at {param_range[min_idx]:.2f}', zorder=5)
    
    plt.xlabel(f'{param_name} Value')
    plt.ylabel('Cut Value')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    if filename:
        plt.savefig(filename)
        print(f"Visualization saved to {filename}")
    
    plt.show()
    
    # Restore original graph
    problem.graph = original_graph.copy()
    problem.build_hamiltonian()

def parameter_interaction_heatmap(problem, param1_name: str, param1_range: List[float],
                                param2_name: str, param2_range: List[float],
                                title: str = "Parameter Interaction", figsize: Tuple[int, int] = (12, 10),
                                filename: Optional[str] = None) -> None:
    """
    Create a heatmap showing how two parameters interact to affect solution quality.
    
    Args:
        problem: Problem instance (e.g., MaxCutProblem)
        param1_name: Name of the first parameter modifier
        param1_range: List of values for first parameter
        param2_name: Name of the second parameter modifier
        param2_range: List of values for second parameter
        title: Plot title
        figsize: Figure size
        filename: Optional filename to save visualization
    """
    # Store original graph to restore later
    original_graph = problem.graph.copy()
    
    # Create grid for heatmap
    cut_values = np.zeros((len(param1_range), len(param2_range)))
    
    # Calculate solution quality for each parameter combination
    for i, val1 in enumerate(tqdm(param1_range, desc=f"Testing {param1_name} values")):
        for j, val2 in enumerate(param2_range):
            problem.reset_parameters()
            kwarg1_name = _get_modifier_param_name(problem, param1_name)
            kwarg2_name = _get_modifier_param_name(problem, param2_name)
            
            if kwarg1_name:
                problem.modify_parameters(param1_name, **{kwarg1_name: val1})
            else:
                problem.modify_parameters(param1_name)

            if kwarg2_name:
                problem.modify_parameters(param2_name, **{kwarg2_name: val2})
            else:
                problem.modify_parameters(param2_name)
                
            solution = problem.solve_classically()
            cut_values[i, j] = solution['cut_value']
    
    # Create heatmap
    plt.figure(figsize=figsize)
    ax = sns.heatmap(cut_values, annot=True, fmt=".2f", cmap="viridis",
                    xticklabels=[f"{val:.2f}" for val in param2_range],
                    yticklabels=[f"{val:.2f}" for val in param1_range])
    
    # Find optimal parameter combination
    max_idx = np.unravel_index(np.argmax(cut_values), cut_values.shape)
    rect = plt.Rectangle((max_idx[1], max_idx[0]), 1, 1, fill=False, 
                        edgecolor='red', lw=3, clip_on=False)
    ax.add_patch(rect)
    
    plt.xlabel(f'{param2_name} Value')
    plt.ylabel(f'{param1_name} Value')
    plt.title(title)
    
    if filename:
        plt.savefig(filename)
        print(f"Visualization saved to {filename}")
    
    plt.show()
    
    # Restore original graph
    problem.graph = original_graph.copy()
    problem.build_hamiltonian()

def solution_stability(problem, param_name: str, param_range: List[float],
                     title: str = "Solution Stability Analysis", figsize: Tuple[int, int] = (12, 6),
                     filename: Optional[str] = None) -> None:
    """
    Analyze and visualize how stable the solutions are across parameter changes.
    
    Args:
        problem: Problem instance (e.g., MaxCutProblem)
        param_name: Name of the parameter modifier to apply
        param_range: List of parameter values to test
        title: Plot title
        figsize: Figure size
        filename: Optional filename to save visualization
    """
    # Store original graph to restore later
    original_graph = problem.graph.copy()
    
    # Store solutions for each parameter value
    solutions = []
    cut_values = []
    
    # Get solutions for each parameter value
    for value in tqdm(param_range, desc=f"Testing {param_name} values"):
        problem.reset_parameters()
        kwarg_name = _get_modifier_param_name(problem, param_name)
        if kwarg_name:
            problem.modify_parameters(param_name, **{kwarg_name: value})
        else:
            problem.modify_parameters(param_name)
        solution = problem.solve_classically()
        solutions.append(solution['assignment'])
        cut_values.append(solution['cut_value'])
    
    # Calculate solution similarity matrix (Hamming distance)
    n_solutions = len(solutions)
    similarity = np.zeros((n_solutions, n_solutions))
    
    for i in range(n_solutions):
        for j in range(n_solutions):
            sol_i = [solutions[i][node] for node in range(problem.graph.number_of_nodes())]
            sol_j = [solutions[j][node] for node in range(problem.graph.number_of_nodes())]
            
            # Calculate Hamming distance (number of different assignments)
            ham_distance = sum(a != b for a, b in zip(sol_i, sol_j))
            
            # Convert to similarity (0 = different, 1 = same)
            similarity[i, j] = 1.0 - (ham_distance / len(sol_i))
    
    # Create plots - solution similarity heatmap and cut value plot
    plt.figure(figsize=figsize)
    
    # Solution similarity heatmap
    plt.subplot(1, 2, 1)
    sns.heatmap(similarity, annot=False, cmap="YlGnBu",
               xticklabels=[f"{val:.2f}" for val in param_range],
               yticklabels=[f"{val:.2f}" for val in param_range])
    plt.xlabel(f'{param_name} Value')
    plt.ylabel(f'{param_name} Value')
    plt.title('Solution Similarity Matrix')
    
    # Cut value plot
    plt.subplot(1, 2, 2)
    plt.plot(param_range, cut_values, 'o-', linewidth=2)
    
    # Add markers for solution changes
    for i in range(1, len(solutions)):
        sol_prev = [solutions[i-1][node] for node in range(problem.graph.number_of_nodes())]
        sol_curr = [solutions[i][node] for node in range(problem.graph.number_of_nodes())]
        
        # If solutions are different, mark on the plot
        if sol_prev != sol_curr:
            plt.axvline(x=param_range[i], color='r', linestyle='--', alpha=0.3)
    
    plt.xlabel(f'{param_name} Value')
    plt.ylabel('Cut Value')
    plt.title('Cut Value vs. Parameter')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if filename:
        plt.savefig(filename)
        print(f"Visualization saved to {filename}")
    
    plt.show()
    
    # Restore original graph
    problem.graph = original_graph.copy()
    problem.build_hamiltonian() 