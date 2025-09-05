import matplotlib.pyplot as plt
from typing import Dict, List

from hamiltonians.problems.maxcut import MaxCutProblem

def inspect_solution_landscape(
    problem: MaxCutProblem,
    modifier_name: str,
    param_name: str,
    param_values: List[float]
) -> Dict[str, List[float]]:
    """
    Inspect the solution landscape of a MaxCut problem by varying a parameter.

    Args:
        problem: The MaxCutProblem instance.
        modifier_name: The name of the modifier function to apply.
        param_name: The name of the parameter to vary in the modifier function.
        param_values: A list of values for the parameter to inspect.

    Returns:
        A dictionary containing the parameter values and the corresponding optimal cut values.
    """
    cut_values = []

    for value in param_values:
        problem.reset_parameters()
        
        # Create a dictionary for keyword arguments for the modifier
        modifier_kwargs = {param_name: value}
        problem.modify_parameters(modifier_name, **modifier_kwargs)
        
        solution = problem.solve_classically()
        cut_values.append(solution["cut_value"])

    return {
        "param_values": param_values,
        "cut_values": cut_values
    }

def plot_solution_landscape(
    landscape_data: Dict[str, List[float]],
    param_name: str,
    title: str = "MaxCut Solution Landscape",
    save_path: str = None
):
    """
    Plot the solution landscape.

    Args:
        landscape_data: Data from inspect_solution_landscape.
        param_name: The name of the parameter that was varied.
        title: The title for the plot.
        save_path: If provided, saves the plot to this file path.
    """
    param_values = landscape_data["param_values"]
    cut_values = landscape_data["cut_values"]

    plt.figure(figsize=(10, 6))
    plt.plot(param_values, cut_values, marker='o', linestyle='-')
    plt.xlabel(param_name)
    plt.ylabel("Optimal Cut Value")
    plt.title(title)
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

    plt.show()
