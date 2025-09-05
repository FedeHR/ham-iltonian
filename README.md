# 🍖⚛️ ham-iltonian
A Python library for encoding a variety of combinatorial optimization problems in the form of Ising Hamiltonians, based on the formulations described in "Ising Formulations of Many NP Problems" by Andrew Lucas (2014). The Hamiltonians have the ability to be parametrized, reflecting the effect of different factors, such as time, in challenges like TSP, MaxCut, Knapsack and Portfolio Optimization.

Key features:
- 🔄 Automatic conversion of classical problems to Ising Hamiltonians
- 📊 Support for multiple optimization problems (MaxCut, TSP, Knapsack, Portfolio Optimization, Number Partitioning)
- 🔌 Integration with PennyLane for quantum simulations
- 📈 Classical solvers included for benchmarking
- 🎨 Built-in visualization tools for solutions
- ⚡ Parametrized Hamiltonian support for dynamic problems (dynamic time, risk factors, or any other desired parameter)
- 🧩 Easy problem instance generation with utility functions
- 📊 Parameter effect visualization tools to analyze how modifications influence solutions

Useful for researchers and developers working at the intersection of quantum computing and combinatorial optimization problems.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### MaxCut Example

```python
# Create a MaxCut problem instance
from hamiltonians import create_maxcut_instance

maxcut = create_maxcut_instance(
    n_nodes=5,
    edge_probability=0.7,
    weight_range=(0.5, 2.0),
    graph_type="random",
    seed=42
)

# Build and print the Hamiltonian
maxcut.build_hamiltonian()
print("Original Hamiltonian:")
maxcut.print_hamiltonian(truncate=True)

# Modify parameters. The Hamiltonian is automatically rebuilt after modification.
# THe modify_parameter functions takes a modifier function (e.g. linear scaling) and a global parameter for the modifier function
maxcut.modify_parameters("linear", 0.5)
print("\nModified Hamiltonian (with linear modifier):")
maxcut.print_hamiltonian(truncate=True)

# Solve classically for comparison
solution = maxcut.solve_classically()
print(f"\nSolution: {solution}")

# Visualize the solution
maxcut.visualize_solution(solution)
```

### Knapsack Example

```python
# Create a Knapsack problem instance
from hamiltonians import create_knapsack_instance

knapsack = create_knapsack_instance(n_items=5, max_weight=50.0, seed=42)

# The Hamiltonian is built upon instantiation for Knapsack
print("Original Hamiltonian:")
knapsack.print_hamiltonian(truncate=True, max_length=200)

# Modify item values. The Hamiltonian is automatically rebuilt.
# Scale all item values by a factor of 1.5
knapsack.modify_parameters("scale_values", 1.5)
print("\nModified Hamiltonian (with scaled values):")
knapsack.print_hamiltonian(truncate=True, max_length=200)

# Solve classically for comparison
solution = knapsack.solve_classically()
print(f"\nSolution: {solution}")

# Visualize the solution
knapsack.visualize_solution(solution)
```

### Visualizing Parameter Effects

The library includes tools to visualize how different parameter values affect problem structure and solutions:

```python
from hamiltonians import parameter_sensitivity_plot, solution_evolution
from hamiltonians import create_maxcut_instance

# Create a problem with 5 nodes
maxcut = create_maxcut_instance(
    n_nodes=5,
    edge_probability=0.7,
    weight_range=(0.5, 2.0),
    graph_type="random",
    seed=42
)

# See how solution quality changes across parameter values
parameter_sensitivity_plot(
    maxcut,
    "linear",
    param_range=[0.0, 0.2, 0.4, 0.6, 0.8],
    title="Cut Value vs Linear Modifier"
)

# Visualize how solutions evolve with different parameter values
solution_evolution(
    maxcut,
    "edge_density_scaling",
    param_values=[0.0, 1.0, 2.0],
    title="Solution Evolution with Edge Density Scaling"
)
```

For more examples, see `src/examples/parameter_analysis.py`.

### Other problem types

```python
from hamiltonians import (
    create_tsp_instance,
    create_knapsack_instance,
    create_portfolio_instance,
    create_number_partitioning_instance
)

# TSP with 5 cities
tsp = create_tsp_instance(n_cities=5, seed=42)

# Knapsack with 10 items
knapsack = create_knapsack_instance(n_items=10, max_weight=100.0, seed=42)

# Portfolio with 5 assets
portfolio = create_portfolio_instance(n_assets=5, seed=42)

# Number Partitioning with 8 numbers
numbers = [5, 8, 13, 4, 7, 10, 2, 3]
number_partitioning = create_number_partitioning_instance(numbers=numbers, seed=42)
```

### Evaluating Solutions

You can evaluate any potential solution using the provided methods:

```python
# Evaluate a specific bit assignment
bitstring = "10110"  # For a 5-node MaxCut problem
evaluation = maxcut.evaluate_bitstring(bitstring)
print(f"Cut value for {bitstring}: {evaluation['cut_value']}")
```

## References

Based on the formulations described in "Ising Formulations of Many NP Problems" by Andrew Lucas (2014).

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details. 
