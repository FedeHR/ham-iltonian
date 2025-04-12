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

Useful for researchers and developers working at the intersection of quantum computing and combinatorial optimization problems.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
# Create a MaxCut problem instance using utility functions
from src.problems.instance_generators import create_maxcut_instance

# Create a problem with 5 nodes
maxcut = create_maxcut_instance(
    n_nodes=5,
    edge_probability=0.7,
    weight_range=(0.5, 2.0),
    graph_type="random",
    seed=42
)

# Create the corresponding Hamiltonian
maxcut.build_hamiltonian()
hamiltonian = maxcut.hamiltonian
print(hamiltonian)

# Solve classically for comparison
solution = maxcut.solve_classically()
print(solution)

# Visualize the solution
problem.visualize_solution(solution)
```

### Parameter Modification

```python
# Modify parameters of the problem, the Hamiltonian will be automatically rebuilt
problem.modify_parameters("edge_density_scaling", 1.5)

# Solve again with modified parameters
modified_solution = problem.solve_classically()
print(modified_solution)

# Reset to original parameters
problem.reset_parameters()
```

### Other problem types

```python
from src.problems.instance_generators import (
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
evaluation = problem.evaluate_bitstring(bitstring)
print(f"Cut value for {bitstring}: {evaluation['cut_value']}")
```

## References

Based on the formulations described in "Ising Formulations of Many NP Problems" by Andrew Lucas (2014).

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details. 
