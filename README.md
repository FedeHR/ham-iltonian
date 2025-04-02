# 🍖⚛️ ham-iltonian
A Python library for encoding a variety of combinatorial optimization problems into quantum Hamiltonians. The Hamiltonians have the ability to be parametrized, reflecting the effect of different factors, such as time, in challenges like TSP, MaxCut, Knapsack and Portfolio Optimization.

Key features:
- 🔄 Automatic conversion of classical problems to quantum Hamiltonians
- 📊 Support for multiple optimization problems (MaxCut, TSP, Knapsack, Portfolio Optimization)
- 🔌 Integration with PennyLane for quantum simulations
- 📈 Classical solvers included for benchmarking
- 🎨 Built-in visualization tools for solutions
- ⚡ Parametrized Hamiltonian support for dynamic problems (dynamic time, risk factors, or any other desired parameter)
- 🧩 Easy problem instance generation with utility functions

Perfect for researchers and developers working at the intersection of quantum computing and optimization problems.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
# Create a MaxCut problem instance using utility functions
from src.problems.instance_generators import create_maxcut_instance

# Create a problem with 5 nodes
problem = create_maxcut_instance(
    n_nodes=5,
    edge_probability=0.7,
    weight_range=(0.5, 2.0),
    graph_type="random",
    seed=42
)

# Create the corresponding Hamiltonian
hamiltonian = problem.create_hamiltonian()
print(hamiltonian)

# Solve classically for comparison
solution = problem.solve_classically()
problem.print_solution("classical")
```

### Other problem types

```python
from src.problems.instance_generators import (
    create_tsp_instance,
    create_knapsack_instance,
    create_portfolio_instance
)

# TSP with 5 cities
tsp = create_tsp_instance(n_cities=5, seed=42)

# Knapsack with 10 items
knapsack = create_knapsack_instance(n_items=10, max_weight=100.0, seed=42)

# Portfolio with 5 assets
portfolio = create_portfolio_instance(n_assets=5, seed=42)
```

## References

Based on the formulations described in "Ising Formulations of Many NP Problems" by Andrew Lucas (2014).

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details. 
