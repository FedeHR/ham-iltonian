# 🍖⚛️ ham-iltonian
A Python library for encoding a variety of combinatorial optimization problems into quantum Hamiltonians. The Hamiltonians have the ability to be parametrized, reflecting the effect of different factors, such as time, in challenges like TSP, MaxCut, Knapsack and Portfolio Optimization.

Key features:
- 🔄 Automatic conversion of classical problems to quantum Hamiltonians
- 📊 Support for multiple optimization problems (MaxCut, TSP, Knapsack, Portfolio Optimization, Number Partitioning)
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

### Parametrized Hamiltonians

The library supports parametrized Hamiltonians, allowing you to model problems that change based on external parameters. For example, in TSP, travel distances might vary depending on the time of day due to traffic.

```python
from src.problems.instance_generators import create_tsp_instance
import numpy as np
import matplotlib.pyplot as plt

# Create a TSP instance
coordinates = [
    (0, 0),    # City 0
    (0, 10),   # City 1
    (10, 10),  # City 2
    (10, 0)    # City 3
]
tsp = create_tsp_instance(coordinates=coordinates, name="Time-Dependent TSP")

# Create a time-dependent Hamiltonian
time_dependent_hamiltonian = tsp.create_hamiltonian(time_dependent=True)

# Solve the problem at different times of day
print("Finding optimal routes at different times of day...")

# Morning rush hour (8:00 AM)
morning_solution = tsp.solve_with_parameters({'time': 8.0}, solution_name="morning")
print(f"Morning route distance: {morning_solution['total_distance']:.2f}")

# Mid-day (12:00 PM)
midday_solution = tsp.solve_with_parameters({'time': 12.0}, solution_name="midday")
print(f"Mid-day route distance: {midday_solution['total_distance']:.2f}")

# Evening rush hour (5:00 PM)
evening_solution = tsp.solve_with_parameters({'time': 17.0}, solution_name="evening")
print(f"Evening route distance: {evening_solution['total_distance']:.2f}")

# Compare solutions
tsp.print_comparison(["morning", "midday", "evening"])

# Visualize the different routes
tsp.visualize_solution(morning_solution, filename="tsp_morning_solution.png")
tsp.visualize_solution(midday_solution, filename="tsp_midday_solution.png")
tsp.visualize_solution(evening_solution, filename="tsp_evening_solution.png")
```

## References

Based on the formulations described in "Ising Formulations of Many NP Problems" by Andrew Lucas (2014).

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details. 
