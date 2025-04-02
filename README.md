# 🍖⚛️ ham-iltonian
A Python library for encoding a variety of combinatorial optimization problems into quantum Hamiltonians. The Hamiltonians have the ability to be parametrized, reflecting the effect of different factors, such as time, in challenges like TSP, MaxCut, Knapsack and Portfolio Optimization.

Key features:
- 🔄 Automatic conversion of classical problems to quantum Hamiltonians
- 📊 Support for multiple optimization problems (MaxCut, TSP, Knapsack, Portfolio Optimization)
- 🔌 Integration with PennyLane for quantum simulations
- 📈 Classical solvers included for benchmarking
- 🎨 Built-in visualization tools for solutions
- ⚡ Parametrized Hamiltonian support for dynamic problems (dynamic time, risk factors, or any other desired parameter)

Perfect for researchers and developers working at the intersection of quantum computing and optimization problems.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
# Using MaxCut as an example
import networkx as nx
from src.hamiltonians.maxcut import create_maxcut_hamiltonian

# Create a graph for the MaxCut problem
G = nx.Graph()
G.add_weighted_edges_from([(0, 1, 1.0), (1, 2, 0.5), (0, 2, 0.8)])

# Create the corresponding Hamiltonian
hamiltonian = create_maxcut_hamiltonian(G)
print(hamiltonian)
```

## References

Based on the formulations described in "Ising Formulations of Many NP Problems" by Andrew Lucas (2014).

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details. 
