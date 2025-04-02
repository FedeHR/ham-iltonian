#!/usr/bin/env python3
"""
Example usage of the MaxCutProblem class.
"""
import networkx as nx
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.problems import MaxCutProblem

# Create a simple weighted graph
G = nx.Graph()
G.add_weighted_edges_from([
    (0, 1, 1.0),
    (0, 2, 2.0),
    (1, 2, 1.0),
    (1, 3, 2.0),
    (2, 3, 1.0)
])

# Create the MaxCutProblem instance
problem = MaxCutProblem(G, name="Example MaxCut")

# Print problem information
problem.print_info()

# Create the Hamiltonian
hamiltonian = problem.create_hamiltonian()
problem.print_hamiltonian()

# Convert to PennyLane Hamiltonian
problem.print_pennylane_hamiltonian()

# Solve using classical brute force
print("\nSolving classically...")
classical_solution = problem.solve_classically()
problem.print_solution("classical")

# Set up a simple QAOA circuit
def qaoa_layer(gamma, beta):
    # Problem unitary
    qml.ApproxTimeEvolution(hamiltonian.to_pennylane(), gamma, 1)
    # Mixer unitary
    for q in range(G.number_of_nodes()):
        qml.RX(2 * beta, wires=q)

# Create a quantum device and QNode
dev = qml.device("default.qubit", wires=G.number_of_nodes())

@qml.qnode(dev)
def cost_function(params):
    # Initialize in the plus state
    for q in range(G.number_of_nodes()):
        qml.Hadamard(wires=q)
        
    # Apply QAOA layers
    for gamma, beta in zip(params[0], params[1]):
        qaoa_layer(gamma, beta)
        
    # Return the expectation value
    return qml.expval(hamiltonian.to_pennylane())

# Initialize QAOA parameters
layers = 2
np.random.seed(42)
params = np.random.uniform(0, np.pi, (2, layers))

print("\nRunning QAOA...")
print(f"QAOA parameters - Gammas: {params[0]}, Betas: {params[1]}")
print(f"Initial QAOA cost: {cost_function(params)}")

# Get probabilities of each computational basis state
@qml.qnode(dev)
def get_state(params):
    # Initialize in the plus state
    for q in range(G.number_of_nodes()):
        qml.Hadamard(wires=q)
        
    # Apply QAOA layers
    for gamma, beta in zip(params[0], params[1]):
        qaoa_layer(gamma, beta)
        
    # Return the state probabilities
    return qml.probs(wires=range(G.number_of_nodes()))

# Get the most probable bitstring from QAOA
probs = get_state(params)
qaoa_bitstring = np.binary_repr(np.argmax(probs), width=G.number_of_nodes())
print(f"QAOA most probable bitstring: {qaoa_bitstring}")

# Get the QAOA solution
qaoa_solution = problem.get_solution_from_bitstring(qaoa_bitstring)
problem.add_solution("qaoa", qaoa_solution)
problem.print_solution("qaoa")

# Compare solutions
problem.print_comparison(["classical", "qaoa"])

# Visualize the solutions
print("\nClassical solution:")
problem.visualize_solution(classical_solution, filename="maxcut_classical_solution.png")

print("\nQAOA solution:")
problem.visualize_solution(qaoa_solution, filename="maxcut_qaoa_solution.png") 