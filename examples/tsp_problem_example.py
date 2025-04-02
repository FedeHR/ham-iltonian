#!/usr/bin/env python3
"""
Example usage of the TSPProblem class with the new utility functions.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.problems.instance_generators import create_tsp_instance

# Create a TSP problem instance using the utility function
# We'll create a problem with 4 cities at specific coordinates
coordinates = [
    (0, 0),     # City 0
    (0, 10),    # City 1
    (10, 10),   # City 2
    (10, 0)     # City 3
]

problem = create_tsp_instance(
    coordinates=coordinates,
    name="Simple TSP Example",
    seed=42
)

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

# Visualize the solution
print("\nClassical solution:")
problem.visualize_solution(classical_solution, filename="tsp_classical_solution.png")

# Let's create another TSP instance with random cities
print("\n\n" + "="*50)
print("Creating a random TSP instance")
print("="*50)

random_tsp = create_tsp_instance(
    n_cities=6,
    coordinate_range=(0, 100),
    name="Random 6-city TSP",
    seed=123
)

# Print problem information
random_tsp.print_info()

# Solve using classical brute force
print("\nSolving classically...")
random_solution = random_tsp.solve_classically()
random_tsp.print_solution("classical")

# Visualize the solution
print("\nRandom TSP solution:")
random_tsp.visualize_solution(random_solution, filename="random_tsp_solution.png")