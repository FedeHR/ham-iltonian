#!/usr/bin/env python3
"""
Example usage of the KnapsackProblem class with the new utility functions.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.problems.instance_generators import create_knapsack_instance

# Create a small knapsack problem using predefined values and weights
values = np.array([10, 20, 15, 25, 30])
weights = np.array([1, 3, 2, 5, 4])
max_weight = 8

# Create the problem instance
problem = create_knapsack_instance(
    n_items=len(values),
    max_weight=max_weight, 
    name="Simple Knapsack Example"
)

# Override the random values and weights with our predefined ones
problem.values = values
problem.weights = weights

# Print problem information
problem.print_info()

# Create the Hamiltonian
hamiltonian = problem.create_hamiltonian()
problem.print_hamiltonian()

# Solve using classical brute force
print("\nSolving classically...")
classical_solution = problem.solve_classically()
problem.print_solution("classical")

# Visualize the solution
print("\nClassical solution:")
problem.visualize_solution(classical_solution, filename="knapsack_classical_solution.png")

# Create another random knapsack instance using the utility function
print("\n\n" + "="*50)
print("Creating a random knapsack instance")
print("="*50)

random_problem = create_knapsack_instance(
    n_items=8,
    max_weight=50,
    value_range=(10, 100),
    weight_range=(5, 25),
    name="Random Knapsack Problem",
    seed=123
)

# Print problem information
random_problem.print_info()

# Solve using classical brute force
print("\nSolving classically...")
random_solution = random_problem.solve_classically()
random_problem.print_solution("classical")

# Visualize the solution
print("\nRandom knapsack solution:")
random_problem.visualize_solution(random_solution, filename="random_knapsack_solution.png")