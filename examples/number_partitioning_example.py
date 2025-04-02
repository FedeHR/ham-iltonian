#!/usr/bin/env python3
"""
Example usage of the NumberPartitioningProblem class.
"""
import numpy as np
import matplotlib
matplotlib.use('agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.problems import NumberPartitioningProblem

# Create a set of numbers to partition
numbers = [5, 8, 13, 4, 7, 10]

# Create the NumberPartitioningProblem instance
problem = NumberPartitioningProblem(numbers, name="Example Number Partitioning")

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

# Test some example partitions
test_bitstrings = [
    "101010",  # Alternating assignment
    "111000",  # First half/second half split
    "100100",  # Select the largest numbers
    "011011",  # Select all but the largest numbers
]

valid_solutions = problem.test_bitstrings(test_bitstrings, name_prefix="test")

# Compare solutions with names if there are at least two solutions
if len(valid_solutions) >= 1:
    solution_names = ["classical"] + valid_solutions
    problem.print_comparison(solution_names)

# Find the best test solution
best_test = max(valid_solutions, key=lambda name: problem.calculate_quality(problem.get_solution(name))) if valid_solutions else None

# Visualize the classical solution
print("\nClassical solution:")
problem.visualize_solution(classical_solution, filename="number_partitioning_classical_solution.png")

# Visualize the best test solution if it exists
if best_test:
    print(f"\nBest test solution ({best_test}):")
    problem.visualize_solution(problem.get_solution(best_test), filename="number_partitioning_test_solution.png")
