#!/usr/bin/env python3
"""
Example usage of the NumberPartitioningProblem class.

NOTE: This example is a stub for future implementation. 
The NumberPartitioningProblem class is not yet implemented.
"""
import numpy as np
import matplotlib
matplotlib.use('agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("="*80)
print("NOTE: The NumberPartitioningProblem is not yet implemented.")
print("This example file is a placeholder for future implementation.")
print("="*80)
print("\nTo implement this feature, you need to:")
print("1. Create a new file src/problems/number_partitioning.py with the NumberPartitioningProblem class")
print("2. Add the class to src/problems/__init__.py")
print("3. Complete the create_number_partitioning_instance function in src/problems/instance_generators.py")
print("\nExiting now.")
sys.exit(0)

# The rest of this file is for future implementation
# from src.problems.instance_generators import create_number_partitioning_instance

# # Create a set of numbers to partition
# numbers = [5, 8, 13, 4, 7, 10]

# # Create the NumberPartitioningProblem instance using the utility function
# problem = create_number_partitioning_instance(
#     numbers=numbers,
#     name="Example Number Partitioning"
# )

# # Print problem information
# problem.print_info()

# # Create the Hamiltonian
# hamiltonian = problem.create_hamiltonian()
# problem.print_hamiltonian()

# # Convert to PennyLane Hamiltonian
# problem.print_pennylane_hamiltonian()

# # Solve using classical brute force
# print("\nSolving classically...")
# classical_solution = problem.solve_classically()
# problem.print_solution("classical")

# # Test some example partitions
# test_bitstrings = [
#     "101010",  # Alternating assignment
#     "111000",  # First half/second half split
#     "100100",  # Select the largest numbers
#     "011011",  # Select all but the largest numbers
# ]

# valid_solutions = problem.test_bitstrings(test_bitstrings, name_prefix="test")

# # Compare solutions with names if there are at least two solutions
# if len(valid_solutions) >= 1:
#     solution_names = ["classical"] + valid_solutions
#     problem.print_comparison(solution_names)

# # Find the best test solution
# best_test = max(valid_solutions, key=lambda name: problem.calculate_quality(problem.get_solution(name))) if valid_solutions else None

# # Visualize the classical solution
# print("\nClassical solution:")
# problem.visualize_solution(classical_solution, filename="number_partitioning_classical_solution.png")

# # Visualize the best test solution if it exists
# if best_test:
#     print(f"\nBest test solution ({best_test}):")
#     problem.visualize_solution(problem.get_solution(best_test), filename="number_partitioning_test_solution.png")
