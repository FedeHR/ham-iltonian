#!/usr/bin/env python3
"""
Example usage of the KnapsackProblem class.
"""
import sys
import os
# Add the parent directory to path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.problems import KnapsackProblem

values = [10, 15, 7, 12, 8]
weights = [3, 5, 2, 4, 3]
capacity = 10
problem = KnapsackProblem(values, weights, capacity, name="Example Knapsack")
problem.print_info()

hamiltonian = problem.create_hamiltonian(penalty=20.0)
problem.print_hamiltonian()

# Convert to PennyLane Hamiltonian
problem.print_pennylane_hamiltonian()

# Solve using classical brute force
print("\nSolving classically...")
classical_solution = problem.solve_classically()
problem.print_solution("classical")
print(f"Total weight: {classical_solution['total_weight']} / {capacity}")

# Test some example solutions
test_solutions = [
    "10100",  # Items 0 and 2
    "01010",  # Items 1 and 3
    "11000",  # Items 0 and 1
    "00111",  # Items 2, 3, and 4
]

valid_solutions = problem.test_bitstrings(test_solutions, name_prefix="test")

# Compare valid solutions with the classical solution
if valid_solutions:
    solution_names = ["classical"] + valid_solutions
    problem.print_comparison(solution_names)
    
    # Find the best test solution
    best_test = max(valid_solutions, key=lambda name: problem.calculate_quality(problem.get_solution(name)))
    
    # Visualize the classical solution
    print("\nClassical solution:")
    problem.visualize_solution(classical_solution, filename="knapsack_classical_solution.png")
    
    # Visualize the best test solution
    print(f"\nBest test solution ({best_test}):")
    problem.visualize_solution(problem.get_solution(best_test), filename="knapsack_test_solution.png")
else:
    print("No valid test solutions found.")
    
    # Visualize the classical solution
    print("\nClassical solution:")
    problem.visualize_solution(classical_solution, filename="knapsack_classical_solution.png") 