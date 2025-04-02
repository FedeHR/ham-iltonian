#!/usr/bin/env python3
"""
Example usage of the TSPProblem class.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.problems import TSPProblem

# Create a simple TSP problem
n_cities = 4
# Distance matrix - each entry is the distance between cities i and j
distances = np.array([
    [0.0, 2.0, 5.0, 7.0],
    [2.0, 0.0, 3.0, 6.0],
    [5.0, 3.0, 0.0, 2.0],
    [7.0, 6.0, 2.0, 0.0]
])

# Define city positions for visualization
city_positions = {
    0: (0, 0),   # City 0 at (0, 0)
    1: (0, 1),   # City 1 at (0, 1)
    2: (1, 1),   # City 2 at (1, 1)
    3: (1, 0)    # City 3 at (1, 0)
}

# Create the TSPProblem instance
problem = TSPProblem(distances, city_positions, name="Example TSP")

# Print problem information
problem.print_info()

# Create the Hamiltonian
hamiltonian = problem.create_hamiltonian()
problem.print_hamiltonian(truncate=True, max_length=500)

# Solve using classical brute force
print("\nSolving classically...")
classical_solution = problem.solve_classically()
problem.print_solution("classical")

# Test some manually created solutions
print("\nTesting manual solutions:")

# Test tour 1: 0->1->2->3->0
bitstring1 = "1000" + "0100" + "0010" + "0001"
solution1 = problem.get_solution_from_bitstring(bitstring1)
problem.add_solution("manual_0123", solution1)
print("\nTour 0->1->2->3->0:")
problem.print_solution("manual_0123")

# Test tour 2: 0->1->3->2->0
bitstring2 = "1000" + "0100" + "0001" + "0010"
solution2 = problem.get_solution_from_bitstring(bitstring2)
problem.add_solution("manual_0132", solution2)
print("\nTour 0->1->3->2->0:")
problem.print_solution("manual_0132")

# Compare solutions
solution_names = ["classical", "manual_0123", "manual_0132"]
problem.print_comparison(solution_names)

# Visualize the solutions
print("\nClassical solution:")
problem.visualize_solution(classical_solution, filename="tsp_classical_solution.png")

print("\nManual solution 1 (0->1->2->3->0):")
problem.visualize_solution(solution1, filename="tsp_manual_solution1.png")

print("\nManual solution 2 (0->1->3->2->0):")
problem.visualize_solution(solution2, filename="tsp_manual_solution2.png") 