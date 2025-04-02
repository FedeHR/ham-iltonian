#!/usr/bin/env python3
"""
Example usage of the TSPProblem class with different TSP variants.
Shows how to create various types of TSP instances including grid-based TSP.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.instance_generators import create_tsp_instance

# 1. Simple TSP with manual coordinates (square)
print("="*50)
print("1. Simple TSP with manual coordinates")
print("="*50)
coordinates = [
    (0, 0),     # City 0
    (0, 10),    # City 1
    (10, 10),   # City 2
    (10, 0)     # City 3
]

manual_tsp = create_tsp_instance(
    coordinates=coordinates,
    name="Simple Square TSP",
    seed=42
)

# Print problem information
manual_tsp.print_info()

# Solve using classical brute force
print("\nSolving classically...")
manual_solution = manual_tsp.solve_classically()
manual_tsp.print_solution("classical")

# Visualize the solution
print("\nManual coordinates solution:")
manual_tsp.visualize_solution(manual_solution, filename="manual_tsp_solution.png")

# 2. Grid-based TSP
print("\n" + "="*50)
print("2. Grid-based TSP")
print("="*50)

grid_tsp = create_tsp_instance(
    n_cities=16,
    tsp_type="grid",
    grid_dims=(4, 4),
    coordinate_range=(0, 30),
    name="4x4 Grid TSP",
    seed=123
)

# Print problem information
grid_tsp.print_info()

# Solve using classical brute force
print("\nSolving classically...")
grid_solution = grid_tsp.solve_classically()
grid_tsp.print_solution("classical")

# Visualize the solution
print("\nGrid TSP solution:")
grid_tsp.visualize_solution(grid_solution, filename="grid_tsp_solution.png")

# 3. Circular TSP
print("\n" + "="*50)
print("3. Circular TSP")
print("="*50)

circle_tsp = create_tsp_instance(
    n_cities=8,
    tsp_type="circle",
    coordinate_range=(0, 100),
    name="Circular TSP with 8 cities",
    seed=456
)

# Print problem information
circle_tsp.print_info()

# Solve using classical brute force
print("\nSolving classically...")
circle_solution = circle_tsp.solve_classically()
circle_tsp.print_solution("classical")

# Visualize the solution
print("\nCircular TSP solution:")
circle_tsp.visualize_solution(circle_solution, filename="circle_tsp_solution.png")

# 4. Clustered TSP
print("\n" + "="*50)
print("4. Clustered TSP")
print("="*50)

clustered_tsp = create_tsp_instance(
    n_cities=15,
    tsp_type="clustered",
    clusters=3,
    cluster_std=5.0,
    coordinate_range=(0, 100),
    name="Clustered TSP with 3 clusters",
    seed=789
)

# Print problem information
clustered_tsp.print_info()

# Solve using classical brute force
print("\nSolving classically...")
clustered_solution = clustered_tsp.solve_classically()
clustered_tsp.print_solution("classical")

# Visualize the solution
print("\nClustered TSP solution:")
clustered_tsp.visualize_solution(clustered_solution, filename="clustered_tsp_solution.png")

# 5. Asymmetric TSP
print("\n" + "="*50)
print("5. Asymmetric TSP (Random)")
print("="*50)

asymmetric_tsp = create_tsp_instance(
    n_cities=6,
    tsp_type="random",
    symmetric=False,
    coordinate_range=(0, 50),
    name="Asymmetric Random TSP",
    seed=101
)

# Print problem information
asymmetric_tsp.print_info()

# Solve using classical brute force
print("\nSolving classically...")
asymmetric_solution = asymmetric_tsp.solve_classically()
asymmetric_tsp.print_solution("classical")

# Visualize the solution
print("\nAsymmetric TSP solution:")
asymmetric_tsp.visualize_solution(asymmetric_solution, filename="asymmetric_tsp_solution.png")