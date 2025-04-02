#!/usr/bin/env python3
"""
Example usage of the PortfolioProblem class.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.problems.instance_generators import create_portfolio_instance

# Create a portfolio optimization problem
returns = np.array([0.10, 0.15, 0.05, 0.12, 0.08])  # Expected returns (10%, 15%, etc.)
risk_matrix = np.array([
    [0.05, 0.02, 0.01, 0.02, 0.01],  # Covariance matrix representing risk
    [0.02, 0.06, 0.01, 0.02, 0.02],
    [0.01, 0.01, 0.04, 0.01, 0.01],
    [0.02, 0.02, 0.01, 0.07, 0.02],
    [0.01, 0.02, 0.01, 0.02, 0.05]
])
budget = 3  # Maximum number of assets to select
risk_factor = 1.0  # Weight for risk term (higher values mean more risk-averse)

# Create asset names for better visualization
asset_names = ["Tech", "Finance", "Energy", "Healthcare", "Consumer"]

# Create the PortfolioProblem instance using the utility function
problem = create_portfolio_instance(
    n_assets=len(returns),
    returns=returns,
    risk_matrix=risk_matrix,
    name="Example Portfolio"
)

# Set additional properties not covered by the factory function
problem.budget = budget
problem.risk_factor = risk_factor
problem.asset_names = asset_names

# Print problem information
problem.print_info()

# Create the Hamiltonian
hamiltonian = problem.create_hamiltonian()
problem.print_hamiltonian()

# Solve using classical brute force
print("\nSolving classically...")
classical_solution = problem.solve_classically()
problem.print_solution("classical")

# Display additional portfolio-specific information for the classical solution
if classical_solution["valid"]:
    selected_assets = [asset_names[i] for i in classical_solution['selected_assets']]
    print(f"Selected assets: {selected_assets}")
    print(f"Return/Risk ratio: {classical_solution['expected_return'] / max(0.0001, classical_solution['risk']):.4f}")

# Test some example portfolios
test_portfolios = [
    "10100",  # Tech and Energy
    "01010",  # Finance and Healthcare
    "11100",  # Tech, Finance, and Energy
    "00111",  # Energy, Healthcare, and Consumer
    "10101",  # Tech, Energy, and Consumer
]

valid_solutions = problem.test_bitstrings(test_portfolios, name_prefix="test")

# Compare valid solutions
if valid_solutions:
    solution_names = ["classical"] + valid_solutions
    problem.print_comparison(solution_names)
    
    # Find the best test solution
    best_test = max(valid_solutions, key=lambda name: problem.calculate_quality(problem.get_solution(name)))
    
    # Visualize the classical solution
    print("\nClassical solution:")
    problem.visualize_solution(classical_solution, filename="portfolio_classical_solution.png")
    
    # Visualize the best test solution
    print(f"\nBest test solution ({best_test}):")
    problem.visualize_solution(problem.get_solution(best_test), filename="portfolio_test_solution.png")
else:
    print("No valid test solutions found.")
    
    # Visualize the classical solution
    print("\nClassical solution:")
    problem.visualize_solution(classical_solution, filename="portfolio_classical_solution.png")

# Compare different risk factors
print("\nExploring different risk factors:")
risk_factors = [0.5, 1.0, 2.0, 4.0]
results = []

for rf in risk_factors:
    # Create a new problem with this risk factor
    rf_problem = create_portfolio_instance(
        n_assets=len(returns),
        returns=returns,
        risk_matrix=risk_matrix,
        name=f"Portfolio (RF={rf})"
    )
    
    # Set additional properties not covered by the factory function
    rf_problem.budget = budget
    rf_problem.risk_factor = rf
    rf_problem.asset_names = asset_names
    
    # Solve it
    solution = rf_problem.solve_classically()
    
    # Record results
    selected_assets = [asset_names[i] for i in solution['selected_assets']]
    results.append({
        "risk_factor": rf,
        "selected_assets": selected_assets,
        "return": solution['expected_return'],
        "risk": solution['risk'],
        "ratio": solution['expected_return'] / max(0.0001, solution['risk'])
    })
    
    print(f"\nRisk Factor: {rf}")
    rf_problem.print_solution("classical")
    print(f"Selected assets: {selected_assets}")
    print(f"Return/Risk ratio: {solution['expected_return'] / max(0.0001, solution['risk']):.4f}")
    
    # Visualize
    rf_problem.visualize_solution(solution, filename=f"portfolio_rf_{rf}.png") 