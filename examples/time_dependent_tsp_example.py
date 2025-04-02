#!/usr/bin/env python3
"""
Example of time-dependent TSP where the optimal route changes based on time of day.
This simulates how traffic conditions might affect travel times at different hours.
"""
import numpy as np
import matplotlib
matplotlib.use('agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.problems import TSPProblem

# Create a simple city layout - in this case, a grid of cities
def create_grid_cities(n, m):
    """Create an n x m grid of cities."""
    cities = []
    names = []
    positions = []
    
    city_idx = 0
    for i in range(n):
        for j in range(m):
            cities.append((i, j))
            names.append(f"City {city_idx}")
            positions.append([i, j])
            city_idx += 1
    
    return cities, names, np.array(positions)

# Create a distance matrix with some roads more affected by traffic than others
def create_distance_matrix(cities, traffic_sensitivity=None):
    """
    Create a distance matrix between cities.
    
    Args:
        cities: List of (x, y) city coordinates
        traffic_sensitivity: Optional dict mapping (city1, city2) pairs to 
                            traffic sensitivity (0-1, higher means more affected by time)
    """
    n = len(cities)
    distances = np.zeros((n, n))
    
    if traffic_sensitivity is None:
        traffic_sensitivity = {}
    
    for i in range(n):
        for j in range(i+1, n):
            # Calculate Euclidean distance
            city1 = cities[i]
            city2 = cities[j]
            
            # Base distance
            dx = city2[0] - city1[0]
            dy = city2[1] - city1[1]
            dist = np.sqrt(dx*dx + dy*dy)
            
            # Store distances symmetrically
            distances[i, j] = dist
            distances[j, i] = dist
    
    return distances

def mark_time_sensitive_roads(problem, sensitivities):
    """Mark roads that are sensitive to time (e.g., highways with traffic)."""
    n_cities = problem.n_cities
    plt.figure(figsize=(10, 8))
    
    # Draw city points
    plt.scatter(problem.city_positions[:, 0], problem.city_positions[:, 1], c='blue', s=100, zorder=10)
    
    # Label cities
    for i, (x, y) in enumerate(problem.city_positions):
        plt.text(x, y + 0.05, problem.city_names[i], ha='center', va='center', fontsize=12)
    
    # Draw all roads with their traffic sensitivity
    for i in range(n_cities):
        for j in range(i+1, n_cities):
            x1, y1 = problem.city_positions[i]
            x2, y2 = problem.city_positions[j]
            
            sensitivity = sensitivities.get((i, j), 0.0)
            if sensitivity > 0:
                # Color based on sensitivity (red = high, yellow = medium, green = low)
                if sensitivity > 0.7:
                    color = 'red'
                    width = 2.5
                    label = "High traffic"
                elif sensitivity > 0.3:
                    color = 'orange'
                    width = 2.0
                    label = "Medium traffic"
                else:
                    color = 'green'
                    width = 1.5
                    label = "Low traffic"
                
                plt.plot([x1, x2], [y1, y2], color=color, linewidth=width, alpha=0.7)
    
    plt.title("Road Network with Traffic Sensitivity")
    plt.grid(alpha=0.3)
    
    # Add a simple legend
    plt.plot([], [], 'red', linewidth=2.5, label='High traffic sensitivity')
    plt.plot([], [], 'orange', linewidth=2.0, label='Medium traffic sensitivity')
    plt.plot([], [], 'green', linewidth=1.5, label='Low traffic sensitivity')
    plt.legend()
    
    plt.savefig("tsp_road_network.png")
    print("Road network visualization saved to tsp_road_network.png")
    plt.close()

# Create a sample city layout (3x3 grid = 9 cities)
cities, city_names, city_positions = create_grid_cities(3, 3)

# Define which roads are sensitive to traffic
# Higher values mean more affected by time of day
traffic_sensitivities = {
    # Main highways (horizontal and vertical)
    (0, 1): 0.8, (1, 2): 0.8,  # East-West highway
    (3, 4): 0.8, (4, 5): 0.8,  # East-West highway
    (6, 7): 0.5, (7, 8): 0.5,  # East-West secondary road
    (0, 3): 0.8, (3, 6): 0.8,  # North-South highway
    (1, 4): 0.8, (4, 7): 0.8,  # North-South highway
    (2, 5): 0.5, (5, 8): 0.5,  # North-South secondary road
    
    # Diagonal roads (less affected by traffic)
    (0, 4): 0.3, (4, 8): 0.3,  # Diagonal secondary roads
    (2, 4): 0.3, (4, 6): 0.3,  # Diagonal secondary roads
    
    # Make sure we have symmetrical data
    (1, 0): 0.8, (2, 1): 0.8,
    (4, 3): 0.8, (5, 4): 0.8,
    (7, 6): 0.5, (8, 7): 0.5,
    (3, 0): 0.8, (6, 3): 0.8,
    (4, 1): 0.8, (7, 4): 0.8,
    (5, 2): 0.5, (8, 5): 0.5,
    (4, 0): 0.3, (8, 4): 0.3,
    (4, 2): 0.3, (6, 4): 0.3,
}

# Create basic distances
distances = create_distance_matrix(cities)

# Create a custom city name list to make it more realistic
custom_names = [
    "Downtown", "Mall", "Airport", 
    "University", "Central Park", "Industrial Zone",
    "Suburbs", "Beach", "Mountains"
]

# Create the TSP Problem
problem = TSPProblem(
    distances=distances,
    city_names=custom_names,
    city_positions=city_positions,
    name="Time-Dependent TSP"
)

# Print problem information
problem.print_info()

# Visualize the road network with traffic sensitivity
mark_time_sensitive_roads(problem, traffic_sensitivities)

# Create time-dependent Hamiltonian
print("\nCreating time-dependent Hamiltonian...")
hamiltonian = problem.create_hamiltonian(time_dependent=True)
problem.print_hamiltonian(truncate=True)

# Solve for morning rush hour (8:00 AM)
print("\nSolving for morning rush hour (8:00 AM)...")
morning_solution = problem.solve_with_parameters({'time': 8.0}, solution_name="morning")
problem.print_solution("morning")
problem.visualize_solution(morning_solution, filename="tsp_morning_solution.png")

# Solve for mid-day (12:00 PM)
print("\nSolving for mid-day (12:00 PM)...")
midday_solution = problem.solve_with_parameters({'time': 12.0}, solution_name="midday")
problem.print_solution("midday")
problem.visualize_solution(midday_solution, filename="tsp_midday_solution.png")

# Solve for evening rush hour (5:00 PM)
print("\nSolving for evening rush hour (5:00 PM)...")
evening_solution = problem.solve_with_parameters({'time': 17.0}, solution_name="evening")
problem.print_solution("evening")
problem.visualize_solution(evening_solution, filename="tsp_evening_solution.png")

# Solve for night (11:00 PM)
print("\nSolving for night (11:00 PM)...")
night_solution = problem.solve_with_parameters({'time': 23.0}, solution_name="night")
problem.print_solution("night")
problem.visualize_solution(night_solution, filename="tsp_night_solution.png")

# Compare all solutions
print("\nComparing solutions at different times:")
problem.print_comparison(["morning", "midday", "evening", "night"])

# Create visualization showing the effect of time
problem.visualize_time_effect(filename="tsp_time_effect.png")

# Create a detailed time analysis
print("\nDetailed time analysis:")
hours = np.arange(0, 24, 2)  # Every 2 hours
problem.visualize_parameter_effect('time', hours, filename="tsp_time_analysis.png")

print("\nAll visualizations have been saved. Check the output directory for PNG files.") 