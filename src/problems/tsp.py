"""
Traveling Salesman Problem (TSP) implementation.
"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

from .base import Problem
from ..hamiltonians.tsp import create_tsp_hamiltonian, get_tsp_solution
from ..utils.classical_solvers import solve_tsp_brute_force

class TSPProblem(Problem):
    """
    Traveling Salesman Problem representation.
    
    The TSP seeks to find the shortest possible route that visits each city exactly once 
    and returns to the starting city.
    """
    
    def __init__(
        self, 
        distances: np.ndarray, 
        city_names: Optional[List[str]] = None,
        city_positions: Optional[np.ndarray] = None,
        name: str = "TSP"
    ):
        """
        Initialize a TSP problem.
        
        Args:
            distances: Distance matrix between cities (n x n)
            city_names: Optional list of city names
            city_positions: Optional array of city positions for visualization
            name: Name of the problem instance
        """
        super().__init__(name)
        self.distances = distances
        self.n_cities = distances.shape[0]
        
        # Validate distance matrix
        if distances.shape[0] != distances.shape[1]:
            raise ValueError("Distance matrix must be square")
        assert np.all(np.diag(distances) == 0), "Diagonal of distance matrix should be 0"
        
        self.metadata["n_cities"] = self.n_cities
        self.metadata["distance_range"] = [float(np.min(distances)), float(np.max(distances))]
        self.metadata["avg_distance"] = float(np.mean(distances))
        
        # Set city names if not provided
        if city_names is None:
            self.city_names = [f"City {i}" for i in range(self.n_cities)]
        else:
            if len(city_names) != self.n_cities:
                raise ValueError("Number of city names must match number of cities")
            self.city_names = city_names
        
        # Set city positions if not provided (for visualization)
        if city_positions is None:
            # Generate random positions in 2D plane
            self.city_positions = np.random.rand(self.n_cities, 2)
        else:
            if city_positions.shape[0] != self.n_cities:
                raise ValueError("Number of city positions must match number of cities")
            self.city_positions = city_positions
    
    def create_hamiltonian(self, A: float = None, B: float = None, time_dependent: bool = False) -> Any:
        """
        Create the Hamiltonian for this TSP problem.
        
        Args:
            A: Coefficient for the constraint terms (default: automatically calculated)
            B: Coefficient for the distance term (default: 1.0)
            time_dependent: Whether to create a time-dependent Hamiltonian
            
        Returns:
            Hamiltonian for the TSP problem
        """
        # Set default penalty coefficients if not provided
        if A is None:
            A = np.max(self.distances) * 2
        if B is None:
            B = np.max(self.distances) * 2
            
        self._hamiltonian = create_tsp_hamiltonian(
            self.distances, 
            A=A, 
            B=B,
            time_dependent=time_dependent
        )
        return self._hamiltonian
    
    def solve_classically(self, time_of_day: Optional[float] = None) -> Dict[str, Any]:
        """
        Solve the TSP problem using classical methods.
        
        Args:
            time_of_day: Optional time of day (0-24 hours) affecting traffic
            
        Returns:
            Dictionary with solution details
        """
        # Calculate time factor if time_of_day is provided
        time_factor = 1.0
        if time_of_day is not None:
            time_factor = self._calculate_time_factor(time_of_day)
        
        # Apply time factor to distances
        time_adjusted_distances = self.distances * time_factor
        
        # Solve using brute force
        solution = solve_tsp_brute_force(time_adjusted_distances)
        
        # Store solution
        self.add_solution("classical", solution)
        
        # Add time information if provided
        if time_of_day is not None:
            solution["time_of_day"] = time_of_day
            solution["time_factor"] = time_factor
        
        return solution
    
    def solve_with_parameters(self, param_values: Dict[str, Any], solution_name: str = None, **kwargs) -> Dict[str, Any]:
        """
        Solve the problem with specific parameter values.
        
        Args:
            param_values: Dictionary of parameter values to apply
            solution_name: Name to assign to the solution (default: based on param values)
            **kwargs: Problem-specific parameters
            
        Returns:
            Dictionary with solution details
        """
        # Store current parameter values
        self.parameters = param_values.copy()
        
        # Create the parameterized Hamiltonian
        self.create_parameterized_hamiltonian(param_values, **kwargs)
        
        # Calculate time factor if time is provided
        if 'time' in param_values:
            time_of_day = param_values['time']
            time_factor = self._calculate_time_factor(time_of_day)
            kwargs['time_of_day'] = time_of_day
        
        # Solve the problem
        solution = self.solve_classically(**kwargs)
        
        # If time factor wasn't added by solve_classically, add it here
        if 'time' in param_values and 'time_factor' not in solution:
            solution['time_factor'] = self._calculate_time_factor(param_values['time'])
            solution['time_of_day'] = param_values['time']
        
        # If solution_name wasn't provided, generate one from parameters
        if solution_name is None:
            param_str = "_".join([f"{k}_{v}" for k, v in param_values.items()])
            solution_name = f"param_{param_str}"
        
        # Add parameter values to the solution
        solution["parameters"] = param_values.copy()
        
        # Store the solution
        self.add_solution(solution_name, solution)
        
        return solution
    
    def get_solution_from_bitstring(self, bitstring: str) -> Dict[str, Any]:
        """
        Get the TSP solution from a bitstring.
        
        Args:
            bitstring: Binary string representation of the solution
            
        Returns:
            Dictionary with solution details
        """
        # Use current time factor if available in parameters
        time_factor = 1.0
        if 'time' in self.parameters:
            time_factor = self._calculate_time_factor(self.parameters['time'])
        
        # Get the solution using the time-adjusted factor
        solution = get_tsp_solution(
            bitstring, 
            self.n_cities, 
            self.distances,
            time_factor=time_factor
        )
        
        # Add time information if available
        if 'time' in self.parameters:
            solution["time_of_day"] = self.parameters['time']
            solution["time_factor"] = time_factor
        
        return solution
    
    def calculate_quality(self, solution: Dict[str, Any]) -> float:
        """
        Calculate the quality of a TSP solution.
        
        For TSP, quality is the negative of the total distance (since we want to minimize distance).
        
        Args:
            solution: Solution dictionary
            
        Returns:
            Quality metric (higher is better)
        """
        if not solution.get("valid", False):
            return float('-inf')
        
        # Return negative distance (higher is better)
        return -solution.get("total_distance", float('inf'))
    
    def visualize_solution(self, solution: Dict[str, Any], filename: Optional[str] = None) -> None:
        """
        Visualize a TSP solution.
        
        Args:
            solution: Solution dictionary
            filename: Optional filename to save the visualization
        """
        if not solution.get("valid", False):
            print("Cannot visualize invalid solution")
            return
        
        tour = solution["tour"]
        
        # Create a new figure
        plt.figure(figsize=(10, 8))
        
        # Draw city points
        plt.scatter(self.city_positions[:, 0], self.city_positions[:, 1], c='blue', s=100, zorder=10)
        
        # Label cities
        for i, (x, y) in enumerate(self.city_positions):
            plt.text(x, y + 0.05, self.city_names[i], ha='center', va='center', fontsize=12)
        
        # Draw tour lines
        for i in range(len(tour)):
            city1 = tour[i]
            city2 = tour[(i + 1) % len(tour)]
            x1, y1 = self.city_positions[city1]
            x2, y2 = self.city_positions[city2]
            plt.plot([x1, x2], [y1, y2], 'r-', linewidth=2)
            
            # Add direction arrows
            dx = x2 - x1
            dy = y2 - y1
            arrow_x = x1 + 0.6 * dx
            arrow_y = y1 + 0.6 * dy
            plt.arrow(arrow_x, arrow_y, 0.1*dx, 0.1*dy, head_width=0.03, head_length=0.05, fc='black', ec='black')
        
        # Add information about the solution
        distance = solution["total_distance"]
        title = f"{self.name} Solution - Total Distance: {distance:.2f}"
        
        # Add time information if available
        if "time_of_day" in solution:
            time_of_day = solution["time_of_day"]
            time_factor = solution["time_factor"]
            title += f" (Time: {time_of_day:.1f}h, Factor: {time_factor:.2f})"
        
        plt.title(title)
        plt.grid(alpha=0.3)
        
        # Save the figure if filename is provided
        if filename:
            plt.savefig(filename)
            print(f"Visualization saved to {filename}")
        
        # Close the figure to avoid displaying in non-interactive environments
        plt.close()
    
    def _calculate_time_factor(self, time_of_day: float) -> float:
        """
        Calculate the time-dependent distance factor.
        
        Args:
            time_of_day: Time of day (0-24 hours)
            
        Returns:
            Factor to multiply distances by
        """
        time = time_of_day % 24  # Ensure time is in 0-24 range
        
        # Define how traffic changes throughout the day (sample model)
        # 1. Rush hours: 7-9 AM, 4-6 PM (increase distances by up to 50%)
        # 2. Night hours: 11 PM - 5 AM (decrease distances by up to 30%)
        # 3. Otherwise: normal
        
        if 7 <= time < 9:  # Morning rush hour
            factor = 1.0 + 0.5 * (1.0 - abs(time - 8) / 1.0)  # Peak at 8 AM
        elif 16 <= time < 18:  # Evening rush hour 
            factor = 1.0 + 0.5 * (1.0 - abs(time - 17) / 1.0)  # Peak at 5 PM
        elif time >= 23 or time < 5:  # Night hours
            if time >= 23:
                night_time = time - 23
            else:
                night_time = time + 1
            factor = 0.7 + 0.3 * (night_time / 6.0)  # Gradually increase from 11 PM to 5 AM
        else:
            # Normal hours: slight variations
            hour_factor = np.sin(time * np.pi / 12.0) * 0.1
            factor = 1.0 + hour_factor
        
        return factor
    
    def visualize_time_effect(self, filename: Optional[str] = None) -> None:
        """
        Visualize how time of day affects the optimal route.
        
        Args:
            filename: Optional filename to save the visualization
        """
        # Define times to check
        times = [8.0, 12.0, 17.0, 23.0]  # Rush hour, mid-day, evening rush, night
        colors = ['red', 'green', 'blue', 'purple']
        
        # Create a figure
        plt.figure(figsize=(12, 10))
        
        # Draw city points
        plt.scatter(self.city_positions[:, 0], self.city_positions[:, 1], c='black', s=100, zorder=10)
        
        # Label cities
        for i, (x, y) in enumerate(self.city_positions):
            plt.text(x, y + 0.05, self.city_names[i], ha='center', va='center', fontsize=12)
        
        # Solve for each time and plot
        legend_elements = []
        
        for time, color in zip(times, colors):
            # Solve classically for this time
            solution_name = f"time_{time}"
            solution = self.get_solution(solution_name)
            
            if solution is None:
                solution = self.solve_with_parameters({'time': time}, solution_name=solution_name)
            
            if solution.get("valid", False):
                tour = solution["tour"]
                distance = solution["total_distance"]
                time_factor = solution["time_factor"]
                
                # Draw tour lines
                for i in range(len(tour)):
                    city1 = tour[i]
                    city2 = tour[(i + 1) % len(tour)]
                    x1, y1 = self.city_positions[city1]
                    x2, y2 = self.city_positions[city2]
                    plt.plot([x1, x2], [y1, y2], color=color, linewidth=2, alpha=0.7)
                
                # Add to legend
                time_str = f"{int(time)}:00" if time.is_integer() else f"{int(time)}:{int(60*(time-int(time)))}0"
                legend_elements.append(
                    mlines.Line2D([], [], color=color, linewidth=2, 
                                label=f"Time {time_str} - Dist: {distance:.2f} (Factor: {time_factor:.2f})")
                )
        
        plt.title(f"{self.name} - Routes at Different Times of Day")
        plt.grid(alpha=0.3)
        plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
        
        # Save the figure if filename is provided
        if filename:
            plt.savefig(filename)
            print(f"Time effect visualization saved to {filename}")
        
        # Close the figure to avoid displaying in non-interactive environments
        plt.close()
    
    def __str__(self) -> str:
        """
        Return a string representation of the TSP problem.
        
        Returns:
            String description of the problem
        """
        n_cities = self.metadata["n_cities"]
        
        # Calculate some statistics about the distances
        min_dist = np.min(self.distances[np.nonzero(self.distances)])
        max_dist = np.max(self.distances)
        avg_dist = np.sum(self.distances) / (n_cities * (n_cities - 1))
        
        return f"{self.name} Problem with {n_cities} cities\n" \
               f"Distance range: [{min_dist:.2f}, {max_dist:.2f}], Average: {avg_dist:.2f}" 