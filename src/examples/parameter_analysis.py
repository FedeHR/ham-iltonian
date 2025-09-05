"""
Example script demonstrating parameter effect visualizations for MaxCut problems.
This script shows how to use the visualization utilities to analyze parameter effects.
"""
import sys
import os
import numpy as np

# Add the parent directory to sys.path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hamiltonians.problems import create_maxcut_instance
from hamiltonians.utils.parameter_viz import (
    compare_graph_weights,
    solution_evolution,
    parameter_sensitivity_plot,
    parameter_interaction_heatmap,
    solution_stability
)

def main():
    # Create a sample MaxCut problem
    print("Creating MaxCut problem instance...")
    maxcut = create_maxcut_instance(
        n_nodes=6,  # Small enough to visualize clearly
        edge_probability=0.7,
        init_weight_range=(0.5, 2.0),
        graph_type="random",
        seed=42
    )
    
    # Show the original graph
    print("Visualizing original graph...")
    maxcut.visualize_graph()
    
    # 1. Visualize weight changes with edge_density_scaling parameter
    print("\nAnalyzing edge_density_scaling parameter effects...")
    edge_scaling_values = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    compare_graph_weights(
        maxcut, 
        "edge_density_scaling", 
        edge_scaling_values,
        title="Edge Weights with Different Density Scaling",
        filename="output/edge_density_weights.png"
    )
    
    # 2. Visualize solution evolution with weight_emphasis parameter
    print("\nAnalyzing weight_emphasis parameter effects on solutions...")
    emphasis_values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    solution_evolution(
        maxcut,
        "weight_emphasis",
        emphasis_values,
        title="Solution Evolution with Weight Emphasis",
        filename="output/weight_emphasis_solutions.png"
    )
    
    # 3. Create parameter sensitivity plot
    print("\nCreating parameter sensitivity analysis...")
    param_range = np.linspace(0.0, 3.0, 15)
    parameter_sensitivity_plot(
        maxcut,
        "weight_emphasis",
        param_range,
        title="Cut Value Sensitivity to Weight Emphasis",
        filename="output/weight_emphasis_sensitivity.png"
    )
    
    # 4. Parameter interaction analysis (weight_emphasis and edge_density_scaling)
    print("\nAnalyzing parameter interactions...")
    emphasis_range = np.linspace(0.0, 2.0, 5)
    density_range = np.linspace(0.0, 2.0, 5)
    parameter_interaction_heatmap(
        maxcut,
        "weight_emphasis", emphasis_range,
        "edge_density_scaling", density_range,
        title="Parameter Interaction: Weight Emphasis vs. Edge Density Scaling",
        filename="output/parameter_interaction.png"
    )
    
    # 5. Solution stability analysis
    print("\nAnalyzing solution stability...")
    angle_range = np.linspace(0.0, np.pi, 20)
    solution_stability(
        maxcut,
        "weighted_sine",
        angle_range,
        title="Solution Stability with Sine Wave Modifier",
        filename="output/sine_solution_stability.png"
    )
    
    print("\nAll visualizations completed! Check the output directory for the saved images.")

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    main() 