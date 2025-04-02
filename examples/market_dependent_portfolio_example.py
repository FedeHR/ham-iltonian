#!/usr/bin/env python3
"""
Example of market-dependent Portfolio Optimization where the optimal asset selection
changes based on market conditions (bull/bear market, volatility, interest rates).
"""
import numpy as np
import matplotlib
matplotlib.use('agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.problems import PortfolioProblem

# Create a sample portfolio problem with different asset types
def create_sample_portfolio():
    """Create a realistic portfolio with different asset types."""
    # Asset names and types
    asset_names = [
        "Tech Co.", "E-commerce", "SaaS Provider", "Semiconductor",  # Tech stocks
        "Bank Inc.", "Insurance Co.", "Investment Firm", "Payment Processor",  # Financial stocks
        "Oil & Gas", "Renewable Energy", "Mining Corp",  # Energy stocks
        "Pharma Inc.", "Medical Devices", "Healthcare Services",  # Healthcare stocks
        "Retail Chain", "Food & Beverage", "Luxury Goods",  # Consumer stocks
    ]
    
    asset_types = [
        "tech", "tech", "tech", "tech",
        "finance", "finance", "finance", "finance",
        "energy", "energy", "energy",
        "healthcare", "healthcare", "healthcare",
        "consumer", "consumer", "consumer"
    ]
    
    # Generate some sample base returns
    # Different sectors have different baseline returns and risks
    returns = [
        0.15, 0.18, 0.22, 0.20,  # Tech: higher returns
        0.10, 0.12, 0.11, 0.14,  # Finance: moderate returns
        0.08, 0.09, 0.07,        # Energy: lower returns
        0.12, 0.11, 0.10,        # Healthcare: moderate returns
        0.13, 0.09, 0.16         # Consumer: varied returns
    ]
    
    n_assets = len(returns)
    
    # Create a sample covariance matrix with more realistic correlations
    # Start with a diagonal matrix (independent risks)
    risk_matrix = np.diag([
        0.20, 0.25, 0.30, 0.22,  # Tech: higher risks
        0.15, 0.14, 0.18, 0.16,  # Finance: moderate risks
        0.20, 0.18, 0.22,        # Energy: higher risks
        0.10, 0.12, 0.09,        # Healthcare: lower risks
        0.14, 0.10, 0.18         # Consumer: varied risks
    ])
    
    # Add correlations between assets (simplified)
    # Within each sector, there's higher correlation
    # Between different sectors, correlations vary
    
    # Helper function to set correlation between assets
    def set_correlation(i, j, corr):
        risk_matrix[i, j] = corr * np.sqrt(risk_matrix[i, i] * risk_matrix[j, j])
        risk_matrix[j, i] = risk_matrix[i, j]  # Make sure matrix is symmetric
    
    # Add intra-sector correlations (higher within same sector)
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            if asset_types[i] == asset_types[j]:
                # Same sector: higher correlation (0.5-0.7)
                set_correlation(i, j, 0.5 + np.random.rand() * 0.2)
            else:
                # Different sectors: lower correlation (0.1-0.4)
                # Some sectors have specific relationships
                if (asset_types[i] == "tech" and asset_types[j] == "finance") or \
                   (asset_types[i] == "finance" and asset_types[j] == "tech"):
                    # Tech and Finance have moderate correlation
                    set_correlation(i, j, 0.3 + np.random.rand() * 0.1)
                elif (asset_types[i] == "energy" and asset_types[j] == "finance") or \
                     (asset_types[i] == "finance" and asset_types[j] == "energy"):
                    # Energy and Finance have moderate correlation
                    set_correlation(i, j, 0.3 + np.random.rand() * 0.1)
                elif (asset_types[i] == "healthcare" and asset_types[j] == "consumer") or \
                     (asset_types[i] == "consumer" and asset_types[j] == "healthcare"):
                    # Healthcare and Consumer have moderate correlation
                    set_correlation(i, j, 0.25 + np.random.rand() * 0.15)
                else:
                    # Default low correlation
                    set_correlation(i, j, 0.1 + np.random.rand() * 0.2)
    
    # Create the portfolio problem
    return PortfolioProblem(
        returns=returns,
        risk_matrix=risk_matrix,
        budget=5,  # Select 5 assets
        risk_factor=1.0,  # Balance between return and risk
        asset_names=asset_names,
        asset_types=asset_types,
        name="Market-Dependent Portfolio"
    )

# Create the portfolio problem
problem = create_sample_portfolio()

# Print problem information
problem.print_info()

# Create time-dependent Hamiltonian
print("\nCreating market-dependent Hamiltonian...")
hamiltonian = problem.create_hamiltonian(time_dependent=True)
problem.print_hamiltonian(truncate=True)

# Define different market conditions to test
market_conditions = [
    {
        "name": "Bull Market",
        "description": "Strong economic growth, rising stock prices",
        "market_trend": 0.8,    # Strongly positive market
        "volatility": 0.3,      # Low volatility
        "interest_rate": 2.0    # Moderate interest rates
    },
    {
        "name": "Bear Market",
        "description": "Economic downturn, falling stock prices",
        "market_trend": -0.8,   # Strongly negative market
        "volatility": 0.7,      # High volatility
        "interest_rate": 1.0    # Low interest rates (stimulus)
    },
    {
        "name": "High Volatility",
        "description": "Uncertain market conditions, high price fluctuations",
        "market_trend": 0.0,    # Neutral market trend
        "volatility": 0.9,      # Very high volatility
        "interest_rate": 2.5    # Moderate-high interest rates
    },
    {
        "name": "High Interest Rates",
        "description": "Central bank tightening, higher borrowing costs",
        "market_trend": 0.0,    # Neutral market trend
        "volatility": 0.4,      # Moderate volatility
        "interest_rate": 5.0    # High interest rates
    }
]

# Solve for each market condition
for mc in market_conditions:
    print(f"\nSolving for {mc['name']} ({mc['description']})...")
    solution_name = f"market_{mc['name'].lower().replace(' ', '_')}"
    
    # Solve with these market conditions
    solution = problem.solve_with_parameters(
        {'market_conditions': mc}, 
        solution_name=solution_name
    )
    
    # Print solution details
    problem.print_solution(solution_name)
    
    # Visualize the solution
    problem.visualize_solution(solution, filename=f"portfolio_{solution_name}.png")

# Compare all solutions
print("\nComparing portfolios under different market conditions:")
solution_names = [f"market_{mc['name'].lower().replace(' ', '_')}" for mc in market_conditions]
problem.print_comparison(solution_names)

# Create visualization showing the effect of market conditions
problem.visualize_market_effect(filename="portfolio_market_effect.png")

# Create detailed visualizations of parameter effects

# Effect of market trend
print("\nAnalyzing effect of market trend:")
market_trends = np.linspace(-1.0, 1.0, 11)  # From strong bear to strong bull
for trend in market_trends:
    params = {'market_conditions': {'market_trend': trend, 'volatility': 0.5, 'interest_rate': 2.0}}
    problem.solve_with_parameters(params, solution_name=f"trend_{trend:.1f}")

problem.visualize_parameter_effect(
    'market_conditions', 
    [{'market_trend': t, 'volatility': 0.5, 'interest_rate': 2.0} for t in market_trends], 
    metric_name="expected_return",
    filename="portfolio_market_trend_effect.png"
)

# Effect of volatility
print("\nAnalyzing effect of volatility:")
volatilities = np.linspace(0.1, 1.0, 10)  # From low to high volatility
for vol in volatilities:
    params = {'market_conditions': {'market_trend': 0.0, 'volatility': vol, 'interest_rate': 2.0}}
    problem.solve_with_parameters(params, solution_name=f"volatility_{vol:.1f}")

problem.visualize_parameter_effect(
    'market_conditions', 
    [{'market_trend': 0.0, 'volatility': v, 'interest_rate': 2.0} for v in volatilities], 
    metric_name="risk",
    filename="portfolio_volatility_effect.png"
)

# Effect of interest rates
print("\nAnalyzing effect of interest rates:")
interest_rates = np.linspace(0.5, 6.0, 12)  # From low to high interest rates
for rate in interest_rates:
    params = {'market_conditions': {'market_trend': 0.0, 'volatility': 0.5, 'interest_rate': rate}}
    problem.solve_with_parameters(params, solution_name=f"interest_{rate:.1f}")

problem.visualize_parameter_effect(
    'market_conditions', 
    [{'market_trend': 0.0, 'volatility': 0.5, 'interest_rate': r} for r in interest_rates], 
    metric_name="return_risk_ratio",
    filename="portfolio_interest_rate_effect.png"
)

print("\nAll visualizations have been saved. Check the output directory for PNG files.") 