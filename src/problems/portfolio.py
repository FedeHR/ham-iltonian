"""
Portfolio Optimization Problem implementation.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Tuple, Union
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

from .base import Problem
from ..hamiltonians.portfolio import create_portfolio_hamiltonian, get_portfolio_solution
from ..utils.classical_solvers import solve_portfolio_brute_force

class PortfolioProblem(Problem):
    """
    Portfolio Optimization Problem class.
    
    The Portfolio Optimization Problem seeks to select a set of assets that
    maximizes expected return subject to risk and budget constraints.
    """
    
    def __init__(
        self, 
        returns: List[float], 
        risk_matrix: np.ndarray, 
        budget: int,
        risk_factor: float = 1.0,
        asset_names: Optional[List[str]] = None,
        asset_types: Optional[List[str]] = None,
        name: str = "Portfolio Optimization"
    ):
        """
        Initialize a Portfolio Optimization Problem instance.
        
        Args:
            returns: List of expected returns for each asset
            risk_matrix: Covariance matrix representing risk
            budget: Maximum number of assets to select
            risk_factor: Weight for risk term (higher values mean more risk-averse)
            asset_names: Optional list of asset names (e.g., "AAPL", "MSFT")
            asset_types: Optional list of asset types/sectors (e.g., "tech", "finance")
            name: Name of the problem instance
        """
        super().__init__(name)
        
        # Validate input
        self.n_assets = len(returns)
        if risk_matrix.shape != (self.n_assets, self.n_assets):
            raise ValueError("Risk matrix must be square and match the number of returns")
        
        self.returns = returns
        self.risk_matrix = risk_matrix
        self.budget = budget
        self.risk_factor = risk_factor
        
        # Set asset names if not provided
        if asset_names is None:
            self.asset_names = [f"Asset {i}" for i in range(self.n_assets)]
        else:
            if len(asset_names) != self.n_assets:
                raise ValueError("Number of asset names must match number of returns")
            self.asset_names = asset_names
        
        # Set asset types if not provided
        if asset_types is None:
            self.asset_types = ["general" for _ in range(self.n_assets)]
        else:
            if len(asset_types) != self.n_assets:
                raise ValueError("Number of asset types must match number of returns")
            self.asset_types = asset_types
        
        # Store problem metadata
        self.metadata["n_assets"] = self.n_assets
        self.metadata["budget"] = self.budget
        self.metadata["risk_factor"] = self.risk_factor
        self.metadata["max_return"] = float(np.max(returns))
        self.metadata["asset_names"] = self.asset_names
        self.metadata["asset_types"] = self.asset_types
    
    def create_hamiltonian(self, time_dependent: bool = False) -> Any:
        """
        Create the Hamiltonian for this Portfolio Optimization problem.
        
        Args:
            time_dependent: Whether to create a time-dependent Hamiltonian
            
        Returns:
            Hamiltonian for the Portfolio Optimization problem
        """
        hamiltonian = create_portfolio_hamiltonian(
            self.returns, 
            self.risk_matrix, 
            self.budget,
            self.risk_factor,
            time_dependent=time_dependent
        )
        
        # Add asset types metadata for time-dependent Hamiltonian
        if time_dependent:
            hamiltonian.metadata["asset_types"] = self.asset_types
        
        self._hamiltonian = hamiltonian
        return hamiltonian
    
    def solve_classically(self, market_conditions: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Solve the Portfolio Optimization problem using classical methods.
        
        Args:
            market_conditions: Optional market conditions affecting returns
            
        Returns:
            Dictionary with solution details
        """
        # Adjust returns if market conditions are provided
        adjusted_returns = self.returns
        if market_conditions:
            adjusted_returns = self._adjust_returns_for_market(market_conditions)
        
        # Solve using brute force
        solution = solve_portfolio_brute_force(
            adjusted_returns, 
            self.risk_matrix, 
            self.budget, 
            self.risk_factor
        )
        
        # Add market conditions to the solution
        if market_conditions:
            solution["market_conditions"] = market_conditions
        
        # Store solution
        self.add_solution("classical", solution)
        return solution
    
    def get_solution_from_bitstring(self, bitstring: str) -> Dict[str, Any]:
        """
        Get the Portfolio Optimization solution from a bitstring.
        
        Args:
            bitstring: Bit string representation
            
        Returns:
            Dictionary with solution details
        """
        # Use current market conditions if available in parameters
        market_conditions = None
        if 'market_conditions' in self.parameters:
            market_conditions = self.parameters['market_conditions']
        
        # Get the solution
        solution = get_portfolio_solution(
            bitstring, 
            self.returns, 
            self.risk_matrix, 
            self.budget,
            market_conditions=market_conditions
        )
        
        return solution
    
    def calculate_quality(self, solution: Dict[str, Any]) -> float:
        """
        Calculate the quality of a Portfolio Optimization solution.
        
        For portfolio optimization, quality is the difference between expected return and risk.
        Higher values indicate better solutions.
        
        Args:
            solution: Solution dictionary
            
        Returns:
            Quality metric (higher is better)
        """
        if not solution.get("valid", False):
            return float('-inf')
        
        # Return expected return minus risk*risk_factor
        return solution.get("expected_return", 0.0) - self.risk_factor * solution.get("risk", float('inf'))
    
    def _adjust_returns_for_market(self, market_conditions: Dict[str, float]) -> List[float]:
        """
        Adjust returns based on market conditions.
        
        Args:
            market_conditions: Dictionary of market condition parameters
            
        Returns:
            List of adjusted returns
        """
        market_trend = market_conditions.get('market_trend', 0.0)
        volatility = market_conditions.get('volatility', 0.5)
        interest_rate = market_conditions.get('interest_rate', 2.0)
        
        adjusted_returns = []
        
        for i, base_return in enumerate(self.returns):
            asset_type = self.asset_types[i]
            
            # Adjust based on asset type and market conditions
            if asset_type == "tech":
                # Tech stocks are more volatile and responsive to market trends
                factor = 1.0 + (market_trend * 1.5) - (volatility * 0.8) - (interest_rate * 0.1)
            elif asset_type == "finance":
                # Financial stocks benefit from higher interest rates but are affected by volatility
                factor = 1.0 + (market_trend * 0.8) + (interest_rate * 0.2) - (volatility * 0.5)
            elif asset_type == "energy":
                # Energy stocks are less correlated with market trends
                factor = 1.0 + (market_trend * 0.3) - (volatility * 0.2) - (interest_rate * 0.05)
            elif asset_type == "healthcare":
                # Healthcare is defensive, less affected by market conditions
                factor = 1.0 + (market_trend * 0.5) - (volatility * 0.1) - (interest_rate * 0.05)
            elif asset_type == "consumer":
                # Consumer stocks are moderately affected by market trends
                factor = 1.0 + (market_trend * 0.7) - (volatility * 0.3) - (interest_rate * 0.1)
            else:
                # General case
                factor = 1.0 + (market_trend * 0.5) - (volatility * 0.3) - (interest_rate * 0.1)
            
            # Apply the adjustment to the return
            adjusted_returns.append(base_return * factor)
        
        return adjusted_returns
    
    def visualize_solution(self, solution: Dict[str, Any], filename: Optional[str] = None) -> None:
        """
        Visualize a Portfolio Optimization solution.
        
        Args:
            solution: Solution dictionary
            filename: Optional filename to save the visualization
        """
        if not solution.get("valid", False):
            print("Cannot visualize invalid solution")
            return
        
        selected_assets = solution["selected_assets"]
        expected_return = solution["expected_return"]
        risk = solution["risk"]
        ratio = solution.get("return_risk_ratio", expected_return / risk if risk > 0 else float('inf'))
        
        # Market conditions info if available
        market_info = ""
        if "market_conditions" in solution and solution["market_conditions"]:
            mc = solution["market_conditions"]
            market_trend = mc.get('market_trend', 0.0)
            volatility = mc.get('volatility', 0.5)
            interest_rate = mc.get('interest_rate', 2.0)
            market_info = f"Market Trend: {market_trend:.2f}, Volatility: {volatility:.2f}, Interest Rate: {interest_rate:.2f}%"
        
        # Create a new figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        
        # Plot selected assets with their weights (equal weights in this simple model)
        ax1.barh(
            [self.asset_names[i] for i in selected_assets],
            [self.returns[i] for i in selected_assets],
            color='skyblue'
        )
        ax1.set_title('Selected Assets and Their Returns')
        ax1.set_xlabel('Expected Return')
        ax1.set_ylabel('Asset')
        ax1.grid(alpha=0.3)
        
        # Plot a risk-return comparison of all assets, highlighting selected ones
        for i in range(self.n_assets):
            asset_risk = self.risk_matrix[i, i]  # Variance (diagonal of covariance matrix)
            asset_return = self.returns[i]
            
            if i in selected_assets:
                ax2.scatter(asset_risk, asset_return, c='red', s=100, zorder=10)
                ax2.text(asset_risk, asset_return + 0.01, self.asset_names[i], ha='center')
            else:
                ax2.scatter(asset_risk, asset_return, c='blue', s=50, alpha=0.5)
        
        ax2.set_title('Risk-Return Profile')
        ax2.set_xlabel('Risk (Variance)')
        ax2.set_ylabel('Expected Return')
        ax2.grid(alpha=0.3)
        
        # Mark selected portfolio
        portfolio_title = f"{self.name} Solution\n"
        portfolio_title += f"Expected Return: {expected_return:.4f}, Risk: {risk:.4f}, Ratio: {ratio:.4f}\n"
        if market_info:
            portfolio_title += market_info
        
        plt.suptitle(portfolio_title)
        plt.tight_layout()
        
        # Save the figure if filename is provided
        if filename:
            plt.savefig(filename)
            print(f"Visualization saved to {filename}")
        
        # Close the figure to avoid displaying in non-interactive environments
        plt.close()
    
    def visualize_market_effect(self, filename: Optional[str] = None) -> None:
        """
        Visualize how market conditions affect the optimal portfolio.
        
        Args:
            filename: Optional filename to save the visualization
        """
        # Define market conditions to check
        market_conditions = [
            {"name": "Bull Market", "market_trend": 0.8, "volatility": 0.3, "interest_rate": 2.0},
            {"name": "Bear Market", "market_trend": -0.8, "volatility": 0.7, "interest_rate": 1.0},
            {"name": "High Volatility", "market_trend": 0.0, "volatility": 0.9, "interest_rate": 2.5},
            {"name": "High Interest Rates", "market_trend": 0.0, "volatility": 0.4, "interest_rate": 5.0}
        ]
        
        # Solve for each market condition
        for mc in market_conditions:
            solution_name = f"market_{mc['name'].lower().replace(' ', '_')}"
            existing_solution = self.get_solution(solution_name)
            
            if existing_solution is None:
                # Solve with these market conditions
                solution = self.solve_with_parameters(
                    {'market_conditions': mc}, 
                    solution_name=solution_name
                )
        
        # Create a figure to compare portfolios
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        # Plot each solution
        for i, mc in enumerate(market_conditions):
            solution_name = f"market_{mc['name'].lower().replace(' ', '_')}"
            solution = self.get_solution(solution_name)
            
            if solution and solution.get("valid", False):
                ax = axes[i]
                selected_assets = solution["selected_assets"]
                expected_return = solution["expected_return"]
                risk = solution["risk"]
                ratio = solution.get("return_risk_ratio", expected_return / risk if risk > 0 else float('inf'))
                
                # Plot selected assets
                asset_colors = []
                for j in range(self.n_assets):
                    if j in selected_assets:
                        if self.asset_types[j] == "tech":
                            color = 'skyblue'
                        elif self.asset_types[j] == "finance":
                            color = 'green'
                        elif self.asset_types[j] == "energy":
                            color = 'orange'
                        elif self.asset_types[j] == "healthcare":
                            color = 'purple'
                        elif self.asset_types[j] == "consumer":
                            color = 'pink'
                        else:
                            color = 'gray'
                        
                        asset_colors.append(color)
                
                # Bar chart of selected assets
                ax.barh(
                    [self.asset_names[i] for i in selected_assets],
                    [self.returns[i] for i in selected_assets],
                    color=asset_colors
                )
                
                ax.set_title(f"{mc['name']}\nReturn: {expected_return:.3f}, Risk: {risk:.3f}, Ratio: {ratio:.2f}")
                ax.set_xlabel('Expected Return')
                ax.grid(alpha=0.3)
        
        # Create a custom legend for asset types
        legend_elements = []
        asset_type_colors = {
            "tech": 'skyblue',
            "finance": 'green',
            "energy": 'orange',
            "healthcare": 'purple',
            "consumer": 'pink',
            "general": 'gray'
        }
        
        for asset_type, color in asset_type_colors.items():
            if asset_type in self.asset_types:
                legend_elements.append(
                    mpatches.Patch(color=color, label=asset_type.capitalize())
                )
        
        plt.figlegend(handles=legend_elements, loc='lower center', ncol=len(legend_elements))
        plt.suptitle(f"{self.name} - Portfolios Under Different Market Conditions")
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # Save the figure if filename is provided
        if filename:
            plt.savefig(filename)
            print(f"Market effect visualization saved to {filename}")
        
        # Close the figure to avoid displaying in non-interactive environments
        plt.close()
    
    def __str__(self) -> str:
        """
        Return a string representation of the problem.
        
        Returns:
            String representation
        """
        asset_info = ", ".join([f"{name} ({atype})" for name, atype in zip(self.asset_names, self.asset_types)])
        return (f"{self.name} Problem\n"
                f"Assets: {self.n_assets}, Budget: {self.budget}, Risk Factor: {self.risk_factor}\n"
                f"Assets: {asset_info}") 