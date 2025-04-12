"""
Portfolio Optimization Problem implementation.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any
import matplotlib.patches as mpatches

from problems.base import Problem
from hamiltonian import Hamiltonian
from utils.pauli_utils import create_z_term, create_zz_term
from utils.classical_solvers import solve_portfolio_brute_force
from parameter_modifiers.portfolio import get_modifiers

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
        
        self.returns = np.array(returns)
        self.risk_matrix = np.array(risk_matrix)
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
        
        # Store original values for parameter resets
        self.original_returns = self.returns.copy()
        self.original_risk_matrix = self.risk_matrix.copy()
        self.original_risk_factor = self.risk_factor
        
        # Register modifiers
        default_modifiers = get_modifiers()
        self.modifier_functions.update(default_modifiers)
        
        # Build the initial Hamiltonian
        self.build_hamiltonian()
    
    def _apply_modifier(self, modifier_name: str, *args) -> None:
        """
        Apply the modifier to the problem parameters.
        
        Args:
            modifier_name: Name of the modifier function to apply
            *args: Parameters for the modifier function
        """
        modifier_func = self.modifier_functions[modifier_name]
        
        if modifier_name == "expected_return":
            # Apply to all returns
            self.returns = np.array([modifier_func(ret, *args) for ret in self.returns])
            self.metadata["max_return"] = float(np.max(self.returns))
            
        elif modifier_name == "risk_aversion":
            # Apply to risk factor
            self.risk_factor = modifier_func(self.risk_factor, *args)
            self.metadata["risk_factor"] = self.risk_factor
            
        elif modifier_name == "market_volatility":
            # Apply to risk matrix
            for i in range(self.n_assets):
                for j in range(self.n_assets):
                    self.risk_matrix[i, j] = modifier_func(self.risk_matrix[i, j], *args)
    
    def build_hamiltonian(self) -> None:
        """
        Build the Hamiltonian for this Portfolio Optimization problem following Lucas (2014) formulation.
        
        The formulation has three components:
        1. Return term: -∑_i r_i x_i  (to maximize return, negative for minimization)
        2. Risk term: ∑_i,j σ_ij x_i x_j  (to minimize risk)
        3. Budget constraint: A(∑_i x_i - B)^2  (to enforce budget constraint)
        
        Where:
        - r_i is the expected return of asset i
        - σ_ij is the covariance between assets i and j
        - B is the budget (number of assets to select)
        - A is a penalty coefficient for violating the budget constraint
        """
        # Create a new Hamiltonian or clear the existing one
        if self.hamiltonian is None:
            self.hamiltonian = Hamiltonian(self.n_assets)
        else:
            self.hamiltonian.clear()
        
        # Add metadata
        self.hamiltonian.metadata = {
            "problem": "Portfolio",
            "n_assets": self.n_assets,
            "returns": self.returns.tolist(),
            "risk_matrix": self.risk_matrix.tolist(),
            "budget": self.budget,
            "risk_factor": self.risk_factor,
            "asset_types": self.asset_types
        }
        
        # 1. Return term (linear) - Negative because we're maximizing returns
        for i in range(self.n_assets):
            coef, term = create_z_term(i, -self.returns[i] / 2)  # Divide by 2 for Z operator convention
            self.hamiltonian.add_term(coef, term)
        
        # 2. Risk term (quadratic)
        for i in range(self.n_assets):
            for j in range(i+1, self.n_assets):
                coef, term = create_zz_term(i, j, self.risk_factor * self.risk_matrix[i, j] / 4)  # Divide by 4 for ZZ operator convention
                self.hamiltonian.add_term(coef, term)
        
        # 3. Budget constraint: penalty for selecting more than budget assets
        # We calculate a penalty coefficient large enough to enforce the constraint
        penalty = 2.0 * max(abs(np.max(self.returns)), abs(np.max(self.risk_matrix))) * self.n_assets
        
        # We penalize: penalty * (∑_i x_i - budget)^2
        # Expanded: penalty * (∑_i x_i)^2 - 2*penalty*budget*∑_i x_i + penalty*budget^2
        
        # Constant term: penalty*budget^2
        self.hamiltonian.add_constant(penalty * self.budget * self.budget)
        
        # Linear terms: penalty*∑_i x_i - 2*penalty*budget*∑_i x_i = penalty*(1-2*budget)*∑_i x_i
        for i in range(self.n_assets):
            coef, term = create_z_term(i, penalty * (1 - 2 * self.budget) / 2)  # Divide by 2 for Z operator convention
            self.hamiltonian.add_term(coef, term)
        
        # Quadratic terms: penalty*(∑_i x_i)^2 = penalty*∑_i x_i + penalty*∑_i≠j x_i x_j
        # Since we already added the linear part above, we just need the cross-terms
        for i in range(self.n_assets):
            for j in range(i+1, self.n_assets):
                coef, term = create_zz_term(i, j, penalty / 4)  # Divide by 4 for ZZ operator convention
                self.hamiltonian.add_term(coef, term)
    
    def solve_classically(self, **kwargs) -> Dict[str, Any]:
        """
        Solve the Portfolio Optimization problem using classical methods.
        
        Returns:
            Dictionary with solution details
        """
        # Solve using brute force
        solution = solve_portfolio_brute_force(
            self.returns, 
            self.risk_matrix, 
            self.budget, 
            self.risk_factor
        )
        
        # Store solution
        self.solutions["classical"] = solution
        return solution
    
    def evaluate_bitstring(self, bitstring: str) -> Dict[str, Any]:
        """
        Get the Portfolio Optimization solution from a bitstring.
        
        Args:
            bitstring: Bit string representation
            
        Returns:
            Dictionary with solution details
        """
        if isinstance(bitstring, str):
            assignment = [int(bit) for bit in bitstring]
        else:
            assignment = bitstring
        
        # Ensure the assignment has the right length
        if len(assignment) < self.n_assets:
            assignment = assignment + [0] * (self.n_assets - len(assignment))
        elif len(assignment) > self.n_assets:
            assignment = assignment[:self.n_assets]
        
        # Get selected assets
        selected_assets = [i for i, bit in enumerate(assignment) if bit == 1]
        total_selected = len(selected_assets)
        
        # Calculate expected return
        expected_return = sum(self.returns[i] for i in selected_assets)
        
        # Calculate risk (portfolio variance)
        risk = 0.0
        for i in selected_assets:
            for j in selected_assets:
                risk += self.risk_matrix[i, j]
        
        # Check if solution is valid (meets budget constraint)
        valid = total_selected <= self.budget
        
        return {
            "bitstring": bitstring,
            "selected_assets": selected_assets,
            "asset_names": [self.asset_names[i] for i in selected_assets],
            "expected_return": expected_return,
            "risk": risk,
            "total_selected": total_selected,
            "budget": self.budget,
            "valid": valid,
            "quality": expected_return - self.risk_factor * risk if valid else float('-inf')
        }
    
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
    
    def reset_parameters(self):
        """
        Reset all parameters to their original values.
        """
        self.returns = self.original_returns.copy()
        self.risk_matrix = self.original_risk_matrix.copy()
        self.risk_factor = self.original_risk_factor
        
        # Update metadata
        self.metadata["max_return"] = float(np.max(self.returns))
        self.metadata["risk_factor"] = self.risk_factor
        
        # Rebuild the Hamiltonian
        self.build_hamiltonian()
    
    def visualize_solution(self, solution: Dict[str, Any], filename: Optional[str] = None) -> None:
        """
        Visualize a Portfolio Optimization solution.
        
        Args:
            solution: Solution dictionary
            filename: Optional filename to save the visualization
        """
        if not solution:
            raise ValueError("No solution provided")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Extract data
        selected_assets = solution.get("selected_assets", [])
        expected_return = solution.get("expected_return", 0.0)
        risk = solution.get("risk", 0.0)
        
        # Plot all assets with returns vs. risk contribution
        asset_returns = self.returns
        asset_risks = np.array([self.risk_matrix[i, i] for i in range(self.n_assets)])
        
        # Calculate risk contribution for each asset
        risk_contributions = []
        for i in range(self.n_assets):
            contribution = 0
            for j in range(self.n_assets):
                if j in selected_assets:
                    contribution += self.risk_matrix[i, j]
            risk_contributions.append(contribution)
        
        # Create color map based on asset types
        unique_types = list(set(self.asset_types))
        color_map = plt.cm.get_cmap('tab10', len(unique_types))
        colors = [color_map(unique_types.index(asset_type)) for asset_type in self.asset_types]
        
        # Plot 1: Return vs. Risk Contribution
        for i in range(self.n_assets):
            marker = 'o' if i in selected_assets else 'x'
            size = 100 if i in selected_assets else 50
            ax1.scatter(risk_contributions[i], asset_returns[i], 
                     c=[colors[i]], marker=marker, s=size, 
                     label=self.asset_names[i] if i in selected_assets else None)
            
            # Add labels for selected assets
            if i in selected_assets:
                ax1.annotate(self.asset_names[i], 
                          (risk_contributions[i], asset_returns[i]),
                          xytext=(5, 5), textcoords='offset points')
        
        ax1.set_xlabel('Risk Contribution')
        ax1.set_ylabel('Expected Return')
        ax1.set_title('Asset Return vs. Risk Contribution')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Portfolio Composition
        if selected_assets:
            # Calculate portfolio weights (equal weights for simplicity)
            weights = [1.0 / len(selected_assets)] * len(selected_assets)
            selected_names = [self.asset_names[i] for i in selected_assets]
            selected_types = [self.asset_types[i] for i in selected_assets]
            
            # Get colors for the selected assets
            pie_colors = [colors[i] for i in selected_assets]
            
            # Create pie chart
            ax2.pie(weights, labels=selected_names, colors=pie_colors, autopct='%1.1f%%')
            ax2.set_title(f'Portfolio Composition\nReturn: {expected_return:.3f}, Risk: {risk:.3f}')
            
            # Add a legend for asset types
            type_patches = []
            for i, asset_type in enumerate(unique_types):
                if asset_type in selected_types:
                    type_patches.append(mpatches.Patch(color=color_map(i), label=asset_type))
            
            if type_patches:
                ax2.legend(handles=type_patches, loc='upper right', bbox_to_anchor=(1.3, 1))
        else:
            ax2.text(0.5, 0.5, 'No assets selected', ha='center', va='center', fontsize=14)
            ax2.axis('off')
        
        # Add overall portfolio metrics
        plt.figtext(0.5, 0.01, 
                  f"Portfolio metrics: Expected Return = {expected_return:.3f}, Risk = {risk:.3f}, "
                  f"Return/Risk Ratio = {expected_return/risk if risk else 0:.3f}\n"
                  f"Selected {len(selected_assets)}/{self.n_assets} assets (Budget: {self.budget})",
                  ha='center', fontsize=12)
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename)
            print(f"Visualization saved to {filename}")
        
        plt.show()
    
    def __str__(self) -> str:
        """
        Return a string representation of the Portfolio Optimization problem.
        
        Returns:
            String description of the problem
        """
        assets_str = ", ".join(self.asset_names[:5])
        if len(self.asset_names) > 5:
            assets_str += f", ... (+{len(self.asset_names) - 5} more)"
        
        return f"{self.name} Problem with {self.n_assets} assets\n" \
               f"Budget: {self.budget} assets\n" \
               f"Risk Factor: {self.risk_factor}\n" \
               f"Assets: {assets_str}" 