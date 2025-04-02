"""
Implementation of the Portfolio Optimization Problem Hamiltonian.

The Portfolio Optimization Problem seeks to select a set of assets that
maximizes expected return subject to risk and budget constraints.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from .base import Hamiltonian
from ..utils.pauli_utils import create_z_term, create_zz_term

def create_portfolio_hamiltonian(
    returns: List[float], 
    risk_matrix: np.ndarray, 
    budget: int,
    risk_factor: float = 1.0,
    time_dependent: bool = False
) -> Hamiltonian:
    """
    Create a Hamiltonian for the Portfolio Optimization Problem.
    
    The objective is to maximize return while minimizing risk,
    subject to a budget constraint on the number of assets.
    
    Args:
        returns: List of expected returns for each asset
        risk_matrix: Covariance matrix representing risk
        budget: Maximum number of assets to select
        risk_factor: Weight for risk term (higher values mean more risk-averse)
        time_dependent: Whether to create a time-dependent Hamiltonian
        
    Returns:
        Hamiltonian for the Portfolio Optimization Problem
    """
    n_assets = len(returns)
    
    # Validate input
    if risk_matrix.shape != (n_assets, n_assets):
        raise ValueError("Risk matrix must be square and match the number of returns")
    
    # Create the Hamiltonian
    hamiltonian = Hamiltonian(n_assets)
    
    # Add metadata
    hamiltonian.metadata["problem"] = "Portfolio"
    hamiltonian.metadata["n_assets"] = n_assets
    hamiltonian.metadata["returns"] = returns
    hamiltonian.metadata["risk_matrix"] = risk_matrix.tolist()
    hamiltonian.metadata["budget"] = budget
    hamiltonian.metadata["risk_factor"] = risk_factor
    
    # The following defines a maximization problem, so we negate it for our
    # minimization framework
    
    if time_dependent:
        # For time-dependent returns, we'll use parameter functions
        
        # Define a function that adjusts returns based on economic conditions
        def economic_modifier(base_coef, params):
            # Extract the economic condition parameters
            market_trend = params.get('market_trend', 0.0)  # -1 to 1 (bear to bull)
            volatility = params.get('volatility', 0.5)      # 0 to 1 (low to high)
            interest_rate = params.get('interest_rate', 2.0)  # Percent
            
            # Different asset classes respond differently to market conditions
            asset_idx = int(base_coef)  # We encoded the asset index in the coefficient
            asset_type = hamiltonian.metadata.get("asset_types", ["general"] * n_assets)[asset_idx]
            
            # Base return
            base_return = returns[asset_idx]
            
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
            adjusted_return = base_return * factor
            
            # Return negative for minimization framework
            return -adjusted_return
        
        # Return term (linear)
        for i in range(n_assets):
            qubit_idx = i
            # Use a parametric term that depends on economic conditions
            # We use the index as the "base coefficient" to identify the asset in the modifier
            hamiltonian.add_parametric_term(
                i, 
                f"Z{qubit_idx}", 
                'market_conditions',
                economic_modifier
            )
        
        # Risk term (quadratic)
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                coef, term = create_zz_term(i, j, risk_factor * risk_matrix[i, j] / 2)
                hamiltonian.add_term(coef, term)
    else:
        # Standard portfolio with fixed returns
        # Return term (linear)
        for i in range(n_assets):
            qubit_idx = i
            coef, term = create_z_term(qubit_idx, -returns[i])
            hamiltonian.add_term(coef, term)
        
        # Risk term (quadratic)
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                coef, term = create_zz_term(i, j, risk_factor * risk_matrix[i, j] / 2)
                hamiltonian.add_term(coef, term)
    
    # Budget constraint: penalty for selecting more than budget assets
    # This is implemented as a soft constraint with a high penalty
    penalty = 2.0 * max(abs(np.max(returns)), abs(np.max(risk_matrix))) * n_assets
    
    # We penalize: penalty * (sum_i x_i - budget)^2
    # Expanded: penalty * (sum_i x_i)^2 - 2*penalty*budget*sum_i x_i + penalty*budget^2
    # For binary x_i, (sum_i x_i)^2 = sum_i x_i + 2*sum_{i<j} x_i*x_j
    
    # Constant term
    hamiltonian.add_constant(penalty * budget * budget)
    
    # Linear terms: -2*penalty*budget*sum_i x_i + penalty*sum_i x_i
    for i in range(n_assets):
        coef, term = create_z_term(i, penalty * (1 - 2 * budget) / 2)
        hamiltonian.add_term(coef, term)
    
    # Quadratic terms: 2*penalty*sum_{i<j} x_i*x_j
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            coef, term = create_zz_term(i, j, penalty / 2)
            hamiltonian.add_term(coef, term)
    
    return hamiltonian

def get_portfolio_solution(
    bit_string: Union[str, List[int]], 
    returns: List[float], 
    risk_matrix: np.ndarray, 
    budget: int,
    market_conditions: Dict[str, float] = None
) -> Dict:
    """
    Get the Portfolio Optimization solution from a bit string.
    
    Args:
        bit_string: Bit string representation of the solution
        returns: List of expected returns for each asset
        risk_matrix: Covariance matrix representing risk
        budget: Maximum number of assets to select
        market_conditions: Optional dictionary of market condition parameters
        
    Returns:
        Dictionary with solution information
    """
    if isinstance(bit_string, str):
        assignment = [int(bit) for bit in bit_string]
    else:
        assignment = bit_string
    
    n_assets = len(returns)
    
    # Ensure the assignment has the right length
    if len(assignment) < n_assets:
        assignment = assignment + [0] * (n_assets - len(assignment))
    elif len(assignment) > n_assets:
        assignment = assignment[:n_assets]
    
    # Get selected assets
    selected_assets = [i for i, bit in enumerate(assignment) if bit == 1]
    total_selected = len(selected_assets)
    
    # Adjust returns if market conditions are provided
    adjusted_returns = returns
    if market_conditions:
        market_trend = market_conditions.get('market_trend', 0.0)
        volatility = market_conditions.get('volatility', 0.5)
        interest_rate = market_conditions.get('interest_rate', 2.0)
        
        # Apply simple adjustments based on market conditions
        # In a real application, this would be more sophisticated
        adjusted_returns = [
            r * (1 + market_trend * 0.5 - volatility * 0.3 - interest_rate * 0.05)
            for r in returns
        ]
    
    # Calculate expected return
    expected_return = sum(adjusted_returns[i] for i in selected_assets)
    
    # Calculate risk
    risk = 0.0
    for i in selected_assets:
        for j in selected_assets:
            risk += risk_matrix[i, j]
    
    # Check if budget constraint is satisfied
    valid = total_selected <= budget
    
    return {
        "selected_assets": selected_assets,
        "total_selected": total_selected,
        "expected_return": expected_return,
        "risk": risk,
        "return_risk_ratio": expected_return / risk if risk > 0 else float('inf'),
        "valid": valid,
        "market_conditions": market_conditions
    } 