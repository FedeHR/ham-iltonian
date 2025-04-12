"""
Parameter modifier functions for Portfolio Optimization.
"""
import numpy as np
from typing import Callable, Dict

def expected_return_modifier(return_value: float, scaling_factor: float) -> float:
    """
    Modify the expected return of an asset.
    
    Can be used to model optimistic/pessimistic market scenarios.
    
    Args:
        return_value: Original expected return
        scaling_factor: Factor to scale the return by where:
            - scaling_factor > 1.0: Optimistic scenario (higher returns)
            - scaling_factor = 1.0: No change (baseline returns)
            - 0 < scaling_factor < 1.0: Pessimistic scenario (lower returns)
            - scaling_factor < 0: Negative returns (market crash scenario)
            
    Returns:
        Modified expected return
    """
    return return_value * scaling_factor

def risk_aversion_modifier(coefficient: float, risk_factor: float) -> float:
    """
    Modify the risk aversion coefficient to change the risk tolerance.
    
    Args:
        coefficient: Original coefficient (risk aversion lambda or covariance element)
        risk_factor: Risk adjustment factor where:
            - risk_factor > 1.0: More risk-averse (higher penalty for risk)
            - risk_factor = 1.0: No change (baseline risk tolerance)
            - 0 < risk_factor < 1.0: More risk-seeking (lower penalty for risk)
            
    Returns:
        Modified coefficient
    """
    return coefficient * risk_factor

def market_volatility_modifier(covariance: float, volatility_factor: float) -> float:
    """
    Modify the covariance matrix to reflect different market volatility scenarios.
    
    Args:
        covariance: Original covariance value between assets
        volatility_factor: Factor to scale the covariance by where:
            - volatility_factor > 1.0: Higher market volatility
            - volatility_factor = 1.0: No change (baseline volatility)
            - 0 < volatility_factor < 1.0: Lower market volatility
            
    Returns:
        Modified covariance value
    """
    return covariance * volatility_factor

def get_modifiers() -> Dict[str, Callable]:
    """
    Get a dictionary of all Portfolio Optimization modifier functions.
    
    Returns:
        Dictionary mapping modifier names to modifier functions
    """
    return {
        "expected_return": expected_return_modifier,
        "risk_aversion": risk_aversion_modifier,
        "market_volatility": market_volatility_modifier,
    } 