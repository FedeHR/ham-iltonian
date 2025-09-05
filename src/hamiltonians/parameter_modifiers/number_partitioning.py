"""
Parameter modifier functions for the Number Partitioning problem.
"""
import numpy as np
from typing import Callable, Dict

def scaling_modifier(value: float, scaling_factor: float) -> float:
    """
    Scale the number by a factor.
    
    This can help to test the sensitivity of algorithms to the magnitude of the numbers.
    
    Args:
        value: Original number value
        scaling_factor: Factor to scale the number by
        
    Returns:
        Modified number
    """
    return value * scaling_factor

def value_emphasis_modifier(value: float, emphasis_factor: float) -> float:
    """
    Modifier that emphasizes or de-emphasizes number values in the Number Partitioning problem.
    
    This can alter the difficulty of the problem by:
    - Increasing the disparity between numbers (emphasis_factor > 1)
    - Reducing the disparity between numbers (0 < emphasis_factor < 1)
    - Making all numbers uniform (emphasis_factor = 0)
    
    Args:
        value: Original number
        emphasis_factor: Emphasis factor where:
            - emphasis_factor > 1: Increases differences between numbers
            - emphasis_factor = 1: No change (original values)
            - 0 < emphasis_factor < 1: Reduces differences
            - emphasis_factor = 0: All values become equal
        
    Returns:
        Modified value
    """
    if emphasis_factor == 1.0:
        return value
        
    sign = -1 if value < 0 else 1
    abs_value = abs(value)
    
    if emphasis_factor == 0:
        # Make all values equal (preserves sign)
        return sign
    elif emphasis_factor > 1.0:
        # Emphasize differences by raising to a power
        return sign * (abs_value ** emphasis_factor)
    else:
        # De-emphasize differences
        return sign * (abs_value ** emphasis_factor)

def get_modifiers() -> Dict[str, Callable]:
    """
    Get a dictionary of all Number Partitioning modifier functions.
    
    Returns:
        Dictionary mapping modifier names to modifier functions
    """
    return {
        "scaling": scaling_modifier,
        "value_emphasis": value_emphasis_modifier,
    } 