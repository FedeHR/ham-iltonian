"""
Parameter modifier functions for the Knapsack problem.
"""
from typing import Callable, Dict

def linear_value_scaling(value: float, scaling_factor: float, sensitivity: float) -> float:
    """
    Linearly scale the item value by a factor, considering its sensitivity.
    
    Args:
        value: Original item value
        scaling_factor: Factor to scale the value by
        sensitivity: Sensitivity of the item
        
    Returns:
        Modified value
    """
    return value + scaling_factor * sensitivity

def quadratic_value_scaling(value: float, scaling_factor: float, sensitivity: float) -> float:
    """
    Quadratically scale the item value by a factor, considering its sensitivity.
    
    Args:
        value: Original item value
        scaling_factor: Factor to scale the value by
        sensitivity: Sensitivity of the item
        
    Returns:
        Modified value
    """
    return value + (scaling_factor * sensitivity) ** 2

def qubic_value_scaling(value: float, scaling_factor: float, sensitivity: float) -> float:
    """
    Cubically scale the item value by a factor, considering its sensitivity.
    
    Args:
        value: Original item value
        scaling_factor: Factor to scale the value by
        sensitivity: Sensitivity of the item
        
    Returns:
        Modified value
    """
    return value + (scaling_factor * sensitivity) ** 3

def value_scaling_modifier(value: float, scaling_factor: float) -> float:
    """
    Scale item values by a factor.
    
    Args:
        value: Original item value
        scaling_factor: Factor to scale the value by
        
    Returns:
        Modified value
    """
    return value * scaling_factor

def weight_scaling_modifier(weight: float, scaling_factor: float) -> float:
    """
    Scale item weights by a factor.
    
    Args:
        weight: Original item weight
        scaling_factor: Factor to scale the weight by
        
    Returns:
        Modified weight
    """
    return weight * scaling_factor

def capacity_scaling_modifier(capacity: float, scaling_factor: float) -> float:
    """
    Scale knapsack capacity by a factor.
    
    Args:
        capacity: Original capacity
        scaling_factor: Factor to scale the capacity by
        
    Returns:
        Modified capacity
    """
    return capacity * scaling_factor

def penalty_modifier(penalty: float, scaling_factor: float) -> float:
    """
    Adjust the penalty term by a factor.
    
    Args:
        penalty: Original penalty value
        scaling_factor: Factor to scale the penalty by
        
    Returns:
        Modified penalty
    """
    return penalty * scaling_factor

def linear_value_modifier(value: float, offset: float) -> float:
    """
    Add a linear offset to item values.
    
    Args:
        value: Original item value
        offset: Value to add
        
    Returns:
        Modified value
    """
    return value + offset

def linear_weight_modifier(weight: float, offset: float) -> float:
    """
    Add a linear offset to item weights.
    
    Args:
        weight: Original item weight
        offset: Value to add
        
    Returns:
        Modified weight
    """
    return weight + offset


def get_modifiers() -> Dict[str, Callable]:
    """
    Get a dictionary of all Knapsack modifier functions.
    
    Returns:
        Dictionary mapping modifier names to modifier functions
    """
    return {
        "scale_values": value_scaling_modifier,
        "scale_weights": weight_scaling_modifier,
        "scale_capacity": capacity_scaling_modifier,
        "adjust_penalty": penalty_modifier,
        "linear_value": linear_value_modifier,
        "linear_weight": linear_weight_modifier,
        "linear_value_scaling": linear_value_scaling,
        "quadratic_value_scaling": quadratic_value_scaling,
        "qubic_value_scaling": qubic_value_scaling,
    } 