"""
Parameter modifier functions for the Traveling Salesman Problem.
"""
import numpy as np
from typing import Callable, Dict

def traffic_congestion_modifier(distance: float, congestion_factor: float) -> float:
    """
    Modifier that simulates traffic congestion by scaling distances.
    
    Args:
        distance: Original distance between cities
        congestion_factor: Factor representing traffic congestion where:
            - congestion_factor = 1.0: No change (normal traffic)
            - congestion_factor > 1.0: Heavy traffic (increases distances)
            - congestion_factor < 1.0: Light traffic (decreases distances)
            
    Returns:
        Modified distance
    """
    return distance * congestion_factor

def time_of_day_modifier(distance: float, hour_of_day: float) -> float:
    """
    Modifier that adjusts distances based on time of day to simulate varying traffic conditions.
    
    Args:
        distance: Original distance between cities
        hour_of_day: Hour of the day (0-24) where:
            - 7-9: Morning rush hour (increases distances)
            - 16-18: Evening rush hour (increases distances)
            - 23-5: Night hours (decreases distances)
            - Other times: Normal conditions with slight variations
            
    Returns:
        Modified distance
    """
    # Rush hours: 7-9 AM, 4-6 PM (increase distances by up to 50%)
    # Night hours: 11 PM - 5 AM (decrease distances by up to 30%)
    # Otherwise: normal with slight variations
    
    if 7 <= hour_of_day < 9:  # Morning rush hour
        factor = 1.0 + 0.5 * (1.0 - abs(hour_of_day - 8) / 1.0)  # Peak at 8 AM
    elif 16 <= hour_of_day < 18:  # Evening rush hour 
        factor = 1.0 + 0.5 * (1.0 - abs(hour_of_day - 17) / 1.0)  # Peak at 5 PM
    elif hour_of_day >= 23 or hour_of_day < 5:  # Night hours
        if hour_of_day >= 23:
            night_time = hour_of_day - 23
        else:
            night_time = hour_of_day + 1
        factor = 0.7 + 0.3 * (night_time / 6.0)  # Gradually increase from 11 PM to 5 AM
    else:
        # Normal hours: slight variations
        hour_factor = np.sin(hour_of_day * np.pi / 12.0) * 0.1
        factor = 1.0 + hour_factor
    
    return distance * factor

def get_modifiers() -> Dict[str, Callable]:
    """
    Get a dictionary of all TSP modifier functions.
    
    Returns:
        Dictionary mapping modifier names to modifier functions
    """
    return {
        "traffic_congestion": traffic_congestion_modifier,
        "time_of_day": time_of_day_modifier,
    } 