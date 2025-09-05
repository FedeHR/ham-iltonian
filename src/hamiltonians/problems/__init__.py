"""
Combinatorial Optimization Problems.

This package provides classes for representing and solving
combinatorial optimization problems.
"""

from .base import Problem
from .maxcut import MaxCutProblem
from .knapsack import KnapsackProblem
from .tsp import TSPProblem
from .portfolio import PortfolioProblem
from .number_partitioning import NumberPartitioningProblem

# Import instance generators from their new location
from .instance_generators import (
    create_maxcut_instance,
    create_tsp_instance,
    create_knapsack_instance,
    create_portfolio_instance,
    create_number_partitioning_instance
    # create_number_partitioning_instance is not yet implemented
) 