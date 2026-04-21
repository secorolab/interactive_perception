"""
Core Algorithm Package

Shared, reusable components for polygon knowledge representation and motion reasoning.
This package is used by both the simulation and robot controller modules.

Key Components:
- polygon_knowledge: Data structure for polygon geometry
- data_structures: Enums and data classes for motion planning
- helper: Utility functions
- algorithm: Main motion selection and reasoning logic

This package has no ROS dependencies
"""

from .polygon_knowledge import PolygonKnowledge
from .data_structures import (
    ActionType,
    ActionInstance,
    ActionSpec,
    Direction,
    Mode,
    Stop,
    TraversalDirection,
    SPEC_TO_ACTION,
    ACTION_TO_SPEC
)
from .helper import find_dof, is_close, point_in_list, are_list_elements_uniform, get_random_points_on_line, get_unit_vector, generate_internal_angles
from .algorithm import (
    next_action,
    propagate_parameters,
    feasible_bounded_lengths,
    find_unique_pattern,
    get_unique_pattern_ref_index,
    rearrange_rck_using_prior_knowledge,
    fill_missing_parameters,
)

__all__ = [
    "PolygonKnowledge",
    "ActionType",
    "ActionInstance",
    "ActionSpec",
    "Direction",
    "Mode",
    "Stop",
    "TraversalDirection",
    "SPEC_TO_ACTION",
    "ACTION_TO_SPEC",
    "find_dof",
    "is_close",
    "point_in_list",
    "next_action",
    "propagate_parameters",
    "feasible_bounded_lengths",
    "find_unique_pattern",
    "get_unique_pattern_ref_index",
    "rearrange_rck_using_prior_knowledge",
    "fill_missing_parameters",
    "get_random_points_on_line",
    "get_unit_vector",
    "generate_internal_angles",
    "point_in_list",
    "are_list_elements_uniform"
]

__version__ = "1.0.0"
