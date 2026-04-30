
from enum import Enum
from dataclasses import dataclass
from typing import Dict

class ActionType(Enum):
    SLIDE_OVER_SURFACE_UNTIL_EDGE = 1                           # get one point and dihedral angle on edge index 0
    SLIDE_AGAINST_EDGE_CCK = 2                                  # slide until edge-vector is estimated
    SLIDE_AGAINST_EDGE_CK = 3
    SLIDE_OVER_SURFACE_PARALLEL_TO_EDGE_CCK = 4                 # get one point on next edge
    SLIDE_OVER_SURFACE_PARALLEL_TO_EDGE_CK = 5
    SLIDE_AGAINST_EDGE_UNTIL_CORNER_CCK = 6                     # slide until corner to know if next corner angle is reflexive
    SLIDE_AGAINST_EDGE_UNTIL_CORNER_CK = 7
    SLIDE_AGAINST_VERTICAL_SURFACE_CCK = 8                      # when dihedral angle is 90 degrees, slide against adjacent surface
    SLIDE_AGAINST_VERTICAL_SURFACE_CK = 9
    SLIDE_AGAINST_VERTICAL_SURFACE_UNTIL_CORNER_CCK = 10
    SLIDE_AGAINST_VERTICAL_SURFACE_UNTIL_CORNER_CK = 11
    MOVE_PARALLEL_FROM_OUTSIDE_TO_EDGE_UNTIL_CONTACT_CCK = 12   # when dihedral is 270 and previous action was to move until corner, move in (-edge_vector) of edge explored in previous action
    MOVE_PARALLEL_FROM_OUTSIDE_TO_EDGE_UNTIL_CONTACT_CK = 13
    SLIDE_OVER_SURFACE_PARALLEL_FROM_OUTSIDE_TO_EDGE_CK = 14
    SLIDE_OVER_SURFACE_PARALLEL_FROM_OUTSIDE_TO_EDGE_CCK = 15
    SLIDE_OVER_SURFACE_PERPENDICULAR_TO_EDGE_GIVEN_ONE_POINT = 16


class Direction(Enum):
    CCK = "CCK"
    CK = "CK"
    UNKNOWN = "UNKNOWN"


class Mode(Enum):
    AGAINST_EDGE = "AGAINST_EDGE"
    AGAINST_VERTICAL = "AGAINST_VERTICAL"
    PARALLEL_OVER_SURFACE = "PARALLEL_OVER_SURFACE"
    OVER_SURFACE = "OVER_SURFACE"
    PERPENDICULAR_TO_EDGE_OVER_SURFACE = "PERPENDICULAR_TO_EDGE_OVER_SURFACE"
    PARALLEL_OVER_SURFACE_FROM_OUTSIDE = "PARALLEL_OVER_SURFACE_FROM_OUTSIDE"
    PARALLEL_IN_FREE_SPACE_FROM_OUTSIDE = "PARALLEL_IN_FREE_SPACE_FROM_OUTSIDE"
    UNKNOWN = "UNKNOWN"


class Stop(Enum):
    VECTOR_ONLY = "VECTOR_ONLY"
    UNTIL_CORNER = "UNTIL_CORNER"
    UNTIL_EDGE_CONTACT = "UNTIL_EDGE_CONTACT"
    UNKNOWN = "UNKNOWN"


class TraversalDirection(Enum):
    CLOCKWISE = 0
    COUNTERCLOCKWISE = 1


@dataclass(frozen=True)
class ActionSpec:
    direction: Direction
    mode: Mode
    stop: Stop


@dataclass(frozen=True)
class ActionInstance:
    action_type: ActionType
    edge_index: int


SPEC_TO_ACTION: Dict[ActionSpec, ActionType] = {
    ActionSpec(Direction.CCK, Mode.PERPENDICULAR_TO_EDGE_OVER_SURFACE, Stop.UNTIL_EDGE_CONTACT): ActionType.SLIDE_OVER_SURFACE_PERPENDICULAR_TO_EDGE_GIVEN_ONE_POINT,
    ActionSpec(Direction.CCK, Mode.OVER_SURFACE, Stop.UNTIL_EDGE_CONTACT): ActionType.SLIDE_OVER_SURFACE_UNTIL_EDGE,
    ActionSpec(Direction.CCK, Mode.AGAINST_EDGE, Stop.VECTOR_ONLY): ActionType.SLIDE_AGAINST_EDGE_CCK,
    ActionSpec(Direction.CK,  Mode.AGAINST_EDGE, Stop.VECTOR_ONLY): ActionType.SLIDE_AGAINST_EDGE_CK,
    ActionSpec(Direction.CCK, Mode.AGAINST_EDGE, Stop.UNTIL_CORNER): ActionType.SLIDE_AGAINST_EDGE_UNTIL_CORNER_CCK,
    ActionSpec(Direction.CK,  Mode.AGAINST_EDGE, Stop.UNTIL_CORNER): ActionType.SLIDE_AGAINST_EDGE_UNTIL_CORNER_CK,
    ActionSpec(Direction.CCK, Mode.PARALLEL_OVER_SURFACE, Stop.UNTIL_EDGE_CONTACT): ActionType.SLIDE_OVER_SURFACE_PARALLEL_TO_EDGE_CCK,
    ActionSpec(Direction.CK,  Mode.PARALLEL_OVER_SURFACE, Stop.UNTIL_EDGE_CONTACT): ActionType.SLIDE_OVER_SURFACE_PARALLEL_TO_EDGE_CK,
    ActionSpec(Direction.CCK, Mode.PARALLEL_OVER_SURFACE_FROM_OUTSIDE, Stop.UNTIL_EDGE_CONTACT): ActionType.SLIDE_OVER_SURFACE_PARALLEL_FROM_OUTSIDE_TO_EDGE_CCK,
    ActionSpec(Direction.CK,  Mode.PARALLEL_OVER_SURFACE_FROM_OUTSIDE, Stop.UNTIL_EDGE_CONTACT): ActionType.SLIDE_OVER_SURFACE_PARALLEL_FROM_OUTSIDE_TO_EDGE_CK,
    ActionSpec(Direction.CCK, Mode.AGAINST_VERTICAL, Stop.VECTOR_ONLY): ActionType.SLIDE_AGAINST_VERTICAL_SURFACE_CCK,
    ActionSpec(Direction.CK,  Mode.AGAINST_VERTICAL, Stop.VECTOR_ONLY): ActionType.SLIDE_AGAINST_VERTICAL_SURFACE_CK,
    ActionSpec(Direction.CCK, Mode.AGAINST_VERTICAL, Stop.UNTIL_CORNER): ActionType.SLIDE_AGAINST_VERTICAL_SURFACE_UNTIL_CORNER_CCK,
    ActionSpec(Direction.CK,  Mode.AGAINST_VERTICAL, Stop.UNTIL_CORNER): ActionType.SLIDE_AGAINST_VERTICAL_SURFACE_UNTIL_CORNER_CK,
    ActionSpec(Direction.CK,  Mode.PARALLEL_IN_FREE_SPACE_FROM_OUTSIDE, Stop.UNTIL_EDGE_CONTACT): ActionType.MOVE_PARALLEL_FROM_OUTSIDE_TO_EDGE_UNTIL_CONTACT_CK,
    ActionSpec(Direction.CCK, Mode.PARALLEL_IN_FREE_SPACE_FROM_OUTSIDE, Stop.UNTIL_EDGE_CONTACT): ActionType.MOVE_PARALLEL_FROM_OUTSIDE_TO_EDGE_UNTIL_CONTACT_CCK
}


ACTION_TO_SPEC: Dict[ActionType, ActionSpec] = {
    action: spec for spec, action in SPEC_TO_ACTION.items()
}