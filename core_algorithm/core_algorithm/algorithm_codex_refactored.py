import copy
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

from .data_structures import (
    SPEC_TO_ACTION,
    ActionInstance,
    ActionSpec,
    ActionType,
    Direction,
    Mode,
    Stop,
)
from .helper import (
    action_spec_from_action,
    are_adjacent,
    are_adjacent_and_action_in_order,
    edge_unit_vector_and_atleast_one_point_known,
    find_dof,
    flip_direction,
    get_random_points_on_line,
    has_unit_vectors_and_points_for_all_edges,
)
from .polygon_knowledge import PolygonKnowledge
from .algorithm_inference import (
    feasible_bounded_lengths,
    fill_missing_parameters,
    find_unique_pattern,
    get_unique_pattern_ref_index,
    propagate_parameters,
    rearrange_rck_using_prior_knowledge,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActionCandidate:
    """
    Fully specified action option produced before final scoring/selection.
    """

    action_type: ActionType
    reference_edge: int
    target_edge: int
    action_spec: ActionSpec
    score: float
    reason: str


@dataclass(frozen=True)
class DecisionContext:
    """
    Immutable inputs shared by candidate generation and scoring helpers.
    """

    know: PolygonKnowledge
    prev_action_type: Optional[ActionType]
    prev_action_spec: Optional[ActionSpec]
    prev_action_edge_idx: Optional[int]
    rck_rearranged: bool
    in_simulation: bool
    gt: Optional[PolygonKnowledge]


def _action_type_from_spec(action_spec: ActionSpec) -> ActionType:
    """
    Convert a structured action specification to the exported action enum.
    """

    try:
        return SPEC_TO_ACTION[action_spec]
    except KeyError as exc:
        raise NotImplementedError(f"Unsupported action spec {action_spec}") from exc


def _candidate(ctx: DecisionContext,
               target_edge: int,
               reference_edge: int,
               action_spec: ActionSpec,
               reason: str) -> Optional[ActionCandidate]:
    """
    Build and score an action candidate, rejecting unsupported specs.
    """

    try:
        action_type = _action_type_from_spec(action_spec)
    except NotImplementedError:
        logger.debug("Rejected unsupported action spec: %s", action_spec)
        return None

    score = _score_candidate(ctx, target_edge, reference_edge, action_spec)
    return ActionCandidate(
        action_type=action_type,
        reference_edge=reference_edge,
        target_edge=target_edge,
        action_spec=action_spec,
        score=score,
        reason=reason,
    )


def _target_edges_to_explore(ctx: DecisionContext) -> list[int]:
    """
    Return edges whose vector or contact points still need exploration.

    Before prior/current alignment, this preserves the original sequential
    behavior. After alignment, any edge adjacent to known geometry is eligible.
    """

    know = ctx.know
    targets: list[int] = []

    for edge_idx in range(know.n_sides):
        prev_idx = (edge_idx - 1) % know.n_sides
        next_idx = (edge_idx + 1) % know.n_sides

        edge_unknown = know.edge_unit_vectors[edge_idx] is None
        no_points_on_edge = len(know.get_all_points_on_edge(edge_idx)) == 0
        if not edge_unknown and not no_points_on_edge:
            continue

        prev_ready = edge_unit_vector_and_atleast_one_point_known(know, prev_idx)
        next_ready = edge_unit_vector_and_atleast_one_point_known(know, next_idx)

        if not ctx.rck_rearranged:
            if (edge_idx == 0 and edge_unknown) or prev_ready:
                return [edge_idx]
        elif prev_ready or next_ready:
            targets.append(edge_idx)

    return targets


def _preferred_direction(ctx: DecisionContext, target_edge: int) -> Direction:
    """
    Choose a traversal direction that prioritizes unknown adjacent reflexivity.
    """

    know = ctx.know
    next_idx = (target_edge + 1) % know.n_sides

    if know.is_reflexive_angle[next_idx] is None:
        return Direction.CCK
    if know.is_reflexive_angle[target_edge] is None:
        return Direction.CK
    if ctx.prev_action_spec and ctx.prev_action_spec.direction in (Direction.CCK, Direction.CK):
        return ctx.prev_action_spec.direction
    return Direction.CCK


def _stop_for_edge_traversal(ctx: DecisionContext,
                             target_edge: int,
                             direction: Direction) -> Stop:
    """
    Choose whether edge traversal should stop at vector-only or corner contact.
    """

    know = ctx.know
    prev_idx = (target_edge - 1) % know.n_sides
    next_idx = (target_edge + 1) % know.n_sides

    if direction == Direction.CCK:
        adjacent_edge = next_idx
        corner_idx = next_idx
    else:
        adjacent_edge = prev_idx
        corner_idx = target_edge

    if edge_unit_vector_and_atleast_one_point_known(know, adjacent_edge):
        return Stop.VECTOR_ONLY
    if know.is_reflexive_angle[corner_idx] is not None and know.corners[corner_idx] is not None:
        return Stop.VECTOR_ONLY
    return Stop.UNTIL_CORNER


def _mode_for_dihedral(dihedral: Optional[float]) -> Optional[Mode]:
    """
    Map a known dihedral angle to the corresponding edge-following mode.
    """

    if dihedral == 90 or dihedral == 90.0:
        return Mode.AGAINST_VERTICAL
    if dihedral == 270 or dihedral == 270.0:
        return Mode.AGAINST_EDGE
    return None


def _candidates_from_current_contact(ctx: DecisionContext,
                                     target_edge: int) -> list[ActionCandidate]:
    """
    Generate candidates that continue naturally from the previous edge contact.
    """

    candidates: list[ActionCandidate] = []
    if ctx.prev_action_spec is None:
        return candidates

    know = ctx.know
    prev_idx = (target_edge - 1) % know.n_sides
    next_idx = (target_edge + 1) % know.n_sides

    adjacent_and_ordered = (
        ctx.prev_action_edge_idx is not None
        and are_adjacent_and_action_in_order(
            ctx.prev_action_spec, target_edge, ctx.prev_action_edge_idx, know.n_sides
        )
    )
    first_edge_case = target_edge == 0 and ctx.prev_action_edge_idx in (None, 0)
    from_outside = ctx.prev_action_spec.mode in (
        Mode.PARALLEL_IN_FREE_SPACE_FROM_OUTSIDE,
        Mode.PARALLEL_OVER_SURFACE_FROM_OUTSIDE,
    )

    if not ((adjacent_and_ordered and not from_outside)
            or first_edge_case
            or (not adjacent_and_ordered and from_outside)):
        return candidates

    target_dihedral = know.dihedrals[target_edge]
    if ctx.prev_action_spec.stop == Stop.UNTIL_EDGE_CONTACT:
        if (
            ctx.prev_action_spec.mode in (
                Mode.OVER_SURFACE,
                Mode.PARALLEL_OVER_SURFACE,
                Mode.PARALLEL_OVER_SURFACE_FROM_OUTSIDE,
            )
            and target_dihedral == 270
        ):
            if ctx.prev_action_spec.direction == Direction.CCK:
                direction = Direction.CK
                reference_edge = prev_idx
            else:
                direction = Direction.CCK
                reference_edge = next_idx
            action_spec = ActionSpec(
                direction,
                Mode.PARALLEL_IN_FREE_SPACE_FROM_OUTSIDE,
                Stop.UNTIL_EDGE_CONTACT,
            )
            candidate = _candidate(
                ctx,
                target_edge,
                reference_edge,
                action_spec,
                "continue from edge contact by approaching a 270-degree edge from outside",
            )
            if candidate:
                candidates.append(candidate)

        mode = _mode_for_dihedral(target_dihedral)
        if mode is not None:
            direction = _preferred_direction(ctx, target_edge)
            stop = _stop_for_edge_traversal(ctx, target_edge, direction)
            candidate = _candidate(
                ctx,
                target_edge,
                target_edge,
                ActionSpec(direction, mode, stop),
                "continue from edge contact along the target edge",
            )
            if candidate:
                candidates.append(candidate)

    if ctx.prev_action_spec.stop == Stop.UNTIL_CORNER:
        candidates.extend(_candidates_from_previous_corner(ctx, target_edge))

    return candidates


def _candidates_from_previous_corner(ctx: DecisionContext,
                                     target_edge: int) -> list[ActionCandidate]:
    """
    Generate candidates that use a previous corner stop to enter the target edge.
    """

    candidates: list[ActionCandidate] = []
    if ctx.prev_action_spec is None or ctx.prev_action_edge_idx is None:
        return candidates

    know = ctx.know
    prev_idx = (target_edge - 1) % know.n_sides
    next_idx = (target_edge + 1) % know.n_sides
    target_dihedral = know.dihedrals[target_edge]

    if ctx.prev_action_edge_idx == prev_idx:
        reference_edge = prev_idx
        direction = Direction.CCK
        corner_reflexive = know.is_reflexive_angle[target_edge]
        adjacent_edge = next_idx
        adjacent_corner = next_idx
    elif ctx.prev_action_edge_idx == next_idx:
        reference_edge = next_idx
        direction = Direction.CK
        corner_reflexive = know.is_reflexive_angle[next_idx]
        adjacent_edge = prev_idx
        adjacent_corner = target_edge
    else:
        return candidates

    mode = None
    if corner_reflexive is True:
        if target_dihedral == 270:
            if ctx.prev_action_spec.mode == Mode.AGAINST_VERTICAL:
                mode = Mode.PARALLEL_OVER_SURFACE_FROM_OUTSIDE
                direction = flip_direction(direction)
            elif ctx.prev_action_spec.mode == Mode.AGAINST_EDGE:
                mode = Mode.AGAINST_EDGE
                reference_edge = target_edge
        elif target_dihedral == 90 or target_dihedral is None:
            mode = Mode.PARALLEL_OVER_SURFACE_FROM_OUTSIDE
            direction = flip_direction(direction)
    elif corner_reflexive is False:
        if target_dihedral == 90:
            if ctx.prev_action_spec.mode == Mode.AGAINST_VERTICAL:
                mode = Mode.AGAINST_VERTICAL
                reference_edge = target_edge
            elif ctx.prev_action_spec.mode == Mode.AGAINST_EDGE:
                mode = Mode.PARALLEL_OVER_SURFACE
        elif target_dihedral is None:
            mode = Mode.PARALLEL_OVER_SURFACE
        elif target_dihedral == 270:
            mode = Mode.PARALLEL_IN_FREE_SPACE_FROM_OUTSIDE
            direction = flip_direction(direction)

    if mode is None:
        return candidates

    if mode in (
        Mode.PARALLEL_IN_FREE_SPACE_FROM_OUTSIDE,
        Mode.PARALLEL_OVER_SURFACE,
        Mode.PERPENDICULAR_TO_EDGE_OVER_SURFACE,
        Mode.PARALLEL_OVER_SURFACE_FROM_OUTSIDE,
    ):
        stop = Stop.UNTIL_EDGE_CONTACT
    elif edge_unit_vector_and_atleast_one_point_known(know, adjacent_edge):
        stop = Stop.VECTOR_ONLY
    elif know.is_reflexive_angle[adjacent_corner] is not None and know.corners[adjacent_corner] is not None:
        stop = Stop.VECTOR_ONLY
    else:
        stop = Stop.UNTIL_CORNER

    candidate = _candidate(
        ctx,
        target_edge,
        reference_edge,
        ActionSpec(direction, mode, stop),
        "continue from previous corner contact",
    )
    return [candidate] if candidate else []


def _candidates_from_adjacent_references(ctx: DecisionContext,
                                         target_edge: int) -> list[ActionCandidate]:
    """
    Generate fallback candidates from adjacent edges with known vectors/points.
    """

    candidates: list[ActionCandidate] = []
    know = ctx.know
    prev_idx = (target_edge - 1) % know.n_sides
    next_idx = (target_edge + 1) % know.n_sides

    for reference_edge, direction, corner_idx in (
        (prev_idx, Direction.CCK, target_edge),
        (next_idx, Direction.CK, next_idx),
    ):
        if not edge_unit_vector_and_atleast_one_point_known(know, reference_edge):
            continue

        corner_reflexive = know.is_reflexive_angle[corner_idx]
        corner = know.corners[corner_idx]
        reference_dihedral = know.dihedrals[reference_edge]

        if corner_reflexive is None:
            if reference_dihedral is None:
                mode = Mode.PERPENDICULAR_TO_EDGE_OVER_SURFACE
                stop = Stop.UNTIL_EDGE_CONTACT
            elif reference_dihedral == 270:
                mode = Mode.AGAINST_EDGE
                stop = Stop.UNTIL_CORNER
            elif reference_dihedral == 90:
                mode = Mode.AGAINST_VERTICAL
                stop = Stop.UNTIL_CORNER
            else:
                continue
        elif corner_reflexive is False:
            mode = Mode.PARALLEL_OVER_SURFACE
            stop = Stop.UNTIL_EDGE_CONTACT
        else:
            if corner is not None:
                mode = Mode.PARALLEL_OVER_SURFACE_FROM_OUTSIDE
                stop = Stop.UNTIL_EDGE_CONTACT
                direction = flip_direction(direction)
            elif reference_dihedral == 270:
                mode = Mode.AGAINST_EDGE
                stop = Stop.UNTIL_CORNER
            elif reference_dihedral == 90:
                mode = Mode.AGAINST_VERTICAL
                stop = Stop.UNTIL_CORNER
            else:
                continue

        candidate = _candidate(
            ctx,
            target_edge,
            reference_edge,
            ActionSpec(direction, mode, stop),
            "explore target edge from an adjacent known reference edge",
        )
        if candidate:
            candidates.append(candidate)

    return candidates


def _all_candidates(ctx: DecisionContext) -> list[ActionCandidate]:
    """
    Generate every feasible action candidate for the current knowledge state.
    """

    know = ctx.know

    if has_unit_vectors_and_points_for_all_edges(know):
        candidates: list[ActionCandidate] = []
        for edge_idx in range(know.n_sides):
            if know.dihedrals[edge_idx] is None:
                candidate = _candidate(
                    ctx,
                    edge_idx,
                    edge_idx,
                    ActionSpec(
                        Direction.CCK,
                        Mode.PERPENDICULAR_TO_EDGE_OVER_SURFACE,
                        Stop.UNTIL_EDGE_CONTACT,
                    ),
                    "all edges are geometrically known; measure missing dihedral",
                )
                if candidate:
                    candidates.append(candidate)
        return candidates

    if all(v is None for v in know.edge_unit_vectors):
        already_at_first_edge = (
            ctx.prev_action_spec is not None
            and ctx.prev_action_spec.stop == Stop.UNTIL_EDGE_CONTACT
        )
        if not already_at_first_edge:
            candidate = _candidate(
                ctx,
                0,
                0,
                ActionSpec(Direction.CCK, Mode.OVER_SURFACE, Stop.UNTIL_EDGE_CONTACT),
                "default first exploratory contact",
            )
            return [candidate] if candidate else []

    candidates = []
    for target_edge in _target_edges_to_explore(ctx):
        candidates.extend(_candidates_from_current_contact(ctx, target_edge))
        candidates.extend(_candidates_from_adjacent_references(ctx, target_edge))

    return candidates


def _score_candidate(ctx: DecisionContext,
                     target_edge: int,
                     reference_edge: int,
                     action_spec: ActionSpec) -> float:
    """
    Score a candidate by expected information gain and execution continuity.
    """

    know = ctx.know
    score = 0.0

    if know.edge_unit_vectors[target_edge] is None:
        score += 30.0
    if len(know.get_all_points_on_edge(target_edge)) == 0:
        score += 20.0
    if know.dihedrals[target_edge] is None:
        score += 8.0

    next_idx = (target_edge + 1) % know.n_sides
    if know.is_reflexive_angle[target_edge] is None:
        score += 6.0
    if know.is_reflexive_angle[next_idx] is None:
        score += 6.0
    if know.corners[target_edge] is None:
        score += 4.0
    if know.corners[next_idx] is None:
        score += 4.0

    if action_spec.stop == Stop.UNTIL_EDGE_CONTACT:
        score += 10.0
    elif action_spec.stop == Stop.UNTIL_CORNER:
        score += 7.0
    elif action_spec.stop == Stop.VECTOR_ONLY:
        score += 4.0

    if edge_unit_vector_and_atleast_one_point_known(know, reference_edge):
        score += 8.0
    if know.dihedrals[reference_edge] is not None:
        score += 3.0

    if ctx.prev_action_edge_idx is not None:
        if reference_edge == ctx.prev_action_edge_idx:
            score += 8.0
        if are_adjacent(target_edge, ctx.prev_action_edge_idx, know.n_sides):
            score += 4.0
        if (
            ctx.prev_action_spec is not None
            and are_adjacent(target_edge, ctx.prev_action_edge_idx, know.n_sides)
            and are_adjacent_and_action_in_order(
                ctx.prev_action_spec,
                target_edge,
                ctx.prev_action_edge_idx,
                know.n_sides,
            )
        ):
            score += 6.0

    if action_spec.mode in (
        Mode.PARALLEL_IN_FREE_SPACE_FROM_OUTSIDE,
        Mode.PARALLEL_OVER_SURFACE_FROM_OUTSIDE,
    ):
        score -= 2.0

    score += _simulated_dof_gain(ctx, target_edge)
    return score


def _simulated_dof_gain(ctx: DecisionContext, target_edge: int) -> float:
    """
    Estimate candidate value in simulation by measuring expected DOF reduction.
    """

    if not ctx.in_simulation or ctx.gt is None:
        return 0.0

    know = ctx.know
    next_idx = (target_edge + 1) % know.n_sides
    if ctx.gt.corners[target_edge] is None or ctx.gt.corners[next_idx] is None:
        return 0.0

    temp_know = copy.deepcopy(know)
    sampled_points = get_random_points_on_line(
        ctx.gt.corners[next_idx],
        ctx.gt.corners[target_edge],
        num_points=2,
    )
    temp_know.internal_points_on_edge[target_edge] = copy.deepcopy(sampled_points)
    temp_know.is_reflexive_angle[target_edge] = ctx.gt.is_reflexive_angle[target_edge]
    before = find_dof(know)
    propagate_parameters(temp_know)
    after = find_dof(temp_know)
    return max(0, before - after) * 12.0


def select_action_candidate(know: PolygonKnowledge,
                            prev_action_instance: ActionInstance | None,
                            rck_rearranged: bool,
                            in_simulation: bool = False,
                            gt: PolygonKnowledge = None) -> Optional[ActionCandidate]:
    """
    Return the highest-scoring candidate for debugging and analysis.

    `next_action` keeps the original public return shape and calls this helper.
    """
    prev_action_type = prev_action_instance.action_type if prev_action_instance is not None else None
    prev_action_spec = action_spec_from_action(prev_action_type) if prev_action_type is not None else None
    prev_action_edge_idx = prev_action_instance.edge_index if prev_action_instance is not None else None

    ctx = DecisionContext(
        know=know,
        prev_action_type=prev_action_type,
        prev_action_spec=prev_action_spec,
        prev_action_edge_idx=prev_action_edge_idx,
        rck_rearranged=rck_rearranged,
        in_simulation=in_simulation,
        gt=gt,
    )

    candidates = _all_candidates(ctx)
    if not candidates:
        return None

    candidates.sort(
        key=lambda candidate: (
            candidate.score,
            -candidate.target_edge,
            -candidate.reference_edge,
            -candidate.action_type.value,
        ),
        reverse=True,
    )
    return candidates[0]


def next_action(know: PolygonKnowledge,
                prev_action_instance: ActionInstance | None,
                rck_rearranged: bool,
                in_simulation: bool = False,
                gt: PolygonKnowledge = None) -> Tuple[Optional[ActionType], Optional[int]]:
    """
    Candidate-generation and scoring implementation of the action selector.

    The public return shape intentionally matches `algorithm.next_action`.
    """
    selected = select_action_candidate(
        know,
        prev_action_instance,
        rck_rearranged,
        in_simulation=in_simulation,
        gt=gt,
    )
    if selected is None:
        return None, None
    logger.info(
        "Selected %s on reference edge %s for target edge %s: %.2f (%s)",
        selected.action_type.name,
        selected.reference_edge,
        selected.target_edge,
        selected.score,
        selected.reason,
    )
    return selected.action_type, selected.reference_edge
