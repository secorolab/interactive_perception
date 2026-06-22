"""Symbolic constraint propagation and action-outcome scoring for polygon exploration.

This module deliberately does not estimate coordinates or create synthetic point
values.  It represents which geometric quantities are determined, which
discrete outcomes remain possible, and which directly observed edge lines are
available.  It can therefore score informative actions on the robot without
ground-truth geometry.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import sympy as sp

from .data_structures import ACTION_TO_SPEC, ActionInstance, ActionType, Direction, Stop
from .helper import action_spec_from_action, find_dof
from .polygon_knowledge import PolygonKnowledge


_DIHEDRAL_DOMAIN = frozenset((90.0, 270.0))
_REFLEXIVITY_DOMAIN = frozenset((False, True))


@dataclass(frozen=True)
class SurfaceEquationSystem:
    """Symbolic variables and planar polygon closure equations.

    The equations encode the continuous geometry.  Discrete dihedral and
    reflexivity state is intentionally kept in :class:`SymbolicKnowledge`,
    because it changes action outcomes rather than the planar edge equations.
    """

    corner_x: tuple[sp.Symbol, ...]
    corner_y: tuple[sp.Symbol, ...]
    edge_heading: tuple[sp.Symbol, ...]
    edge_length: tuple[sp.Symbol, ...]
    corner_angle: tuple[sp.Symbol, ...]
    equations: tuple[sp.Expr, ...]


def build_surface_equations(n_sides: int) -> SurfaceEquationSystem:
    """Build a value-free symbolic representation of a closed planar polygon.

    Edge ``i`` starts at corner ``i`` and ends at corner ``i + 1``.  Each edge
    is represented by a heading and length instead of an explicit unit vector.
    The returned residual expressions are constrained to zero.
    """
    if n_sides < 3:
        raise ValueError("A polygon requires at least three sides")

    corner_x = sp.symbols(f"x0:{n_sides}")
    corner_y = sp.symbols(f"y0:{n_sides}")
    edge_heading = sp.symbols(f"theta0:{n_sides}")
    edge_length = sp.symbols(f"length0:{n_sides}", positive=True)
    corner_angle = sp.symbols(f"alpha0:{n_sides}")

    equations: list[sp.Expr] = []
    for edge_idx in range(n_sides):
        next_idx = (edge_idx + 1) % n_sides
        prev_idx = (edge_idx - 1) % n_sides
        equations.append(
            corner_x[next_idx]
            - corner_x[edge_idx]
            - edge_length[edge_idx] * sp.cos(edge_heading[edge_idx])
        )
        equations.append(
            corner_y[next_idx]
            - corner_y[edge_idx]
            - edge_length[edge_idx] * sp.sin(edge_heading[edge_idx])
        )
        equations.append(
            edge_heading[edge_idx]
            - edge_heading[prev_idx]
            - (sp.pi - corner_angle[edge_idx])
        )

    equations.append(sum(
        edge_length[i] * sp.cos(edge_heading[i]) for i in range(n_sides)
    ))
    equations.append(sum(
        edge_length[i] * sp.sin(edge_heading[i]) for i in range(n_sides)
    ))

    return SurfaceEquationSystem(
        corner_x=tuple(corner_x),
        corner_y=tuple(corner_y),
        edge_heading=tuple(edge_heading),
        edge_length=tuple(edge_length),
        corner_angle=tuple(corner_angle),
        equations=tuple(equations),
    )


@dataclass
class SymbolicKnowledge:
    """Value-free state of determined fields, domains, and direct observations."""

    n_sides: int
    edge_vector_known: list[bool]
    length_known: list[bool]
    corner_known: list[bool]
    corner_angle_known: list[bool]
    dihedral_domains: list[frozenset[float]]
    reflexivity_domains: list[frozenset[bool]]
    direct_point_observed: list[bool]
    direct_line_observed: list[bool]
    adjacent_nonparallel: list[bool]
    derivations: list[str] = field(default_factory=list)

    def clone(self) -> "SymbolicKnowledge":
        """Return an independent copy suitable for candidate outcome evaluation."""
        return copy.deepcopy(self)

    def dihedral_is_known(self, edge_idx: int) -> bool:
        """Return whether edge ``edge_idx`` has a singleton dihedral domain."""
        return len(self.dihedral_domains[edge_idx]) == 1

    def reflexivity_is_known(self, corner_idx: int) -> bool:
        """Return whether corner ``corner_idx`` has a singleton reflexivity domain."""
        return len(self.reflexivity_domains[corner_idx]) == 1

    def add_derivation(self, message: str) -> None:
        """Record a concise explanation for a newly inferred symbolic fact."""
        self.derivations.append(message)


def _vector_pair_is_nonparallel(vector_a: tuple[float, float],
                                vector_b: tuple[float, float],
                                tolerance: float = 1e-6) -> bool:
    """Return whether two known numeric vectors encode a nonparallel relation."""
    norm_product = math.hypot(*vector_a) * math.hypot(*vector_b)
    if norm_product == 0.0:
        return False
    cross = vector_a[0] * vector_b[1] - vector_a[1] * vector_b[0]
    return abs(cross) > tolerance * norm_product


def symbolic_from_polygon_knowledge(knowledge: PolygonKnowledge) -> SymbolicKnowledge:
    """Project numeric knowledge into a value-free symbolic constraint state.

    Numeric values are used only to classify already known fields and relations
    such as whether two observed vectors are nonparallel.  A known reflexivity
    classification also guarantees a nonstraight corner: this codebase assigns
    ``False`` only for angles below 180 degrees and ``True`` only for angles
    above 180 degrees.  The resulting state retains no coordinates, lengths,
    slopes, or continuous vector values.
    """
    n_sides = knowledge.n_sides
    edge_vector_known = [vector is not None for vector in knowledge.edge_unit_vectors]
    length_known = [length is not None for length in knowledge.lengths]
    corner_known = [corner is not None for corner in knowledge.corners]
    corner_angle_known = [angle is not None for angle in knowledge.corner_angles]
    direct_point_observed = [
        len(knowledge.internal_points_on_edge[i]) > 0
        for i in range(n_sides)
    ]
    direct_line_observed = [
        edge_vector_known[i] and direct_point_observed[i]
        for i in range(n_sides)
    ]

    dihedral_domains = [
        frozenset((float(dihedral),)) if dihedral is not None else _DIHEDRAL_DOMAIN
        for dihedral in knowledge.dihedrals
    ]
    reflexivity_domains: list[frozenset[bool]] = []
    for index, reflexivity in enumerate(knowledge.is_reflexive_angle):
        if reflexivity is not None:
            reflexivity_domains.append(frozenset((bool(reflexivity),)))
        elif knowledge.corner_angles[index] is not None:
            reflexivity_domains.append(
                frozenset((knowledge.corner_angles[index] > 180.0,))
            )
        else:
            reflexivity_domains.append(_REFLEXIVITY_DOMAIN)

    adjacent_nonparallel = [False] * n_sides
    for index in range(n_sides):
        previous_index = (index - 1) % n_sides
        previous_vector = knowledge.edge_unit_vectors[previous_index]
        current_vector = knowledge.edge_unit_vectors[index]
        corner_angle = knowledge.corner_angles[index]
        if previous_vector is not None and current_vector is not None:
            adjacent_nonparallel[index] = _vector_pair_is_nonparallel(
                previous_vector,
                current_vector,
            )
        elif len(reflexivity_domains[index]) == 1:
            # A known reflexivity result excludes a 180-degree joint. This is
            # needed when an adjacent vector will be measured by a candidate
            # action, but that vector's numeric value is not retained here.
            adjacent_nonparallel[index] = True
        elif corner_angle is not None:
            adjacent_nonparallel[index] = not math.isclose(
                corner_angle,
                180.0,
                abs_tol=1e-6,
            )

    return SymbolicKnowledge(
        n_sides=n_sides,
        edge_vector_known=edge_vector_known,
        length_known=length_known,
        corner_known=corner_known,
        corner_angle_known=corner_angle_known,
        dihedral_domains=dihedral_domains,
        reflexivity_domains=reflexivity_domains,
        direct_point_observed=direct_point_observed,
        direct_line_observed=direct_line_observed,
        adjacent_nonparallel=adjacent_nonparallel,
    )


def propagate_symbolic(knowledge: SymbolicKnowledge) -> bool:
    """Apply value-free geometric determinacy rules until reaching closure.

    The rules mirror the numerical propagator where possible: edge direction
    propagates through known corner angles, direct adjacent lines determine a
    shared corner, two endpoint corners determine an edge length, and a corner
    plus edge vector and length propagates around the polygon.  It returns
    whether any new fact was inferred.
    """
    changed_any = False
    changed = True

    while changed:
        changed = False
        n_sides = knowledge.n_sides

        for edge_idx in range(n_sides):
            if (knowledge.direct_point_observed[edge_idx]
                    and knowledge.edge_vector_known[edge_idx]
                    and not knowledge.direct_line_observed[edge_idx]):
                knowledge.direct_line_observed[edge_idx] = True
                knowledge.add_derivation(
                    f"edge {edge_idx} line is directly observed from a point and vector"
                )
                changed = True

        for corner_idx in range(n_sides):
            previous_edge = (corner_idx - 1) % n_sides
            current_edge = corner_idx
            if knowledge.corner_angle_known[corner_idx]:
                if (knowledge.edge_vector_known[previous_edge]
                        and not knowledge.edge_vector_known[current_edge]):
                    knowledge.edge_vector_known[current_edge] = True
                    knowledge.add_derivation(
                        f"edge {current_edge} vector follows from edge {previous_edge} and corner {corner_idx}"
                    )
                    changed = True
                elif (knowledge.edge_vector_known[current_edge]
                        and not knowledge.edge_vector_known[previous_edge]):
                    knowledge.edge_vector_known[previous_edge] = True
                    knowledge.add_derivation(
                        f"edge {previous_edge} vector follows from edge {current_edge} and corner {corner_idx}"
                    )
                    changed = True

            if (knowledge.edge_vector_known[previous_edge]
                    and knowledge.edge_vector_known[current_edge]
                    and not knowledge.corner_angle_known[corner_idx]):
                knowledge.corner_angle_known[corner_idx] = True
                knowledge.add_derivation(
                    f"corner {corner_idx} angle follows from adjacent edge vectors"
                )
                changed = True

        unknown_angles = [
            index for index, known in enumerate(knowledge.corner_angle_known)
            if not known
        ]
        if len(unknown_angles) == 1:
            index = unknown_angles[0]
            knowledge.corner_angle_known[index] = True
            knowledge.add_derivation(
                f"corner {index} angle follows from polygon angle-sum closure"
            )
            changed = True

        for corner_idx in range(n_sides):
            previous_edge = (corner_idx - 1) % n_sides
            current_edge = corner_idx
            if (knowledge.direct_line_observed[previous_edge]
                    and knowledge.direct_line_observed[current_edge]
                    and knowledge.adjacent_nonparallel[corner_idx]
                    and not knowledge.corner_known[corner_idx]):
                knowledge.corner_known[corner_idx] = True
                knowledge.add_derivation(
                    f"corner {corner_idx} follows from intersection of adjacent observed lines"
                )
                changed = True

        for edge_idx in range(n_sides):
            next_corner = (edge_idx + 1) % n_sides
            if (knowledge.corner_known[edge_idx]
                    and knowledge.corner_known[next_corner]
                    and not knowledge.length_known[edge_idx]):
                knowledge.length_known[edge_idx] = True
                knowledge.add_derivation(
                    f"edge {edge_idx} length follows from its two endpoint corners"
                )
                changed = True

        for edge_idx in range(n_sides):
            next_corner = (edge_idx + 1) % n_sides
            previous_corner = (edge_idx - 1) % n_sides
            previous_edge = previous_corner
            if (knowledge.corner_known[edge_idx]
                    and knowledge.edge_vector_known[edge_idx]
                    and knowledge.length_known[edge_idx]
                    and not knowledge.corner_known[next_corner]):
                knowledge.corner_known[next_corner] = True
                knowledge.add_derivation(
                    f"corner {next_corner} follows from corner {edge_idx}, edge vector, and length"
                )
                changed = True
            if (knowledge.corner_known[edge_idx]
                    and knowledge.edge_vector_known[previous_edge]
                    and knowledge.length_known[previous_edge]
                    and not knowledge.corner_known[previous_corner]):
                knowledge.corner_known[previous_corner] = True
                knowledge.add_derivation(
                    f"corner {previous_corner} follows from corner {edge_idx}, previous edge vector, and length"
                )
                changed = True

        for edge_idx in range(n_sides):
            previous_edge = (edge_idx - 1) % n_sides
            next_edge = (edge_idx + 1) % n_sides
            next_corner = (edge_idx + 1) % n_sides
            if (knowledge.length_known[edge_idx]
                    and knowledge.edge_vector_known[edge_idx]
                    and not knowledge.direct_line_observed[edge_idx]
                    and knowledge.direct_line_observed[previous_edge]
                    and knowledge.direct_line_observed[next_edge]
                    and knowledge.adjacent_nonparallel[edge_idx]
                    and knowledge.adjacent_nonparallel[next_corner]):
                if not knowledge.corner_known[edge_idx]:
                    knowledge.corner_known[edge_idx] = True
                    knowledge.add_derivation(
                        f"corner {edge_idx} is bracketed by adjacent observed lines"
                    )
                    changed = True
                if not knowledge.corner_known[next_corner]:
                    knowledge.corner_known[next_corner] = True
                    knowledge.add_derivation(
                        f"corner {next_corner} is bracketed by adjacent observed lines"
                    )
                    changed = True

        changed_any = changed_any or changed

    return changed_any


def symbolic_dof(knowledge: SymbolicKnowledge) -> int:
    """Return the current DOF estimate using the numerical model's convention."""
    n_sides = knowledge.n_sides
    dof = 3 * n_sides
    if any(knowledge.edge_vector_known):
        dof -= 1
    if any(knowledge.corner_known):
        dof -= 2
    dof -= min(n_sides - 2, sum(knowledge.corner_angle_known))
    dof -= min(n_sides - 1, sum(knowledge.length_known))
    dof -= sum(
        len(domain) == 1 for domain in knowledge.dihedral_domains
    )
    return dof


@dataclass(frozen=True)
class SymbolicEffect:
    """One possible value-free result of executing an action."""

    description: str
    probability: float
    edge_vectors_known: tuple[int, ...] = ()
    direct_points_observed: tuple[int, ...] = ()
    dihedral_assignments: tuple[tuple[int, float], ...] = ()
    reflexivity_assignments: tuple[tuple[int, bool], ...] = ()


@dataclass(frozen=True)
class SymbolicActionRequest:
    """A feasible action candidate supplied to the symbolic scorer."""

    action_type: ActionType
    reference_edge: int
    cost: float = 1.0
    label: str = ""


@dataclass(frozen=True)
class SymbolicOutcomeEvaluation:
    """Resulting state and DOF for one conditional action outcome."""

    effect: SymbolicEffect
    dof_after: int
    state: SymbolicKnowledge


@dataclass(frozen=True)
class SymbolicActionEvaluation:
    """Information score for an action across its symbolic outcomes."""

    request: SymbolicActionRequest
    target_edge: int
    dof_before: int
    expected_dof_after: float
    worst_case_dof_after: int
    expected_gain: float
    guaranteed_gain: int
    outcomes: tuple[SymbolicOutcomeEvaluation, ...]


def action_target_edge(action_type: ActionType,
                       reference_edge: int,
                       n_sides: int) -> int:
    """Return the edge directly measured or approached by an action."""
    next_edge = (reference_edge + 1) % n_sides
    previous_edge = (reference_edge - 1) % n_sides

    if action_type in (
        ActionType.SLIDE_OVER_SURFACE_PARALLEL_TO_EDGE_CCK,
        ActionType.SLIDE_OVER_SURFACE_PARALLEL_TO_EDGE_WITHIN_RANGE_CCK,
    ):
        return next_edge
    if action_type in (
        ActionType.SLIDE_OVER_SURFACE_PARALLEL_TO_EDGE_CK,
        ActionType.SLIDE_OVER_SURFACE_PARALLEL_TO_EDGE_WITHIN_RANGE_CK,
    ):
        return previous_edge
    if action_type == ActionType.SLIDE_OVER_SURFACE_PARALLEL_FROM_OUTSIDE_TO_EDGE_CCK:
        return previous_edge
    if action_type == ActionType.SLIDE_OVER_SURFACE_PARALLEL_FROM_OUTSIDE_TO_EDGE_CK:
        return next_edge
    if action_type == ActionType.MOVE_PARALLEL_FROM_OUTSIDE_TO_EDGE_UNTIL_CONTACT_CCK:
        return previous_edge
    if action_type == ActionType.MOVE_PARALLEL_FROM_OUTSIDE_TO_EDGE_UNTIL_CONTACT_CK:
        return next_edge
    return reference_edge


def _domain_effects(domain: frozenset[float],
                    edge_idx: int,
                    description: str,
                    add_point_when_90: bool = True) -> tuple[SymbolicEffect, ...]:
    """Create equally likely symbolic dihedral outcomes for an action target."""
    probability = 1.0 / len(domain)
    effects = []
    for dihedral in sorted(domain):
        point_edges = (edge_idx,) if add_point_when_90 and dihedral == 90.0 else ()
        effects.append(SymbolicEffect(
            description=f"{description}: dihedral {dihedral:g}",
            probability=probability,
            direct_points_observed=point_edges,
            dihedral_assignments=((edge_idx, dihedral),),
        ))
    return tuple(effects)


def _corner_reflexivity_effects(state: SymbolicKnowledge,
                                reference_edge: int,
                                direction: Direction,
                                vertical_mode: bool) -> tuple[SymbolicEffect, ...]:
    """Create outcome branches for an until-corner action.

    Edge-following actions resolve only reflexivity.  Sliding against a vertical
    surface also resolves an adjacent dihedral when the reached corner is
    non-reflexive.
    """
    n_sides = state.n_sides
    next_edge = (reference_edge + 1) % n_sides
    previous_edge = (reference_edge - 1) % n_sides
    if direction == Direction.CCK:
        reflexivity_corner = next_edge
        dihedral_edge = next_edge
    else:
        reflexivity_corner = reference_edge
        dihedral_edge = previous_edge

    reflexivity_domain = state.reflexivity_domains[reflexivity_corner]
    effects: list[SymbolicEffect] = []
    reflex_probability = 1.0 / len(reflexivity_domain)
    for reflexive in sorted(reflexivity_domain):
        if not vertical_mode or reflexive:
            effects.append(SymbolicEffect(
                description=(
                    f"corner {reflexivity_corner} is "
                    f"{'reflexive' if reflexive else 'non-reflexive'}"
                ),
                probability=reflex_probability,
                edge_vectors_known=(reference_edge,),
                reflexivity_assignments=((reflexivity_corner, reflexive),),
            ))
            continue

        dihedral_domain = state.dihedral_domains[dihedral_edge]
        dihedral_probability = 1.0 / len(dihedral_domain)
        for dihedral in sorted(dihedral_domain):
            effects.append(SymbolicEffect(
                description=(
                    f"corner {reflexivity_corner} is non-reflexive; "
                    f"edge {dihedral_edge} dihedral is {dihedral:g}"
                ),
                probability=reflex_probability * dihedral_probability,
                edge_vectors_known=(reference_edge,),
                direct_points_observed=(dihedral_edge,) if dihedral == 90.0 else (),
                dihedral_assignments=((dihedral_edge, dihedral),),
                reflexivity_assignments=((reflexivity_corner, False),),
            ))
    return tuple(effects)


def symbolic_outcomes_for_action(state: SymbolicKnowledge,
                                 request: SymbolicActionRequest) -> tuple[SymbolicEffect, ...]:
    """Return conditional, value-free observations expected from one action."""
    action_type = request.action_type
    reference_edge = request.reference_edge
    target_edge = action_target_edge(action_type, reference_edge, state.n_sides)
    action_spec = ACTION_TO_SPEC[action_type]

    if action_type in (
        ActionType.MOVE_PARALLEL_FROM_OUTSIDE_TO_EDGE_UNTIL_CONTACT_CCK,
        ActionType.MOVE_PARALLEL_FROM_OUTSIDE_TO_EDGE_UNTIL_CONTACT_CK,
    ):
        return (SymbolicEffect(
            description=f"direct point observed on edge {target_edge}",
            probability=1.0,
            direct_points_observed=(target_edge,),
        ),)

    if action_type in (
        ActionType.SLIDE_AGAINST_EDGE_CCK,
        ActionType.SLIDE_AGAINST_EDGE_CK,
        ActionType.SLIDE_AGAINST_VERTICAL_SURFACE_CCK,
        ActionType.SLIDE_AGAINST_VERTICAL_SURFACE_CK,
    ):
        return (SymbolicEffect(
            description=f"edge vector observed on edge {reference_edge}",
            probability=1.0,
            edge_vectors_known=(reference_edge,),
        ),)

    if action_type in (
        ActionType.SLIDE_AGAINST_EDGE_UNTIL_CORNER_CCK,
        ActionType.SLIDE_AGAINST_EDGE_UNTIL_CORNER_CK,
    ):
        return _corner_reflexivity_effects(
            state,
            reference_edge,
            action_spec.direction,
            vertical_mode=False,
        )

    if action_type in (
        ActionType.SLIDE_AGAINST_VERTICAL_SURFACE_UNTIL_CORNER_CCK,
        ActionType.SLIDE_AGAINST_VERTICAL_SURFACE_UNTIL_CORNER_CK,
    ):
        return _corner_reflexivity_effects(
            state,
            reference_edge,
            action_spec.direction,
            vertical_mode=True,
        )

    if action_type in (
        ActionType.SLIDE_OVER_SURFACE_PARALLEL_TO_EDGE_WITHIN_RANGE_CCK,
        ActionType.SLIDE_OVER_SURFACE_PARALLEL_TO_EDGE_WITHIN_RANGE_CK,
    ):
        if action_spec.direction == Direction.CCK:
            reflexivity_corner = target_edge
        else:
            reflexivity_corner = reference_edge
        reflexivity_domain = state.reflexivity_domains[reflexivity_corner]
        effects: list[SymbolicEffect] = []
        reflex_probability = 1.0 / len(reflexivity_domain)
        for reflexive in sorted(reflexivity_domain):
            if reflexive:
                effects.append(SymbolicEffect(
                    description=f"no target edge: corner {reflexivity_corner} is reflexive",
                    probability=reflex_probability,
                    reflexivity_assignments=((reflexivity_corner, True),),
                ))
                continue
            for effect in _domain_effects(
                    state.dihedral_domains[target_edge],
                    target_edge,
                    f"corner {reflexivity_corner} is non-reflexive",
            ):
                effects.append(SymbolicEffect(
                    description=effect.description,
                    probability=reflex_probability * effect.probability,
                    direct_points_observed=effect.direct_points_observed,
                    dihedral_assignments=effect.dihedral_assignments,
                    reflexivity_assignments=((reflexivity_corner, False),),
                ))
        return tuple(effects)

    if action_type in (
        ActionType.SLIDE_OVER_SURFACE_UNTIL_EDGE,
        ActionType.SLIDE_OVER_SURFACE_PERPENDICULAR_TO_EDGE_GIVEN_ONE_POINT,
        ActionType.SLIDE_OVER_SURFACE_PARALLEL_TO_EDGE_CCK,
        ActionType.SLIDE_OVER_SURFACE_PARALLEL_TO_EDGE_CK,
        ActionType.SLIDE_OVER_SURFACE_PARALLEL_FROM_OUTSIDE_TO_EDGE_CCK,
        ActionType.SLIDE_OVER_SURFACE_PARALLEL_FROM_OUTSIDE_TO_EDGE_CK,
    ):
        return _domain_effects(
            state.dihedral_domains[target_edge],
            target_edge,
            "surface-contact outcome",
        )

    raise NotImplementedError(f"No symbolic action model for {action_type.name}")


def apply_symbolic_effect(state: SymbolicKnowledge,
                          effect: SymbolicEffect) -> SymbolicKnowledge:
    """Apply one action outcome to a cloned state and propagate its implications."""
    result = state.clone()
    for edge_idx in effect.edge_vectors_known:
        result.edge_vector_known[edge_idx] = True
    for edge_idx in effect.direct_points_observed:
        result.direct_point_observed[edge_idx] = True
    for edge_idx, dihedral in effect.dihedral_assignments:
        result.dihedral_domains[edge_idx] = frozenset((dihedral,))
    for corner_idx, reflexivity in effect.reflexivity_assignments:
        result.reflexivity_domains[corner_idx] = frozenset((reflexivity,))
    result.add_derivation(effect.description)
    propagate_symbolic(result)
    return result


def predict_edge_exploration(state: SymbolicKnowledge,
                             edge_idx: int) -> SymbolicKnowledge:
    """Predict determinacy after the current exploration branch measures an edge.

    This is the symbolic counterpart of the numerical simulator's temporary
    two-point edge measurement.  It adds only the relations guaranteed by a
    completed edge-exploration branch: a vector and one direct point on the
    target edge.  It deliberately does not assign a dihedral, reflexivity, or
    coordinate value.
    """
    if not 0 <= edge_idx < state.n_sides:
        raise IndexError(f"Edge index {edge_idx} is outside the polygon")
    return apply_symbolic_effect(
        state,
        SymbolicEffect(
            description=f"completed exploration branch directly observes edge {edge_idx}",
            probability=1.0,
            edge_vectors_known=(edge_idx,),
            direct_points_observed=(edge_idx,),
        ),
    )


def evaluate_symbolic_action(state: SymbolicKnowledge,
                             request: SymbolicActionRequest) -> SymbolicActionEvaluation:
    """Evaluate expected and guaranteed information gain for one action."""
    before_state = state.clone()
    propagate_symbolic(before_state)
    dof_before = symbolic_dof(before_state)
    target_edge = action_target_edge(
        request.action_type,
        request.reference_edge,
        before_state.n_sides,
    )
    outcomes = []
    for effect in symbolic_outcomes_for_action(before_state, request):
        outcome_state = apply_symbolic_effect(before_state, effect)
        outcomes.append(SymbolicOutcomeEvaluation(
            effect=effect,
            dof_after=symbolic_dof(outcome_state),
            state=outcome_state,
        ))

    probability_sum = sum(outcome.effect.probability for outcome in outcomes)
    if not math.isclose(probability_sum, 1.0, abs_tol=1e-9):
        raise RuntimeError(
            f"Action outcome probabilities for {request.action_type.name} sum to {probability_sum}"
        )

    expected_dof_after = sum(
        outcome.effect.probability * outcome.dof_after
        for outcome in outcomes
    )
    worst_case_dof_after = max(outcome.dof_after for outcome in outcomes)
    return SymbolicActionEvaluation(
        request=request,
        target_edge=target_edge,
        dof_before=dof_before,
        expected_dof_after=expected_dof_after,
        worst_case_dof_after=worst_case_dof_after,
        expected_gain=dof_before - expected_dof_after,
        guaranteed_gain=dof_before - worst_case_dof_after,
        outcomes=tuple(outcomes),
    )


def rank_symbolic_actions(state: SymbolicKnowledge,
                          requests: Iterable[SymbolicActionRequest]) -> list[SymbolicActionEvaluation]:
    """Rank feasible actions by guaranteed gain, expected gain, then execution cost."""
    evaluations = [evaluate_symbolic_action(state, request) for request in requests]
    return sorted(
        evaluations,
        key=lambda evaluation: (
            evaluation.guaranteed_gain,
            evaluation.expected_gain,
            -evaluation.request.cost,
            -evaluation.target_edge,
            -evaluation.request.reference_edge,
            -evaluation.request.action_type.value,
        ),
        reverse=True,
    )


def generate_symbolic_candidates(state: SymbolicKnowledge,
                                 previous_action: Optional[ActionInstance] = None) -> list[SymbolicActionRequest]:
    """Generate conservative action candidates with known symbolic effects.

    This intentionally covers the common edge-contact, edge-vector, and
    pose-anchor cases.  Robot-specific feasibility checks remain in the
    reasoner; callers may instead pass its filtered candidates directly to
    :func:`rank_symbolic_actions`.
    """
    requests: list[SymbolicActionRequest] = []
    n_sides = state.n_sides

    if not any(state.edge_vector_known):
        return [SymbolicActionRequest(
            ActionType.SLIDE_OVER_SURFACE_UNTIL_EDGE,
            0,
            label="default first contact",
        )]

    for edge_idx in range(n_sides):
        if state.direct_line_observed[edge_idx]:
            if not state.dihedral_is_known(edge_idx):
                requests.append(SymbolicActionRequest(
                    ActionType.SLIDE_OVER_SURFACE_PERPENDICULAR_TO_EDGE_GIVEN_ONE_POINT,
                    edge_idx,
                    label="resolve dihedral on observed edge",
                ))
            previous_edge = (edge_idx - 1) % n_sides
            next_edge = (edge_idx + 1) % n_sides
            if not state.direct_point_observed[next_edge]:
                requests.append(SymbolicActionRequest(
                    ActionType.SLIDE_OVER_SURFACE_PARALLEL_TO_EDGE_CCK,
                    edge_idx,
                    label="seek direct point on next edge",
                ))
            if not state.direct_point_observed[previous_edge]:
                requests.append(SymbolicActionRequest(
                    ActionType.SLIDE_OVER_SURFACE_PARALLEL_TO_EDGE_CK,
                    edge_idx,
                    label="seek direct point on previous edge",
                ))

        if state.direct_point_observed[edge_idx] and not state.edge_vector_known[edge_idx]:
            if state.dihedral_is_known(edge_idx):
                dihedral = next(iter(state.dihedral_domains[edge_idx]))
                action_type = (
                    ActionType.SLIDE_AGAINST_VERTICAL_SURFACE_UNTIL_CORNER_CCK
                    if dihedral == 90.0
                    else ActionType.SLIDE_AGAINST_EDGE_UNTIL_CORNER_CCK
                )
                requests.append(SymbolicActionRequest(
                    action_type,
                    edge_idx,
                    label="estimate edge vector from direct point",
                ))

    if previous_action is not None:
        previous_spec = action_spec_from_action(previous_action.action_type)
        if previous_spec is not None and previous_spec.stop in (
                Stop.UNTIL_CORNER,
                Stop.UNTIL_EDGE_CONTACT,
        ):
            reference_edge = previous_action.edge_index
            if reference_edge is not None and state.edge_vector_known[reference_edge]:
                requests.extend((
                    SymbolicActionRequest(
                        ActionType.MOVE_PARALLEL_FROM_OUTSIDE_TO_EDGE_UNTIL_CONTACT_CCK,
                        reference_edge,
                        cost=1.5,
                        label="obtain previous-edge contact point",
                    ),
                    SymbolicActionRequest(
                        ActionType.MOVE_PARALLEL_FROM_OUTSIDE_TO_EDGE_UNTIL_CONTACT_CK,
                        reference_edge,
                        cost=1.5,
                        label="obtain next-edge contact point",
                    ),
                ))

    deduplicated: dict[tuple[ActionType, int], SymbolicActionRequest] = {}
    for request in requests:
        deduplicated[(request.action_type, request.reference_edge)] = request
    return list(deduplicated.values())


def next_symbolic_action(knowledge: PolygonKnowledge,
                         previous_action: Optional[ActionInstance] = None,
                         candidates: Optional[Sequence[SymbolicActionRequest]] = None) -> tuple[Optional[ActionType], Optional[int]]:
    """Select the highest-scoring symbolic candidate with the legacy return shape.

    ``candidates`` should be supplied by a robot-specific feasibility layer
    when available.  Otherwise the module uses its conservative built-in
    candidate generator.
    """
    state = symbolic_from_polygon_knowledge(knowledge)
    propagate_symbolic(state)
    requests = list(candidates) if candidates is not None else generate_symbolic_candidates(
        state,
        previous_action,
    )
    if not requests:
        return None, None
    ranked = rank_symbolic_actions(state, requests)
    selected = ranked[0].request
    return selected.action_type, selected.reference_edge


@dataclass(frozen=True)
class PropagationComparison:
    """Mask-level parity report between numeric and symbolic propagation."""

    numerical_dof: int
    symbolic_dof: int
    field_mismatches: dict[str, tuple[int, ...]]

    @property
    def equivalent(self) -> bool:
        """Return whether compared fields and DOF agree across both propagators."""
        return self.numerical_dof == self.symbolic_dof and not self.field_mismatches


def compare_with_numerical_propagation(knowledge: PolygonKnowledge) -> PropagationComparison:
    """Compare symbolic determinacy against the current numeric propagator.

    The comparison intentionally checks only fields represented by the
    value-free state: vectors, lengths, corners, corner angles, and dihedrals.
    Reflexivity values can require an actual angle value, so they remain an
    action-outcome domain instead of a strict parity field.
    """
    from .algorithm import propagate_parameters

    numerical = copy.deepcopy(knowledge)
    propagate_parameters(numerical)
    symbolic = symbolic_from_polygon_knowledge(knowledge)
    propagate_symbolic(symbolic)

    comparisons = {
        "edge_vector": (
            [value is not None for value in numerical.edge_unit_vectors],
            symbolic.edge_vector_known,
        ),
        "length": (
            [value is not None for value in numerical.lengths],
            symbolic.length_known,
        ),
        "corner": (
            [value is not None for value in numerical.corners],
            symbolic.corner_known,
        ),
        "corner_angle": (
            [value is not None for value in numerical.corner_angles],
            symbolic.corner_angle_known,
        ),
        "dihedral": (
            [value is not None for value in numerical.dihedrals],
            [len(domain) == 1 for domain in symbolic.dihedral_domains],
        ),
    }
    field_mismatches = {
        field: tuple(
            index for index, (numerical_known, symbolic_known) in enumerate(zip(numerical_mask, symbolic_mask))
            if numerical_known != symbolic_known
        )
        for field, (numerical_mask, symbolic_mask) in comparisons.items()
        if any(numerical_known != symbolic_known for numerical_known, symbolic_known in zip(numerical_mask, symbolic_mask))
    }
    return PropagationComparison(
        numerical_dof=find_dof(numerical),
        symbolic_dof=symbolic_dof(symbolic),
        field_mismatches=field_mismatches,
    )
