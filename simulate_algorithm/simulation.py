"""
Simulator for polygon exploration algorithm

Provides standalone simulation environment to test the motion reasoning algorithm
without requiring robot hardware or ROS. Uses core_algorithm for decision making.

Functions:
- simulate_robot(): Main simulation runner with interactive step-by-step exploration
- generate_polygon(): Create test polygons with specified properties
- generate_prior_knowledge_from_gt(): Create robot prior knowledge from ground truth
"""

import math
import random
import numpy as np
from dataclasses import replace
from scipy.optimize import linprog

# Import from core_algorithm module
from core_algorithm import (
    PolygonKnowledge,
    ActionType,
    ActionInstance,
    next_action,
    propagate_parameters,
    find_unique_pattern,
    get_unique_pattern_ref_index,
    find_dof,
    rearrange_rck_using_prior_knowledge,
    fill_missing_parameters,
    get_random_points_on_line,
    get_unit_vector,
    point_in_list,
    are_list_elements_uniform,
    generate_internal_angles
)


def feasible_bounded_lengths(edge_unit_vectors: list,
                             seed: int = 47,
                             lower_bound: float = 0.5,
                             upper_bound: float = 3.0) -> list:
    """
    Find feasible edge lengths within given bounds that satisfy closure condition

    :param edge_unit_vectors: List of edge unit vectors
    :param seed: Random seed for reproducibility
    :param lower_bound: Lower bound for edge lengths
    :param upper_bound: Upper bound for edge lengths
    :return: List of feasible edge lengths
    """

    U = np.asarray(edge_unit_vectors, float)
    n = U.shape[0]

    rng = np.random.default_rng(seed)
    c = rng.standard_normal(n)

    res = linprog(
        c,
        A_eq=U.T,
        b_eq=np.zeros(2),
        bounds=[(lower_bound, upper_bound)] * n,
        method="highs",
    )
    if not res.success:
        raise ValueError("No feasible lengths within bounds.")
    return res.x.tolist()


def generate_prior_knowledge_from_gt(gt: PolygonKnowledge,
                                     degree_of_prior_knowledge: int,
                                     percentage_of_edge_filled: float = 0.6,
                                     random_seed: int = 42) -> PolygonKnowledge:
    """
    Generate robot prior knowledge (rpk) from ground truth based on degree of prior knowledge
    
    :param gt: Ground truth polygon knowledge
    :param degree_of_prior_knowledge: Level of prior knowledge
        - 0: empty/low
        - 1: only reflexivity of angles
        - 2: only corner angles
        - 3: random features (controlled by percentage_of_edge_filled)
        - 4: full knowledge
    :param percentage_of_edge_filled: Percentage of edges with features (between 0 and 1)
    :param random_seed: Random seed for reproducibility
    :return: Robot prior knowledge (rpk)
    """
    if not 0.0 <= percentage_of_edge_filled <= 1.0:
        raise ValueError("percentage_of_edge_filled must be between 0 and 1")

    rpk = PolygonKnowledge(gt.n_sides)
    if degree_of_prior_knowledge == 0:
        pass

    elif degree_of_prior_knowledge == 1:
        for i in range(gt.n_sides):
            rpk.is_reflexive_angle[i] = gt.is_reflexive_angle[i]

    elif degree_of_prior_knowledge == 2:
        for i in range(gt.n_sides):
            rpk.corner_angles[i] = gt.corner_angles[i]

    elif degree_of_prior_knowledge == 3:
        # randomly select which features to fill
        rng = np.random.default_rng(random_seed)
        num_features = 4  # is_reflexive_angle, corner_angles, dihedrals, lengths
        
        # number of sides to fill (rounded up)
        num_sides_to_fill = math.ceil(percentage_of_edge_filled * gt.n_sides)
        
        # choose which sides get a feature
        active_sides = rng.choice(gt.n_sides, size=num_sides_to_fill, replace=False)
        feature_indices = rng.integers(0, num_features, size=num_sides_to_fill)

        for side_idx, feature_idx in zip(active_sides, feature_indices):
            if feature_idx == 0:
                rpk.is_reflexive_angle[side_idx] = gt.is_reflexive_angle[side_idx]
            elif feature_idx == 1:
                rpk.corner_angles[side_idx] = gt.corner_angles[side_idx]
            elif feature_idx == 2:
                rpk.dihedrals[side_idx] = gt.dihedrals[side_idx]
            elif feature_idx == 3:
                rpk.lengths[side_idx] = gt.lengths[side_idx]
    elif degree_of_prior_knowledge == 4:
        for i in range(gt.n_sides):
            rpk.is_reflexive_angle[i] = gt.is_reflexive_angle[i]
            rpk.corner_angles[i] = gt.corner_angles[i]
            rpk.dihedrals[i] = gt.dihedrals[i]
            rpk.lengths[i] = gt.lengths[i]
    else:
        raise ValueError("degree_of_prior_knowledge must be 0, 1, 2, 3, or 4")
    return rpk


def generate_polygon(n_sides: int, angles_deg: list[float], random_seed: int = 47) -> PolygonKnowledge:
    """
    Generate random polygon with given number of edges and angles.
    
    :param n_sides: Number of edges of the polygon
    :param angles_deg: List of corner angles in degrees
    :param random_seed: Random seed for reproducibility
    :return: PolygonKnowledge object with knowledge of the polygon
    """
    assert n_sides >= 3, "a 2D polygon must have at least 3 edges"
    assert len(angles_deg) == n_sides, "angles list length must equal n_sides"

    polygon_knowledge_gt = PolygonKnowledge(n_sides)
    polygon_knowledge_gt.corner_angles = angles_deg

    # Assign reflexivity based on corner angles
    for i in range(n_sides):
        if angles_deg[i] < 180.0:
            polygon_knowledge_gt.is_reflexive_angle[i] = False
        elif angles_deg[i] > 180.0:
            polygon_knowledge_gt.is_reflexive_angle[i] = True
    
    # Set first edge along x-axis as reference
    polygon_knowledge_gt.edge_unit_vectors[0] = (1.0, 0.0)
    polygon_knowledge_gt.corners[0] = (0.0, 0.0)

    # Randomly assign dihedrals as 90 or 270 degrees
    random.seed(random_seed)
    for i in range(n_sides):
        polygon_knowledge_gt.dihedrals[i] = random.choice([90.0, 270.0])
    propagate_parameters(polygon_knowledge_gt)

    # Sample edge lengths satisfying closure condition with positive lengths
    polygon_knowledge_gt.lengths = feasible_bounded_lengths(
        polygon_knowledge_gt.edge_unit_vectors,
        lower_bound=0.5,
        upper_bound=3.0,
        seed=random_seed
    )
    propagate_parameters(polygon_knowledge_gt)

    return polygon_knowledge_gt


def simulate_robot(to_plot: bool = False, 
                   degree_of_prior_knowledge: int = 0,
                   polygon_knowledge_gt: PolygonKnowledge | None = None,
                   shift_in_idx_for_rck: int = 2,
                   percentage_of_edge_filled: float = 0.6,
                   random_seed: int = 42) -> None:
    """
    Run interactive simulation of robot reconstructing a polygon.
    
    :param to_plot: Whether to plot polygon at various stages
    :param degree_of_prior_knowledge: Level of prior knowledge (0-4)
    :param polygon_knowledge_gt: Ground truth polygon; if None, default pentagon created
    :param shift_in_idx_for_rck: Index shift for rck initialization
    :param percentage_of_edge_filled: Percentage of edges with initial features (0-1)
    :param random_seed: Random seed for reproducibility
    :return: None
    """

    # Initialize flags
    unique_pattern_found_in_rpk = False
    unique_pattern_found_in_rck = False
    corner_coordinates_available_in_rpk = False
    rpk_rck_matching_idx_found = False
    rck_rearranged = False
    prev_action_instance = ActionInstance(action_type=None, edge_index=None)

    # Setup ground truth
    if polygon_knowledge_gt is not None:
        gt = polygon_knowledge_gt
        number_of_sides = gt.n_sides
    else:
        # Default: pentagon
        number_of_sides = 5
        gt = PolygonKnowledge(number_of_sides)
        gt.corners = [
            (1.0, 1.0),
            (5.3879, 3.3971),
            (4.4640, 8.3110),
            (-6.8520, 9.7711),
            (-7.2966, 8.8326),
        ]
        for i in range(number_of_sides):
            curr_edge_idx = i
            next_edge_idx = (i + 1) % number_of_sides
            dx = gt.corners[next_edge_idx][0] - gt.corners[curr_edge_idx][0]
            dy = gt.corners[next_edge_idx][1] - gt.corners[curr_edge_idx][1]
            if dx != 0:
                gt.slopes[curr_edge_idx] = dy / dx
            else:
                gt.slopes[curr_edge_idx] = float('inf')
            gt.corner_angles[curr_edge_idx] = 108.0  # degrees
            if gt.corner_angles[curr_edge_idx] < 180.0:
                gt.is_reflexive_angle[curr_edge_idx] = False
            elif gt.corner_angles[curr_edge_idx] > 180.0:
                gt.is_reflexive_angle[curr_edge_idx] = True
            gt.lengths[curr_edge_idx] = np.hypot(dx, dy)
            gt.dihedrals[curr_edge_idx] = 270.0      # degrees

        propagate_parameters(gt)
        gt.print_knowledge("Ground Truth")

    if to_plot:
        gt.plot_polygon("Ground Truth Polygon")

    # Generate robot prior knowledge
    rpk = generate_prior_knowledge_from_gt(gt, 
                                           degree_of_prior_knowledge=degree_of_prior_knowledge, 
                                           percentage_of_edge_filled=percentage_of_edge_filled, 
                                           random_seed=random_seed)
    rpk.print_knowledge("Robot Prior Knowledge (rpk) [Before propagation]")

    propagate_parameters(rpk)
    unique_pattern_found_in_rpk = find_unique_pattern(rpk)
    if any(c is not None for c in rpk.corners):
        corner_coordinates_available_in_rpk = True
    rpk.print_knowledge("Robot Prior Knowledge (rpk) [After propagation]")
    print("unique_pattern_found_in_rpk: ", unique_pattern_found_in_rpk, 
          "; corner_coordinates_available_in_rpk: ", corner_coordinates_available_in_rpk)

    # Initialize robot current knowledge
    rck = PolygonKnowledge(number_of_sides)
    fill_missing_parameters(rck, rpk, rpk_rck_matching_idx_found)
    
    print("Starting reconstruction...")
    steps = 0

    propagate_parameters(rck)
    rck.print_knowledge("Robot Current Knowledge (rck)")
    dof = find_dof(rck)
    print(f"Initial DOF = {dof}")

    # Main simulation loop
    while find_dof(rck) > 0 and steps < 15:
        print("\n ***************  Step number: ", steps, "  ***************")
        act, edge_idx = next_action(rck, prev_action_instance, rck_rearranged, in_simulation=True, gt=gt)
        if act: print("Action to perform: ", act.name, " reference edge: ", edge_idx)
        
        # Interactive user prompt
        attempt = 0
        num_attempts_allowed = 2
        stop_execution = False
        while attempt < num_attempts_allowed:
            answer = input("Do you want to continue? (yes/no): ").strip().lower()
            if answer in ("yes", "y"):
                print("Continuing...")
                break
            elif answer in ("no", "n"):
                print("Stopping.")
                stop_execution = True
                break
            else:
                attempt += 1
                if attempt < num_attempts_allowed:
                    print(f"Invalid input. Please enter 'yes' or 'y' or 'no' or 'n'. Attempts left: {num_attempts_allowed - attempt}")
                else:
                    print("Invalid input. Aborting.")
                    return
        if stop_execution:
            return

        prev_action_instance = replace(prev_action_instance,
                                       action_type=act,
                                       edge_index=edge_idx)

        if act is None or edge_idx is None:
            rck.print_knowledge("Robot Current Knowledge (rck)")
            if to_plot:
                rck.plot_polygon("Final Reconstructed Polygon")
            print("No further action possible based on current knowledge.")
            break
        
        print("Performing action in simulation...")

        # Simulate sensor feedback based on ground truth
        next_edge_idx = (edge_idx+1) % rck.n_sides
        prev_edge_idx = (edge_idx-1) % rck.n_sides
        edge_idx_in_gt = (shift_in_idx_for_rck + edge_idx) % rck.n_sides
        next_edge_idx_in_gt = (shift_in_idx_for_rck + edge_idx + 1) % rck.n_sides
        second_next_edge_idx_in_gt = (shift_in_idx_for_rck + edge_idx + 2) % rck.n_sides
        prev_edge_idx_in_gt = (shift_in_idx_for_rck + edge_idx - 1) % rck.n_sides
        prev_to_prev_edge_idx_in_gt = (shift_in_idx_for_rck + edge_idx - 2) % rck.n_sides
        print("edge_idx_in_gt: ", edge_idx_in_gt, " edge_idx_in_ck: ", edge_idx)

        if rck.dihedrals[edge_idx] is None:
            rck.dihedrals[edge_idx] = gt.dihedrals[edge_idx_in_gt]
        
        # Handle different action types
        if act == ActionType.SLIDE_OVER_SURFACE_UNTIL_EDGE:
            rck.dihedrals[edge_idx] = gt.dihedrals[edge_idx_in_gt]
            if rck.dihedrals[edge_idx] == 90.0:
                rck.internal_points_on_edge[edge_idx].extend(get_random_points_on_line(
                    gt.corners[edge_idx_in_gt], gt.corners[next_edge_idx_in_gt], num_points=1))
        elif act == ActionType.SLIDE_AGAINST_EDGE_CCK:
            rck.internal_points_on_edge[edge_idx].extend(get_random_points_on_line(
                gt.corners[edge_idx_in_gt], gt.corners[next_edge_idx_in_gt], num_points=2))
            rck.edge_unit_vectors[edge_idx] = get_unit_vector(gt.corners[next_edge_idx_in_gt], gt.corners[edge_idx_in_gt])
        elif act == ActionType.SLIDE_AGAINST_EDGE_CK:
            rck.internal_points_on_edge[edge_idx].extend(get_random_points_on_line(
                gt.corners[prev_edge_idx_in_gt], gt.corners[edge_idx_in_gt], num_points=2))
            rck.edge_unit_vectors[edge_idx] = get_unit_vector(gt.corners[next_edge_idx_in_gt], gt.corners[edge_idx_in_gt])
        elif act in (ActionType.SLIDE_OVER_SURFACE_PARALLEL_TO_EDGE_CCK,
                     ActionType.SLIDE_OVER_SURFACE_PARALLEL_TO_EDGE_CK):
            is_cck = act == ActionType.SLIDE_OVER_SURFACE_PARALLEL_TO_EDGE_CCK

            if is_cck:
                first_edge_gt, second_edge_gt = next_edge_idx_in_gt, second_next_edge_idx_in_gt
                edge_in_rck = next_edge_idx
                dihedral_source = first_edge_gt
            else:
                first_edge_gt, second_edge_gt = prev_edge_idx_in_gt, edge_idx_in_gt
                edge_in_rck = prev_edge_idx
                dihedral_source = second_edge_gt
            dihedral_angle = gt.dihedrals[dihedral_source]
            rck.dihedrals[edge_in_rck] = gt.dihedrals[dihedral_source]
            
            if dihedral_angle == 90.0:
                edge_points = rck.internal_points_on_edge[edge_in_rck]
                for _ in range(100):
                    random_point = get_random_points_on_line(gt.corners[first_edge_gt],
                                                            gt.corners[second_edge_gt],
                                                            num_points=1)[0]
                    if random_point not in edge_points:
                        edge_points.append(random_point)
                        break

        elif act == ActionType.SLIDE_AGAINST_EDGE_UNTIL_CORNER_CCK:
            rck.internal_points_on_edge[edge_idx].extend(get_random_points_on_line(
                gt.corners[edge_idx_in_gt], gt.corners[next_edge_idx_in_gt], num_points=2))
            rck.is_reflexive_angle[next_edge_idx] = gt.is_reflexive_angle[next_edge_idx_in_gt]
            rck.edge_unit_vectors[edge_idx] = get_unit_vector(gt.corners[next_edge_idx_in_gt], gt.corners[edge_idx_in_gt])
        elif act == ActionType.SLIDE_AGAINST_VERTICAL_SURFACE_UNTIL_CORNER_CCK:
            rck.internal_points_on_edge[edge_idx].extend(get_random_points_on_line(
                gt.corners[edge_idx_in_gt], gt.corners[next_edge_idx_in_gt], num_points=2))
            rck.is_reflexive_angle[next_edge_idx] = gt.is_reflexive_angle[next_edge_idx_in_gt]
            rck.edge_unit_vectors[edge_idx] = get_unit_vector(gt.corners[next_edge_idx_in_gt], gt.corners[edge_idx_in_gt])
            if rck.is_reflexive_angle[next_edge_idx] == False:
                rck.dihedrals[next_edge_idx] = gt.dihedrals[next_edge_idx_in_gt]
                if rck.dihedrals[next_edge_idx] == 90.0:
                    rck.internal_points_on_edge[next_edge_idx].extend(get_random_points_on_line(
                        gt.corners[next_edge_idx_in_gt], gt.corners[second_next_edge_idx_in_gt], num_points=1))
        elif act == ActionType.SLIDE_AGAINST_EDGE_UNTIL_CORNER_CK:
            rck.internal_points_on_edge[edge_idx].extend(get_random_points_on_line(
                gt.corners[prev_edge_idx_in_gt], gt.corners[edge_idx_in_gt], num_points=2))
            rck.is_reflexive_angle[prev_edge_idx] = gt.is_reflexive_angle[prev_edge_idx_in_gt]
            rck.edge_unit_vectors[edge_idx] = get_unit_vector(gt.corners[prev_edge_idx_in_gt], gt.corners[edge_idx_in_gt])
        elif act == ActionType.SLIDE_AGAINST_VERTICAL_SURFACE_UNTIL_CORNER_CK:
            rck.internal_points_on_edge[edge_idx].extend(get_random_points_on_line(
                gt.corners[prev_edge_idx_in_gt], gt.corners[edge_idx_in_gt], num_points=2))
            rck.is_reflexive_angle[prev_edge_idx] = gt.is_reflexive_angle[prev_edge_idx_in_gt]
            rck.edge_unit_vectors[edge_idx] = get_unit_vector(gt.corners[prev_edge_idx_in_gt], gt.corners[edge_idx_in_gt])
            if rck.is_reflexive_angle[prev_edge_idx] == False:
                rck.dihedrals[prev_edge_idx] = gt.dihedrals[prev_edge_idx_in_gt]
                if rck.dihedrals[prev_edge_idx] == 90.0:
                    rck.internal_points_on_edge[prev_edge_idx].extend(get_random_points_on_line(
                        gt.corners[prev_edge_idx_in_gt], gt.corners[prev_to_prev_edge_idx_in_gt], num_points=1))
        elif act == ActionType.SLIDE_AGAINST_VERTICAL_SURFACE_CCK:
            rck.internal_points_on_edge[edge_idx].extend(get_random_points_on_line(
                gt.corners[edge_idx_in_gt], gt.corners[next_edge_idx_in_gt], num_points=2))
            rck.edge_unit_vectors[edge_idx] = get_unit_vector(gt.corners[next_edge_idx_in_gt], gt.corners[edge_idx_in_gt])
        elif act == ActionType.SLIDE_AGAINST_VERTICAL_SURFACE_CK:
            rck.internal_points_on_edge[edge_idx].extend(get_random_points_on_line(
                gt.corners[edge_idx_in_gt], gt.corners[prev_edge_idx_in_gt], num_points=2))
            rck.edge_unit_vectors[edge_idx] = get_unit_vector(gt.corners[prev_edge_idx_in_gt], gt.corners[edge_idx_in_gt])
        elif act == ActionType.MOVE_PARALLEL_FROM_OUTSIDE_TO_EDGE_UNTIL_CONTACT_CCK:
            rck.internal_points_on_edge[prev_edge_idx].extend(get_random_points_on_line(
                gt.corners[edge_idx_in_gt], gt.corners[prev_edge_idx_in_gt], num_points=1))
        elif act == ActionType.MOVE_PARALLEL_FROM_OUTSIDE_TO_EDGE_UNTIL_CONTACT_CK:
            rck.internal_points_on_edge[next_edge_idx].extend(get_random_points_on_line(
                gt.corners[next_edge_idx_in_gt], gt.corners[second_next_edge_idx_in_gt], num_points=1))
        elif act == ActionType.SLIDE_OVER_SURFACE_PARALLEL_FROM_OUTSIDE_TO_EDGE_CCK:
            rck.internal_points_on_edge[prev_edge_idx].extend(get_random_points_on_line(
                gt.corners[edge_idx_in_gt], gt.corners[prev_edge_idx_in_gt], num_points=1))
            rck.dihedrals[prev_edge_idx] = gt.dihedrals[prev_edge_idx_in_gt]
        elif act == ActionType.SLIDE_OVER_SURFACE_PARALLEL_FROM_OUTSIDE_TO_EDGE_CK:
            rck.internal_points_on_edge[next_edge_idx].extend(get_random_points_on_line(
                gt.corners[edge_idx_in_gt], gt.corners[next_edge_idx_in_gt], num_points=1))
            rck.dihedrals[next_edge_idx] = gt.dihedrals[next_edge_idx_in_gt]
        elif act == ActionType.SLIDE_OVER_SURFACE_PERPENDICULAR_TO_EDGE_GIVEN_ONE_POINT:
            rck.dihedrals[edge_idx] = gt.dihedrals[edge_idx_in_gt]

        # Propagate and check for matching patterns
        propagate_parameters(rck)
        unique_pattern_found_in_rck = find_unique_pattern(rck)

        if unique_pattern_found_in_rpk and unique_pattern_found_in_rck and not rpk_rck_matching_idx_found:
            print("Unique_pattern_found_in_rpk: ", unique_pattern_found_in_rpk
                  , ", unique_pattern_found_in_rck: ", unique_pattern_found_in_rck)
            print("Attempting to match rck with rpk...")
            rpk_rck_matching_idx_found, rpk_first_idx_in_rck = get_unique_pattern_ref_index(rck, rpk)
        if not rpk_rck_matching_idx_found:
            if corner_coordinates_available_in_rpk:
                print("Attempting to match rck with rpk using corner coordinates...")
                rpk_rck_matching_idx_found, rpk_first_idx_in_rck = get_unique_pattern_ref_index(rck, rpk, match_corner_coordinates=True)
            else:
                print("Attempting to find unique pattern in individual parameters...")
                rpk_rck_matching_idx_found, rpk_first_idx_in_rck = get_unique_pattern_ref_index(rck, rpk, find_match_in_individual_parameters=True)
        
        if rpk_rck_matching_idx_found and not rck_rearranged:
            rearrange_rck_using_prior_knowledge(rck, rpk_first_idx_in_rck)
            rck_rearranged = True
            shift_in_idx_for_rck = 0
            fill_missing_parameters(rck, rpk, rpk_rck_matching_idx_found)
            propagate_parameters(rck)
        
        rck.print_knowledge("Robot Current Knowledge (rck)")
        dof = find_dof(rck)
        print(f"DOF = {dof}")
        if to_plot:
            rck.plot_polygon("Reconstructed Polygon")
        steps += 1

if __name__ == "__main__":
    # Test simulation with default parameters
    random_seed = 7
    num_sides = 5
    regular_polygon = False
    
    angles_deg_random = generate_internal_angles(
        n_sides=num_sides,
        equal_angles=regular_polygon,
        random_seed=random_seed
    )
    print("Generated angles (degrees): ", angles_deg_random)
    
    poly_know_gt = generate_polygon(
        n_sides=num_sides,
        angles_deg=angles_deg_random,
        random_seed=random_seed
    )
    poly_know_gt.print_knowledge("Ground truth polygon knowledge")
    
    simulate_robot(
        to_plot=True, 
        polygon_knowledge_gt=poly_know_gt,
        degree_of_prior_knowledge=3,
        shift_in_idx_for_rck=2,
        percentage_of_edge_filled=1.0,
        random_seed=random_seed
    )
