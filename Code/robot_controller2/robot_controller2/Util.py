import os
import subprocess
import math
from typing import List
from robot_controller2 import Templates
from geometry_msgs.msg import Quaternion
import json
from enum import Enum
#from tf_transformations import quaternion_from_euler
from sympy.physics.units import force, velocity
from tf_transformations import euler_from_quaternion
from tf_transformations import quaternion_from_euler
import numpy as np
from scipy.spatial.transform import Rotation as Rscipy

gt = "GREATER_THAN"
eq = "EQUAL"
lt = "LESS_THAN"
pre = "PRE_CONDITION"
per = "PER_CONDITION"
prev = "PREVAIL_CONDITION"
post = "POST_CONDITION"

class StateOfExecution(Enum):
    IDLE = 0
    WAITING_FOR_SERVER = 1
    EXECUTING = 2
    COMPLETED = 3
    FAILED = 4

def pose_from_points(points, use_ransac=False, max_iterations=100, distance_threshold=0.005, min_inliers_ratio=0.7):
    """
    Given >=3 points in 3D, compute:
      - centroid (middle point)
      - pose: position + quaternion with Z axis as normal to the points plane
    
    If use_ransac=True and len(points) > 3, RANSAC is applied to remove noise
    and fit a plane to inlier points. For 3 points, RANSAC is skipped.
    """
    pts = np.array(points)

    # RANSAC plane fitting to remove noise (only if > 3 points)
    if use_ransac and len(pts) > 3:
        min_inliers = max(3, int(min_inliers_ratio * len(pts)))  # At least min_inliers_ratio of points as inliers
        
        best_inliers = None
        best_inlier_count = 0
        
        for _ in range(max_iterations):
            # Randomly select 3 points to define a plane
            sample_indices = np.random.choice(len(pts), 3, replace=False)
            sample_points = pts[sample_indices]
            
            # Calculate plane normal from 3 points
            v1 = sample_points[1] - sample_points[0]
            v2 = sample_points[2] - sample_points[0]
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            
            if norm < 1e-6:  # Points are collinear, skip this sample
                continue
            
            normal = normal / norm
            
            # Calculate signed distance of all points to this plane
            # Using point-to-plane distance formula: |dot(p - p0, n)|
            d = sample_points[0]  # Point on plane
            distances = np.abs(np.dot(pts - d, normal))
            
            # Find inliers (points close to the plane)
            inliers = distances < distance_threshold
            inlier_count = np.sum(inliers)
            
            # Keep best model
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_inliers = inliers
        
        # Use inliers if enough are found, otherwise use all points
        if best_inliers is not None and best_inlier_count >= min_inliers:
            pts = pts[best_inliers]
        else:
            print("Not enough inliers found, using all points for pose estimation.")
        # If not enough inliers found, keep all points
    
    # Centroid from (filtered or original) points
    centroid = np.mean(pts, axis=0)

    # Estimate normal via PCA
    cov = np.cov((pts - centroid).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, np.argmin(eigvals)]
    normal /= np.linalg.norm(normal)

    # Create local axes
    z_axis = normal

    # Pick x_axis as projection of global x
    ref = np.array([1.0, 0.0, 0.0])

    x_axis = ref - np.dot(ref, z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    R = np.stack([x_axis, y_axis, z_axis], axis=1)

    # Convert to quaternion
    quaternion = Rscipy.from_matrix(R).as_quat()

    return centroid, quaternion

def quat_vel_to_rpy(velocity):
    """
    Convert velocity + quaternion to RPY.
    Roll and pitch are always 0.0, yaw depends on velocity.
    """
    vx, vy, vz = velocity

    # Default yaw = 0 if velocity is zero
    if abs(vx) < 1e-6 and abs(vy) < 1e-6:
        yaw = 0.0
    else:
        # atan2 gives direction of velocity in the XY plane
        vel_yaw = np.degrees(np.arctan2(vy, vx))

        yaw = vel_yaw + 90.0

    return [180.0, 0.0, yaw]

def quat_vel_to_rpy_yaw(velocity):
    """
    Convert velocity + quaternion to RPY.
    Roll and pitch are always 0.0, yaw depends on velocity.
    """
    vx, vy, vz = velocity

    # Default yaw = 0 if velocity is zero
    if abs(vx) < 1e-6 and abs(vy) < 1e-6:
        yaw = 0.0
    else:
        # atan2 gives direction of velocity in the XY plane
        vel_yaw = np.degrees(np.arctan2(vy, vx))

        yaw = vel_yaw + 90.0

    return float(yaw)

def is_within_range(value: float, target: float, deviation: float) -> bool:
    if value is None:
        return True
    else:
        return math.isclose(value, target, abs_tol=deviation)



def edit_condition(input, arguments):
    if "action_name" in arguments:
        input["KINOVA_GEN3_2_RIGHT"]['action_name'] = arguments["action_name"]
    if "frame_name" in arguments:
        input["KINOVA_GEN3_2_RIGHT"]['frame_name'] = arguments["frame_name"]
    if "constraint_count" in arguments:
        input["KINOVA_GEN3_2_RIGHT"][arguments["condition_type"]]['constraint_count'] = arguments["constraint_count"]
    if "number_of_disjunctions" in arguments:
        input["KINOVA_GEN3_2_RIGHT"][arguments["condition_type"]]['number_of_disjunctions'] = arguments["number_of_disjunctions"]
    if "position" in arguments:
        # 'position' is the index of constraint in the motion specification
        cnt = 0
        for x in input["KINOVA_GEN3_2_RIGHT"][arguments["condition_type"]]['constraints']:
            cnt = cnt + 1
            if arguments['position'] == cnt:
                if "disjunction_id" in arguments:
                    input["KINOVA_GEN3_2_RIGHT"][arguments["condition_type"]]["constraints"][x] = {
                        'disjunction_id': arguments["disjunction_id"],
                        'type': arguments["type"],
                        'value': arguments["value"],
                        'operator': arguments["operator"]}
                else:
                 input["KINOVA_GEN3_2_RIGHT"][arguments["condition_type"]]["constraints"][x] = {
                        'type': arguments["type"],
                        'value': arguments["value"],
                        'operator': arguments["operator"]}
                return input
    else: return input


def make_action_goal_move(position, orientation, frame_name="marker_frame_0"):
    current_template = Templates.move
    tolerance = 0.01

    px, py, pz = position
    # Convert numpy types to Python native types
    px = float(px) if px is not None else None
    py = float(py) if py is not None else None
    pz = float(pz) if pz is not None else None
    
    xl, yl, zl = px - tolerance, py - tolerance, pz - tolerance
    xh, yh, zh = px + tolerance, py + tolerance, pz + tolerance
    
    qx, qy, qz, qw = orientation
    qx = float(qx) if qx is not None else None
    qy = float(qy) if qy is not None else None
    qz = float(qz) if qz is not None else None
    qw = float(qw) if qw is not None else None

    current_template = edit_condition(current_template, {
                              "frame_name": frame_name
                              })

    current_template = edit_condition(current_template, {
                              "condition_type": "PER_CONDITION",
                              "position": 1,
                              "type": "POSITION_XYZ",
                              "value": [px, py, pz],
                              "operator": [eq, eq, eq]
                              })

    current_template = edit_condition(current_template, {
                                "condition_type": "PER_CONDITION",
                                "position": 2,
                                "type": "ORIENTATION_QUATERNION",
                                "value": [qx, qy, qz, qw],
                                "operator": [eq, eq, eq, eq]
                              })

    current_template = edit_condition(current_template, {
                                "condition_type": "POST_CONDITION",
                                "disjunction_id": 1,
                                "position": 1,
                                "type": "POSITION_XYZ",
                                "value": [xl, yl, zl],
                                "operator": [gt, gt, gt]
                              })

    current_template = edit_condition(current_template, {
                                "condition_type": "POST_CONDITION",
                                "disjunction_id": 1,
                                "position": 2,
                                "type": "POSITION_XYZ",
                                "value": [xh, yh, zh],
                                "operator": [lt, lt, lt]
                              })

    return str(current_template)


def make_action_goal_touch(velocity, orientation, action_name="touch_neutral", frame_name="eddie_base_link", min_contact_time=2.0):
    current_template = Templates.touch
    velocity_x, velocity_y, velocity_z = velocity
    # Convert numpy types to Python native types
    velocity_x = float(velocity_x) if velocity_x is not None else None
    velocity_y = float(velocity_y) if velocity_y is not None else None
    velocity_z = float(velocity_z) if velocity_z is not None else None
    
    qx, qy, qz, qw = orientation
    qx = float(qx) if qx is not None else None
    qy = float(qy) if qy is not None else None
    qz = float(qz) if qz is not None else None
    qw = float(qw) if qw is not None else None

    if action_name:
        current_template = edit_condition(current_template, {
            "action_name": action_name
        })
    else:
        current_template = edit_condition(current_template, {
            "action_name": "touch_neutral"
        })

    current_template = edit_condition(current_template, {
        "frame_name": frame_name
    })

    current_template = edit_condition(current_template, {
        "condition_type": "PER_CONDITION",
        "position": 1,
        "type": "VELOCITY_XYZ",
        "value": [velocity_x, velocity_y, velocity_z],
        "operator": [eq, eq, eq]
    })
    current_template = edit_condition(current_template, {
        "condition_type": "PER_CONDITION",
        "position": 2,
        "type": "ORIENTATION_QUATERNION",
        "value": [qx, qy, qz, qw],
        "operator": [eq, eq, eq, eq]
    })
    
    vel_threshold = 0.005

    if action_name == "touch_neutral":
        current_template = edit_condition(current_template, {
            "condition_type": "POST_CONDITION",
            "number_of_disjunctions": 1,
            "disjunction_id": 1,
            "constraint_count": 2,
            "position": 1,
            "type": "VELOCITY_XYZ",
            "value": [None, None, -vel_threshold],
            "operator": [None, None, gt]
        })
        current_template = edit_condition(current_template, {
            "condition_type": "POST_CONDITION",
            "number_of_disjunctions": 1,
            "disjunction_id": 1,
            "position": 2,
            "type": "TIME_LIMIT",
            "value": min_contact_time,
            "operator": gt
        })
        
    elif action_name == "touch_edge":
        zero_vel_threshold = 0.005
        velocity_ul = zero_vel_threshold
        velocity_ll = -zero_vel_threshold
        
        current_template = edit_condition(current_template, {
            "condition_type": "POST_CONDITION",
            "number_of_disjunctions": 1,
            "disjunction_id": 1,
            "constraint_count": 3,
            "position": 1,
            "type": "VELOCITY_XYZ",
            "value": [velocity_ul, velocity_ul, velocity_ul],
            "operator": [lt, lt, lt]
        })
        current_template = edit_condition(current_template, {
            "condition_type": "POST_CONDITION",
            "number_of_disjunctions": 1,
            "disjunction_id": 1,
            "constraint_count": 3,
            "position": 2,
            "type": "VELOCITY_XYZ",
            "value": [velocity_ll, velocity_ll, velocity_ll],
            "operator": [gt, gt, gt]
        })
        current_template = edit_condition(current_template, {
            "condition_type": "POST_CONDITION",
            "disjunction_id": 1,
            "position": 3,
            "type": "TIME_LIMIT",
            "value": min_contact_time,
            "operator": gt
        })

    return str(current_template)


def make_action_goal_slide(velocity, force, orientation, action_name="slide_to_explore_plane", frame_name="eddie_base_link", time=1.5):
    """
    position is the id of constraint to be updated
    """
    
    current_template = Templates.slide

    velocity = list(velocity)
    # Convert numpy types to Python native types
    velocity = [float(v) if v is not None else None for v in velocity]
    velocity_x, velocity_y, velocity_z = velocity
    vel_opr_list = [eq if v is not None else None for v in velocity]
    
    force = [float(f) if f is not None else None for f in force]
    force_opr_list = [eq if f is not None else None for f in force]

    if orientation:
        qx, qy, qz, qw = orientation
        qx = float(qx) if qx is not None else None
        qy = float(qy) if qy is not None else None
        qz = float(qz) if qz is not None else None
        qw = float(qw) if qw is not None else None

    if action_name:
        current_template = edit_condition(current_template, {
            "action_name": action_name
        })

    current_template = edit_condition(current_template, {
        "frame_name": frame_name
    })

    current_template = edit_condition(current_template, {
        "condition_type": "PER_CONDITION",
        "position": 1,
        "type": "VELOCITY_XYZ",
        "value": velocity,
        "operator": vel_opr_list
    })
    
    current_template = edit_condition(current_template, {
        "condition_type": "PER_CONDITION",
        "position": 2,
        "type": "FORCE_XYZ",
        "value": force,
        "operator": force_opr_list
    })

    if orientation:
        current_template = edit_condition(current_template, {
            "condition_type": "PER_CONDITION",
            "position": 3,
            "type": "ORIENTATION_QUATERNION",
            "value": [qx, qy, qz, qw],
            "operator": [eq, eq, eq, eq]
        })
    else:
        current_template = edit_condition(current_template, {
            "condition_type": "PER_CONDITION",
            "position": 3,
            "type": "ORIENTATION_QUATERNION",
            "value": [None, None, None, None],
            "operator": [None, None, None, None]
        })

    if action_name == "slide_plane_single" or action_name == "slide_plane_multiple":

        #Calculate variables
        min_velocity = 0.01

        if abs(velocity_x) >= abs(velocity_y):
            # X is dominant
            x_post = -min_velocity if velocity_x < 0 else min_velocity
            op_x = gt if velocity_x < 0 else lt
            y_post, op_y = None, None
        else:
            # Y is dominant
            y_post = -min_velocity if velocity_y < 0 else min_velocity
            op_y = gt if velocity_y < 0 else lt
            x_post, op_x = None, None

        current_template = edit_condition(current_template, {
            "condition_type": "POST_CONDITION",
            "number_of_disjunctions": 3,
            "disjunction_id": 1,
            "position": 1,
            "type": "VELOCITY_XYZ",
            "value": [None, None, -0.03],
            "operator": [None, None, lt]
        })

        current_template = edit_condition(current_template, {
            "condition_type": "POST_CONDITION",
            "disjunction_id": 1,
            "position": 2,
            "type": "POSITION_XYZ",
            "value": [None, None, -0.03],
            "operator": [None, None, lt]
        })

        current_template = edit_condition(current_template, {
            "condition_type": "POST_CONDITION",
            "disjunction_id": 2,
            "position": 3,
            "type": "VELOCITY_XYZ",
            "value": [x_post, y_post, None],
            "operator": [op_x, op_y, None]
        })

        current_template = edit_condition(current_template, {
            "condition_type": "POST_CONDITION",
            "disjunction_id": 2,
            "position": 4,
            "type": "TIME_LIMIT",
            "value": 0.5,
            "operator": gt
        })

        current_template = edit_condition(current_template, {
            "condition_type": "POST_CONDITION",
            "disjunction_id": 3,
            "position": 5,
            "type": "POSITION_XYZ",
            "value": [None, None, -0.03],
            "operator": [None, None, lt]
        })

        current_template = edit_condition(current_template, {
            "condition_type": "POST_CONDITION",
            "disjunction_id": 3,
            "position": 6,
            "type": "TIME_LIMIT",
            "value": 1.0,
            "operator": gt
        })


    elif action_name == "slide_edge":
        min_velocity = 0.1

        if abs(velocity_x) >= abs(velocity_y):
            # X is dominant
            x_post = min_velocity if velocity_x < 0 else -min_velocity
            op_x = gt if velocity_x < 0 else lt
            y_post, op_y = None, None
        else:
            # Y is dominant
            y_post = min_velocity if velocity_y < 0 else -min_velocity
            op_y = gt if velocity_y < 0 else lt
            x_post, op_x = None, None
        
        current_template = edit_condition(current_template, {
                "condition_type": "POST_CONDITION",
                "number_of_disjunctions": 3,
                "disjunction_id": 1,
                "position": 1,
                "type": "VELOCITY_XYZ",
                "value": [x_post, y_post, None],
                "operator": [op_x, op_y, None]
            })

        current_template = edit_condition(current_template, {
                "condition_type": "POST_CONDITION",
                "disjunction_id": 2,
                "position": 2,
                "type": "TIME_LIMIT",
                "value": time,
                "operator": gt
            })

        current_template = edit_condition(current_template, {
                "condition_type": "POST_CONDITION",
                "disjunction_id": 3,
                "position": 3,
                "type": "TIME_LIMIT",
                "value": time,
                "operator": gt
            })

        current_template = edit_condition(current_template, {
                "condition_type": "POST_CONDITION",
                "disjunction_id": 4,
                "position": 4,
                "type": "TIME_LIMIT",
                "value": time,
                "operator": gt
            })

    elif action_name == "slide_to_explore_plane":
        """
        disjunction 1: slide over surface until time limit (to explore plane)
        """
        current_template = edit_condition(current_template, {
                "condition_type": "POST_CONDITION",
                "number_of_disjunctions": 1,
                "constraint_count": 1,
                "disjunction_id": 1,
                "position": 1,
                "type": "TIME_LIMIT",
                "value": time,
                "operator": gt
            })

    elif action_name == "slide_until_edge":
        """
        disjunction 1: slide until contact with a surface where dihedral angle=90
        disjunction 2: slide until contact with a surface where dihedral angle=270
        """
        
        vel_tolerance = 0.005
        zero_vel_ul = vel_tolerance
        zero_vel_ll = -vel_tolerance
        
        # disjunction 1
        current_template = edit_condition(current_template, {
                "condition_type": "POST_CONDITION",
                "number_of_disjunctions": 2,
                "constraint_count": 6,
                "disjunction_id": 1,
                "position": 1,
                "type": "VELOCITY_XYZ",
                "value": [zero_vel_ul, zero_vel_ul, zero_vel_ul],
                "operator": [lt, lt, lt]
            })
        current_template = edit_condition(current_template, {
                "condition_type": "POST_CONDITION",
                "disjunction_id": 1,
                "position": 2,
                "type": "VELOCITY_XYZ",
                "value": [zero_vel_ll, zero_vel_ll, zero_vel_ll],
                "operator": [gt, gt, gt]
            })
        current_template = edit_condition(current_template, {
                "condition_type": "POST_CONDITION",
                "disjunction_id": 1,
                "position": 3,
                "type": "POSITION_XYZ",
                "value": [None, None, -0.01],
                "operator": [None, None, gt]
            })
        current_template = edit_condition(current_template, {
                "condition_type": "POST_CONDITION",
                "disjunction_id": 1,
                "position": 4,
                "type": "TIME_LIMIT",
                "value": 3.0,
                "operator": gt
            })

        # disjunction 2
        current_template = edit_condition(current_template, {
                "condition_type": "POST_CONDITION",
                "disjunction_id": 2,
                "position": 5,
                "type": "VELOCITY_XYZ",
                "value": [None, None, -0.015],
                "operator": [None, None, lt]
            })
        current_template = edit_condition(current_template, {
                "condition_type": "POST_CONDITION",
                "disjunction_id": 2,
                "position": 6,
                "type": "POSITION_XYZ",
                "value": [None, None, -0.01],
                "operator": [None, None, lt]
            })
    return str(current_template)


def make_action_goal_yaw(position, yaw=0.0, yaw_threshold=0.1, frame_name="eddie_base_link", time_limit=5.0):
    current_template = Templates.yaw

    pos_x, pos_y, pos_z = position
    # Convert numpy types to Python native types
    pos_x = float(pos_x) if pos_x is not None else None
    pos_y = float(pos_y) if pos_y is not None else None
    pos_z = float(pos_z) if pos_z is not None else None
    yaw = float(yaw) if yaw is not None else None
    yaw_threshold = float(yaw_threshold) if yaw_threshold is not None else None
    time_limit = float(time_limit) if time_limit is not None else None

    current_template = edit_condition(current_template, {
        "frame_name": frame_name
    })
    current_template = edit_condition(current_template, {
        "condition_type": "PER_CONDITION",
        "position": 1,
        "type": "POSITION_XYZ",
        "value": [pos_x, pos_y, pos_z],
        "operator": [eq, eq, eq]
    })
    current_template = edit_condition(current_template, {
        "condition_type": "PER_CONDITION",
        "position": 2,
        "type": "ORIENTATION_YAW",
        "value": yaw,
        "operator": eq
    })

    current_template = edit_condition(current_template, {
        "condition_type": "POST_CONDITION",
        "disjunction_id": 1,
        "position": 1,
        "type": "TIME_LIMIT",
        "value": time_limit,
        "operator": gt
    })

    current_template = edit_condition(current_template, {
        "condition_type": "POST_CONDITION",
        "disjunction_id": 2,
        "position": 2,
        "type": "ORIENTATION_YAW",
        "value": yaw + yaw_threshold,
        "operator": lt
    })

    current_template = edit_condition(current_template, {
        "condition_type": "POST_CONDITION",
        "disjunction_id": 2,
        "position": 3,
        "type": "ORIENTATION_YAW",
        "value": yaw - yaw_threshold,
        "operator": gt
    })

    print(str(current_template))
    #print(json.dumps(current_template, indent=2))
    return str(current_template)