import os
import subprocess
import math
from robot_controller2 import Templates
from geometry_msgs.msg import Quaternion
import json
#from tf_transformations import quaternion_from_euler
from sympy.physics.units import velocity
from tf_transformations import euler_from_quaternion
from tf_transformations import quaternion_from_euler
import numpy as np
from scipy.spatial.transform import Rotation as R


gt = "GREATER_THAN"
eq = "EQUAL"
lt = "LESS_THAN"
pre = "PRE_CONDITION"
per = "PER_CONDITION"
prev = "PREVAIL_CONDITION"
post = "POST_CONDITION"


def calculate_quaternion(roll, pitch, yaw):
    roll = roll * math.pi / 180.0

    pitch = pitch * math.pi / 180.0

    yaw = yaw * math.pi / 180.0

    cy = math.cos(yaw * 0.5)

    sy = math.sin(yaw * 0.5)

    cp = math.cos(pitch * 0.5)

    sp = math.sin(pitch * 0.5)

    cr = math.cos(roll * 0.5)

    sr = math.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy

    qx = sr * cp * cy - cr * sp * sy

    qy = cr * sp * cy + sr * cp * sy

    qz = cr * cp * sy - sr * sp * cy

    return [qx, qy, qz, qw]


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


def look_down_rpy():
    return[180, 0, 0]

def look_down_mf_rpy():
    return[0, 0, 0]

def look_forward():
    return calculate_quaternion(90, 0, 90)

def look_down():
    return calculate_quaternion(180, 0, 0)

def look_down_mf():
    return calculate_quaternion(0, 0, 0)

def look_down_mf_test():
    return calculate_quaternion(0, 0, -90)

def look_custom():
    return calculate_quaternion(180, 180, 80)

def is_negative(value):
    return value < 0


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


def make_action_goal_move(position, orientation, current_position, frame_name="marker_frame_0"):
    current_template = Templates.move

    current_x, current_y, current_z = current_position
    x, y, z = position
    qx, qy, qz, qw = orientation

    # Helper for axis logic
    def process_axis(target, current, tolerance):
        if target is None:
            # No target, keep current position
            return current, None, None, None, None
        if not is_within_range(target, current, tolerance):
            # Need to move
            low = target - tolerance
            high = target + tolerance
            return target, low, high, gt, lt
        # Already within tolerance, keep current
        return current, None, None, None, None
    # X axis
    px, xl, xh, xo1, xo2 = process_axis(x, current_x, 0.02)
    # Y axis
    py, yl, yh, yo1, yo2 = process_axis(y, current_y, 0.02)
    # Z axis
    pz, zl, zh, zo1, zo2 = process_axis(z, current_z, 0.02)


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
                              "operator": [xo1, yo1, zo1]
                              })

    current_template = edit_condition(current_template, {
                              "condition_type": "POST_CONDITION",
                              "disjunction_id": 1,
                              "position": 2,
                              "type": "POSITION_XYZ",
                              "value": [xh, yh, zh],
                              "operator": [xo2, yo2, zo2]
                              })

    # print("Template:")
    # print(current_template)
    return str(current_template)


def make_action_goal_touch(velocity, orientation, current_position,mode="touch_neutral", frame_name="eddie_base_link"):
    current_template = Templates.touch

    velocity_x, velocity_y, velocity_z = velocity
    current_x, current_y, current_z = current_position

    qx, qy, qz, qw = orientation

    if mode:
        current_template = edit_condition(current_template, {
            "action_name": mode
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

    pairing = [[velocity_x, current_x], [velocity_y, current_y], [velocity_z, current_z]]

    value_per = []
    operator_per = []

    for velocity, position in pairing:
        if velocity is None or abs(velocity) < 1e-6:
            value_per.append(position)
            operator_per.append(eq)
        else:
            value_per.append(None)
            operator_per.append(None)


    # Post-condition: contact detection for each axis
    # Threshold to detect contact
    threshold = 0.005

    value = []
    operator = []

    for v in (velocity_x, velocity_y, velocity_z):
        if v is None or abs(v) < 1e-6:
            value.append(0.0)
            operator.append(eq)
        elif v > 0:
            value.append(threshold)
            operator.append(lt)
        else:  # v < 0
            value.append(-threshold)
            operator.append(gt)

    current_template = edit_condition(current_template, {
        "condition_type": "POST_CONDITION",
        "number_of_disjunctions": 1,
        "disjunction_id": 1,
        "position": 1,
        "type": "VELOCITY_XYZ",
        "value": value,
        "operator": operator
    })

    current_template = edit_condition(current_template, {
        "condition_type": "POST_CONDITION",
        "number_of_disjunctions": 1,
        "disjunction_id": 1,
        "position": 2,
        "type": "TIME_LIMIT",
        "value": 2.0,
        "operator": gt
    })

    #print(current_template)
    return str(current_template)



def make_action_goal_slide(velocity, orientation=None, mode="slide_plane_single", frame_name="marker_frame_0", time=1.5):
        current_template = Templates.slide

        velocity_x, velocity_y, velocity_z = velocity

        if orientation:
            qx, qy, qz, qw = orientation


        if mode:
            current_template = edit_condition(current_template, {
                "action_name": mode
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

        if orientation:
            current_template = edit_condition(current_template, {
                "condition_type": "PER_CONDITION",
                "position": 2,
                "type": "ORIENTATION_QUATERNION",
                "value": [qx, qy, qz, qw],
                "operator": [eq, eq, eq, eq]
            })


        if orientation is None:
            current_template = edit_condition(current_template, {
                "condition_type": "PER_CONDITION",
                "position": 2,
                "type": "ORIENTATION_QUATERNION",
                "value": [None, None, None, None],
                "operator": [None, None, None, None]
            })


        if mode == "slide_edge":
            current_template = edit_condition(current_template, {
                "condition_type": "PER_CONDITION",
                "position": 3,
                "type": "POSITION_XYZ",
                "value": [None, None, -0.03],
                "operator": [None, None, lt]
            })






        if mode == "slide_plane_single" or mode == "slide_plane_multiple":

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


        elif mode == "slide_edge":


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

        elif mode == "slide_blind":


            current_template = edit_condition(current_template, {
                    "condition_type": "POST_CONDITION",
                    "number_of_disjunctions": 1,
                    "disjunction_id": 1,
                    "position": 2,
                    "type": "TIME_LIMIT",
                    "value": time,
                    "operator": gt
                })

            current_template = edit_condition(current_template, {
                    "condition_type": "POST_CONDITION",
                    "disjunction_id": 1,
                    "position": 2,
                    "type": "TIME_LIMIT",
                    "value": time,
                    "operator": gt
                })

            current_template = edit_condition(current_template, {
                    "condition_type": "POST_CONDITION",
                    "disjunction_id": 1,
                    "position": 3,
                    "type": "TIME_LIMIT",
                    "value":time,
                    "operator": gt
                })

            current_template = edit_condition(current_template, {
                    "condition_type": "POST_CONDITION",
                    "disjunction_id": 1,
                    "position": 4,
                    "type": "TIME_LIMIT",
                    "value": time,
                    "operator": gt
                })



        #print(current_template)

        return str(current_template)


def make_action_goal_yaw(position, yaw=0.0, yaw_threshold=0.1, frame_name="eddie_base_link", time_limit=5.0):
    current_template = Templates.yaw

    pos_x, pos_y, pos_z = position

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