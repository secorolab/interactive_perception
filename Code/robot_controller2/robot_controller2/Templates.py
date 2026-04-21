
#Amount of arguments per condition
POSITION_XYZ = 3,
FORCE_XYZ = 3,
VELOCITY_XYZ = 3,
TORQUE_RPY = 3,
ORIENTATION_QUATERNION = 4,
ORIENTATION_ROLL = 1,
ORIENTATION_PITCH = 1,
ORIENTATION_YAW = 1


reset = {
"arm_name": "KINOVA_GEN3_2_RIGHT",
'KINOVA_GEN3_2_RIGHT':{
    "action_name": "reset",
    "reach_pre_configuration_joint_angles": "true",
    "pre_configuration_max_deviation_deg": 200.0,
    "pre_configuration_joint_angles_tolerance_deg": 5.0,
    "pre_configuration_joint_angle_0_deg": 16.0,
    "pre_configuration_joint_angle_1_deg": 97.0, #a 80 >180 - 360 // <-180 +360
    "pre_configuration_joint_angle_2_deg": 123.0,
    "pre_configuration_joint_angle_3_deg": 99.0, #a 80
    "pre_configuration_joint_angle_4_deg": 154.6,
    "pre_configuration_joint_angle_5_deg": -65.0, #a 80
    "pre_configuration_joint_angle_6_deg": 301.0,
    'frame_name': 'eddie_base_link',
    'PRE_CONDITION': {
        'constraint_count': 1,
        'constraints': {
            1: {
                'type': 'POSITION_XYZ',
                'value': [None, None, None],
                'operator': [None, None, None]
            },
        }
    },
    "PER_CONDITION": {
        'constraint_count': 1,
        'constraints': {
            1: {
                'type': 'VELOCITY_XYZ',
                'value': [0.0, 0.0, 0.0],
                'operator': ['EQUAL', 'EQUAL', 'EQUAL']
            },
        }
    },
    "POST_CONDITION": {
        'constraint_count': 1,
        'number_of_disjunctions': 1,
        'constraints': {
            1: {
                "disjunction_id": 1,
                'type': 'POSITION_XYZ',
                'value': [100.0, 100.0, 100.0],
                'operator': ['LESS_THAN', 'LESS_THAN', 'LESS_THAN']
            },

        }

    }
}
}



neutral = {
"arm_name": "KINOVA_GEN3_2_RIGHT",
'KINOVA_GEN3_2_RIGHT':{
    "action_name": "neutral",
    "reach_pre_configuration_joint_angles": "false",
    'frame_name': 'ground_truth_object',
    'PRE_CONDITION': {
        'constraint_count': 1,
        'constraints': {
            1: {
                'type': 'POSITION_XYZ',
                'value': [None, None, None],
                'operator': [None, None, None]
            },
        }
    },
    "PER_CONDITION": {
        'constraint_count': 1,
        'constraints': {
            1: {
                'type': 'VELOCITY_XYZ',
                'value': [0.0, 0.0, 0.0],
                'operator': ['EQUAL', 'EQUAL', 'EQUAL']
            },
        }
    },
    "POST_CONDITION": {
        'constraint_count': 1,
        'number_of_disjunctions': 1,
        'constraints': {
            1: {
                "disjunction_id": 1,
                'type': 'POSITION_XYZ',
                'value': [None, None, None],
                'operator': [None, None, None] # TODO: check if it always satisfies post condition immediately
            },
        }
    }
}
}


move = {
"arm_name": "KINOVA_GEN3_2_RIGHT",
    'KINOVA_GEN3_2_RIGHT':{
        "action_name": "move",
        "reach_pre_configuration_joint_angles": "false",
        'frame_name': 'eddie_base_link',
        'PRE_CONDITION': {
            'constraint_count': 0
        },
        "PER_CONDITION": {
            'constraint_count': 2,
            'constraints': {
                1: {
                    'type': 'POSITION_XYZ',
                    'value': [None, None, None],
                    'operator': [None, None, None]
                },
                2: {
                    'type': 'ORIENTATION_QUATERNION',
                    'value': [None, None, None, None],
                    'operator': [None, None, None, None]
                },
            }
        },
        "POST_CONDITION": {
            'constraint_count': 2,
            'number_of_disjunctions': 1,
            'constraints': {
                1: {
                    "disjunction_id": 1,
                    'type': 'POSITION_XYZ',
                    'value': [None, None, None],
                    'operator': [None, None, None]
                },
                2: {
                    "disjunction_id": 1,
                    'type': 'POSITION_XYZ',
                    'value': [None, None, None],
                    'operator': [None, None, None]
                },
            }
        }
    }
}


touch = {
"arm_name": "KINOVA_GEN3_2_RIGHT",
'KINOVA_GEN3_2_RIGHT':{
    "action_name": "touch",
    "reach_pre_configuration_joint_angles": "false",
    'frame_name': 'eddie_base_link',
    'PRE_CONDITION': {
        'constraint_count': 0
    },
    "PER_CONDITION": {
        'constraint_count': 2,
        'constraints': {
            1: {
                'type': 'VELOCITY_XYZ',
                'value': [0.6, -0.5, 0.00],
                'operator': ['EQUAL', 'EQUAL', 'EQUAL']
            },
            2: {
                'type': 'ORIENTATION_QUATERNION',
                'value': [0.5, 0.4999999999999999, 0.5, 0.5000000000000001],
                'operator': ['EQUAL', 'EQUAL', 'EQUAL', 'EQUAL']
            },
        }
    },
    "POST_CONDITION": {
        'constraint_count': 3,
        'number_of_disjunctions': 1,
        'constraints': {
            1: {
                "disjunction_id": 1,
                'type': 'VELOCITY_XYZ',
                'value': [None, None, 0.005],
                'operator': [None, None, 'GREATER_THAN']
            },
            2: {
                "disjunction_id": 1,
                'type': 'VELOCITY_XYZ',
                'value': [None, None, 0.005],
                'operator': [None, None, 'GREATER_THAN']
            },
            3: {
                "disjunction_id": 1,
                'type': 'TiME_LIMIT',
                'value': None,
                'operator': None
                },
            }
        }
    }
}

slide = {
"arm_name": "KINOVA_GEN3_2_RIGHT",
'KINOVA_GEN3_2_RIGHT':{
    "action_name": "slide",
    "reach_pre_configuration_joint_angles": "false",
    'frame_name': 'marker_frame_0',
    'PRE_CONDITION': {
        'constraint_count': 1,
        'constraints': {
            1: {
                'type': 'POSITION_XYZ',
                'value': [None, None, None],
                'operator': [None, None, None]
            },
        }
    },
    "PER_CONDITION": {
        'constraint_count': 3,
        'constraints': {
            1: {
                'type': 'VELOCITY_XYZ',
                'value': [0.6, -0.5, 0.00],
                'operator': ['EQUAL', 'EQUAL', 'EQUAL']
            },
            2: {
                'type': 'FORCE_XYZ',
                'value': [None, None, None],
                'operator': [None, None, None]
            },
            3: {
                'type': 'ORIENTATION_QUATERNION',
                'value': [0.5, 0.4999999999999999, 0.5, 0.5000000000000001],
                'operator': ['EQUAL', 'EQUAL', 'EQUAL', 'EQUAL']
            }
        }
    },
    "POST_CONDITION": {
        'constraint_count': 6,
        'number_of_disjunctions': 6,
        'constraints': {
            1: {
                'disjunction_id': 1,
                'type': 'POSITION_XYZ',
                'value': [None, None, None],
                'operator': [None, None, None]
            },
            2: {
                'disjunction_id': 1,
                'type': 'POSITION_XYZ',
                'value': [None, None, None],
                'operator': [None, None, None]
            },
            3: {
                'disjunction_id': 1,
                'type': 'POSITION_XYZ',
                'value': [None, None, None],
                'operator': [None, None, None]
            },
            4: {
                'disjunction_id': 1,
                'type': 'POSITION_XYZ',
                'value': [None, None, None],
                'operator': [None, None, None]
            },
            5: {
                'disjunction_id': 1,
                'type': 'POSITION_XYZ',
                'value': [None, None, None],
                'operator': [None, None, None]
            },
            6: {
                'disjunction_id': 1,
                'type': 'POSITION_XYZ',
                'value': [None, None, None],
                'operator': [None, None, None]
            }
        }

    }
}
}

#

yaw = {
    "arm_name": "KINOVA_GEN3_2_RIGHT",
    'KINOVA_GEN3_2_RIGHT':{
        "action_name": "orient",
        "reach_pre_configuration_joint_angles": "false",
        'frame_name': 'marker_frame_0',
        'PRE_CONDITION': {
            'constraint_count': 0
        },
        "PER_CONDITION": {
            'constraint_count': 2,
            'constraints': {
                1: {
                    'type': 'POSITION_XYZ',
                    'value': [0.6, -0.5, 0.00],
                    'operator': ['EQUAL', 'EQUAL', 'EQUAL']
                },
                2: {
                    'type': 'ORIENTATION_YAW',
                    'value': 0.0,
                    'operator': 'EQUAL'
                }
            }
        },
        "POST_CONDITION": {
            'constraint_count': 3,
            'number_of_disjunctions': 2,
            'constraints': {
                1: {
                'disjunction_id': 1,
                'type': 'TIME_LIMIT',
                'value': 5.0,
                'operator': 'GREATER_THAN'
                },
                2: {
                'disjunction_id': 2,
                'type': 'ORIENTATION_YAW',
                'value': 0.1,
                'operator': 'LESS_THAN'
                },
                3: {
                'disjunction_id': 2,
                'type': 'ORIENTATION_YAW',
                'value': -0.1,
                'operator': 'GREATER_THAN'
                }
            }
        }
    }
}





