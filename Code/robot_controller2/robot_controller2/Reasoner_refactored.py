import copy
import math
import numpy as np
from typing import Optional, List, Tuple
from enum import Enum
import logging
import networkx as nx
import yaml
import os
from dataclasses import replace
from typing import Literal
import json
import ast
import traceback
import hashlib
import platform
import subprocess
import sys
from datetime import datetime
from scipy.spatial.transform import Rotation as R

# ROS 2 imports
from rclpy.node import Node as ROSNode
import rclpy
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped, Pose, Quaternion, Point, Vector3
from std_msgs.msg import Header, ColorRGBA
from action_msgs.msg import GoalStatus
from rclpy.task import Future
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import tf2_geometry_msgs
from motion_specification_interfaces.action import MotionSpecification

# Local imports
from robot_controller2 import Util, Templates, Graph

from core_algorithm import (
    PolygonKnowledge,
    ActionType,
    ActionInstance,
    ActionSpec,
    Direction,
    Mode,
    Stop,
    next_action,
    propagate_parameters,
    find_unique_pattern,
    get_unique_pattern_ref_index,
    get_last_action_selection_trace,
    find_dof,
    rearrange_rck_using_prior_knowledge,
    fill_missing_parameters,
    ACTION_TO_SPEC
)

logger = logging.getLogger(__name__)

class PolygonKnowledgeToGraphConverter:
    """
    Converts PolygonKnowledge objects to Graph JSON format
    """
    @staticmethod
    def polygon_knw_to_json(pk: PolygonKnowledge, 
                      frame_name: str = "plane_frame_0",
                      polygon_id: str = "polygon_0",
                      ref_frame_pose: Optional[dict] = None,
                      noisy_points_on_edge: Optional[List[List[Tuple[float, float]]]] = None) -> dict:
        """
        Convert PolygonKnowledge to Graph JSON format suitable for create_graph_from_json().
        
        Architecture:
        - Nodes reference data via string IDs (e.g., "pt_0_x", "line_segment_0_length")
        - Data_structure contains actual values indexed by these IDs
        - This separation allows the graph module to update values independently
        
        :param pk: PolygonKnowledge instance
        :param frame_name: Reference frame name
        :param polygon_id: ID for the polygon node
        :param ref_frame_pose: Optional estimated pose for the reference frame
        :param noisy_points_on_edge: Optional raw/noisy edge samples for visualization

        :return: JSON-compatible dict with frame, nodes, data_structure, and edges
        """
        n_sides = pk.n_sides
        nodes = []
        data_structure = []
        edges = []

        def points_to_xyz_list(points):
            points_list = []
            if points is not None:
                for point in points:
                    if len(point) == 2:
                        points_list.append([float(point[0]), float(point[1]), 0.0])
                    elif len(point) == 3:
                        points_list.append([float(point[0]), float(point[1]), float(point[2])])
                    else:
                        points_list.append([float(value) for value in point])
            return points_list
        
        # ========== CREATE CORNER NODES AND DATA ==========
        for i in range(n_sides):
            # Node definition references data via IDs
            nodes.append({
                "id": f"pt_{i}",
                "type": "corner",
                "data": [f"pt_{i}_x", f"pt_{i}_y", f"pt_{i}_z"]
            })
            
            # Create data entries for corner coordinates
            if pk.corners[i] is not None:
                x_val, y_val = pk.corners[i]
                z_val = 0.0
            else:
                x_val, y_val, z_val = None, None, None
            
            data_structure.extend([
                {"id": f"pt_{i}_x", "type": "float", "value": x_val},
                {"id": f"pt_{i}_y", "type": "float", "value": y_val},
                {"id": f"pt_{i}_z", "type": "float", "value": z_val}
            ])
        
        # ========== CREATE SEGMENT NODES AND DATA ==========
        for i in range(n_sides):
            next_i = (i + 1) % n_sides
            
            # Node definition references data via IDs
            nodes.append({
                "id": f"line_segment_{i}",
                "type": "segment",
                "length": f"line_segment_{i}_length",
                "slope_angle_deg": f"line_segment_{i}_slope",
                "corners": [f"pt_{i}", f"pt_{next_i}"],
                "edge_unit_vector": f"line_segment_{i}_unit_vector"
            })
            
            nodes.append({
                "id": f"line_segment_{i}_unit_vector",
                "type": "vector3",
                "data": [f"line_segment_{i}_unit_vector_x", f"line_segment_{i}_unit_vector_y", f"line_segment_{i}_unit_vector_z"]
            })

            
            # Create data entries for segment properties
            length_val = pk.lengths[i] if pk.lengths[i] is not None else None
            slope_val = pk.slopes[i] if pk.slopes[i] is not None else None
            edge_uv = pk.edge_unit_vectors[i] if pk.edge_unit_vectors[i] is not None else (None, None)
            
            data_structure.extend([
                {"id": f"line_segment_{i}_length", "type": "float", "value": length_val},
                {"id": f"line_segment_{i}_slope", "type": "float", "value": slope_val},
                {"id": f"line_segment_{i}_unit_vector_x", "type": "float", "value": edge_uv[0]},
                {"id": f"line_segment_{i}_unit_vector_y", "type": "float", "value": edge_uv[1]},
                {"id": f"line_segment_{i}_unit_vector_z", "type": "float", "value": 0.0}
            ])
        
        # ========== CREATE POLYGON NODE ==========
        nodes.append({
            "id": polygon_id,
            "type": "polygon",
            "corners": [f"pt_{i}" for i in range(n_sides)]
        })
        
        # ========== CREATE CORNER ANGLE NODES AND DATA ==========
        for i in range(n_sides):
            # Node definition references data via IDs
            nodes.append({
                "id": f"corner_angle_{i}",
                "type": "2d_corner_angle",
                "angle": f"corner_angle_{i}_deg",
                "polygon_id": polygon_id,
                "corner": f"pt_{i}",
                "is_refexive": f"is_refexive_ang_{i}"
            })
            
            # Create data entries for corner angles
            angle_val = pk.corner_angles[i] if pk.corner_angles[i] is not None else None
            is_reflex_val = pk.is_reflexive_angle[i] if pk.is_reflexive_angle[i] is not None else None
            
            data_structure.extend([
                {"id": f"corner_angle_{i}_deg", "type": "float", "value": angle_val},
                {"id": f"is_refexive_ang_{i}", "type": "bool", "value": is_reflex_val}
            ])
        
        # ========== CREATE EDGE ANGLE NODES (DIHEDRALS) AND DATA ==========
        for i in range(n_sides):
            # Node definition references data via ID
            nodes.append({
                "id": f"dihedral_angle_{i}",
                "type": "2d_dihedral_angle",
                "angle": f"dihedral_angle_{i}_deg",
                "segment": f"line_segment_{i}",
                "polygons": [polygon_id, f"polygon_{i+1}"]
            })
            
            # Create data entry for dihedral angle
            dihedral_val = pk.dihedrals[i] if pk.dihedrals[i] is not None else None
            data_structure.append({
                "id": f"dihedral_angle_{i}_deg", "type": "float", "value": dihedral_val
            })
        
        # ========== CREATE EXPLORATORY DATA NODES (INTERNAL POINTS ON EDGES) ==========
        for i in range(n_sides):
            # Node definition for internal points on edge
            nodes.append({
                "id": f"internal_points_on_edge_{i}",
                "type": "exploratory_points",
                "segment": f"line_segment_{i}",
                "points_data_id": f"internal_points_on_edge_{i}_data"
            })
            
            # Create data entry for internal points on edge
            # Store as list of [x, y, z] coordinates
            points_list = points_to_xyz_list(pk.internal_points_on_edge[i])
            
            data_structure.append({
                "id": f"internal_points_on_edge_{i}_data",
                "type": "points_array",
                "value": points_list
            })

            nodes.append({
                "id": f"noisy_points_on_edge_{i}",
                "type": "noisy_exploratory_points",
                "segment": f"line_segment_{i}",
                "points_data_id": f"noisy_points_on_edge_{i}_data"
            })

            noisy_points = []
            if noisy_points_on_edge is not None and i < len(noisy_points_on_edge):
                noisy_points = noisy_points_on_edge[i]

            data_structure.append({
                "id": f"noisy_points_on_edge_{i}_data",
                "type": "points_array",
                "value": points_to_xyz_list(noisy_points)
            })
        
        # ========== CREATE EDGES ==========
        # Polygon to segment edges
        for i in range(n_sides):
            edges.append([polygon_id, f"line_segment_{i}"])
        
        # Polygon to corner edges
        for i in range(n_sides):
            edges.append([polygon_id, f"pt_{i}"])
        
        # Corner to segment connections (local edges)
        for i in range(n_sides):
            next_i = (i + 1) % n_sides
            edges.append([f"pt_{i}", f"line_segment_{i}"])
            edges.append([f"pt_{next_i}", f"line_segment_{i}"])
        
        # Corner angle connections
        for i in range(n_sides):
            edges.append([f"corner_angle_{i}", f"pt_{i}"])
        
        # Edge angle connections
        for i in range(n_sides):
            edges.append([f"dihedral_angle_{i}", f"line_segment_{i}"])
        
        # Exploratory data connections
        for i in range(n_sides):
            edges.append([f"internal_points_on_edge_{i}", f"line_segment_{i}"])
            edges.append([f"noisy_points_on_edge_{i}", f"line_segment_{i}"])
        
        graph_json = {
            "frame": {
                "name": frame_name
            },
            "nodes": nodes,
            "data_structure": data_structure,
            "edges": edges
        }
        if ref_frame_pose is not None:
            graph_json["ref_frame_pose"] = ref_frame_pose

        return graph_json

    @staticmethod
    def polygon_knw_to_graph(graph: nx.Graph, pk: PolygonKnowledge) -> None:
        """
        This method syncs all parameter values from PolygonKnowledge into graph node attributes.
        
        :param graph: NetworkX graph object to update (created by create_graph_from_json)
        :param pk: PolygonKnowledge instance with updated values
        """
        # Update corner nodes
        for i in range(pk.n_sides):
            corner_node_id = f"pt_{i}"
            
            if corner_node_id in graph.nodes:
                if pk.corners[i] is not None:
                    x_val, y_val = pk.corners[i]
                    graph.nodes[corner_node_id]['data'] = [x_val, y_val, 0.0]
                else:
                    graph.nodes[corner_node_id]['data'] = [None, None, None]
        
        # Update segment nodes
        for i in range(pk.n_sides):
            segment_node_id = f"line_segment_{i}"
            
            if segment_node_id in graph.nodes:
                # Update length and slope
                graph.nodes[segment_node_id]['length'] = pk.lengths[i]
                graph.nodes[segment_node_id]['slope_angle_deg'] = pk.slopes[i]
                
                # Update edge unit vector if present
                if pk.edge_unit_vectors[i] is not None:
                    ux, uy = pk.edge_unit_vectors[i]
                    graph.nodes[segment_node_id]['edge_unit_vector'] = [ux, uy, 0.0]
                else:
                    graph.nodes[segment_node_id]['edge_unit_vector'] = [None, None, None]
        
        # Update corner angle nodes
        for i in range(pk.n_sides):
            corner_angle_node_id = f"corner_angle_{i}"
            
            if corner_angle_node_id in graph.nodes:
                graph.nodes[corner_angle_node_id]['angle'] = pk.corner_angles[i]
                graph.nodes[corner_angle_node_id]['is_reflex'] = pk.is_reflexive_angle[i]
        
        # Update dihedral angle nodes
        for i in range(pk.n_sides):
            dihedral_node_id = f"dihedral_angle_{i}"
            
            if dihedral_node_id in graph.nodes:
                graph.nodes[dihedral_node_id]['angle'] = pk.dihedrals[i]


class ReasonerNode(ROSNode):
    """
    Reasoner Node which interprets selected action to motion specification, and updates knowledge based on motion results.
    
    This node manages:
    1. Prior knowlegde (rpk) - initial polygon knowledge
    2. Current knowledge (rck) - evolving knowledge from observations
    3. Action selection via core_algorithm.next_action
    4. Knowledge propagation via core_algorithm.propagate_parameters
    5. Graph synchronization for visualization
    """

    def __init__(self):
        """Initialize the Reasoner Node with ROS 2 components and knowledge structures."""
        super().__init__('reasoner_node')
        
        # ====== Load Configuration ======
        self.config = self._load_config()
        if self.config is None:
            self.get_logger().error("Failed to load configuration")
            return            
        
        # ====== ROS 2 communication ======
        self.client = ActionClient(self, MotionSpecification, self.config['frames']['motion_action'])
        self.publisher_marker = self.create_publisher(PoseStamped, self.config['frames']['marker_topic'], 10)
        self.publisher_corner = self.create_publisher(PoseStamped, self.config['frames']['corner_topic'], 10)
        self.publisher_plane = self.create_publisher(PoseStamped, self.config['frames']['plane_topic'], 10)
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.subscription = self.create_subscription(
            PoseStamped,
            self.config['frames']['ee_pose_topic'],
            self.ee_callback,
            10
        )
        
        # ====== Initialize flags ======
        self.unique_pattern_found_in_rpk = False
        self.unique_pattern_found_in_rck = False
        self.corner_coordinates_available_in_rpk = False
        self.rpk_rck_matching_idx_found = False
        self.rck_rearranged = False
        self.execution_start_time = datetime.now().astimezone()
        self.execution_end_time = None
        self.config_source_path = None
        self.initial_end_effector_pose = None
        self.final_rck_saved = False
        self.dof = None
        self.stop_execution = False
        self.last_motion_progress_log_time = 0.0
        
        # ====== Visualization Control ======
        # Set to False to suppress graph visualization (helps in headless/non-interactive environments)
        self.enable_graph_visualization = False
        
        # ====== Knowledge Structures ======
        self.experiment_id = self.config['experiment']['id']
        self.n_sides = self.config['experiment']['n_sides']
        self.validate_individual_match_across_fields = self.config.get(
            'knowledge_propagation', {}
        ).get('validate_individual_match_across_fields', False)
        
        ## Experiment 1: 4 sided polygon
        ## Experiment 2: complex polygon with n = 10 sides
        ## Experiment 3: shelf structure, with each surface having n = 4 sides
        # TODO: represent multiple surfaces with separate PolygonKnowledge instances or a hierarchical structure
        
        # Prior knowledge (rpk): remains constant, includes partial information about the ground truth
        self.rpk = PolygonKnowledge(n_sides=self.n_sides)
        # Initialize rpk (currently hard-coded based on experiment_id)
        self._initialize_prior_knowledge()
        self._propagate_knowledge(knowledge="rpk")
        self.unique_pattern_found_in_rpk = find_unique_pattern(self.rpk)
        if any(c is not None for c in self.rpk.corners):
            self.corner_coordinates_available_in_rpk = True
        
        # Current knowledge (rck): evolves as observations are made
        self.rck = PolygonKnowledge(n_sides=self.n_sides)
        fill_missing_parameters(self.rck, self.rpk, self.rpk_rck_matching_idx_found)
        self._propagate_knowledge(knowledge="rck")
        self.dof = find_dof(self.rck)
        self.initial_current_knowledge_sha256 = hashlib.sha256(
            json.dumps(self._json_safe(self._knowledge_snapshot()), sort_keys=True).encode("utf-8")
        ).hexdigest()
        self.step_count = 0
        self.completed_action_type_count = 0
        self.completed_action_type_instances = []
        # Structured execution trace.  This deliberately stores action and
        # primitive metadata plus compact knowledge deltas; high-rate sensor
        # streams are not duplicated in the RCK JSON file.
        self.current_action_log = None
        self.current_selection_metadata = None
        self.last_guard_diagnostic = None
        self.knowledge_stage_trace = []
        self.active_primitive_log = None
        self.bootstrap_action_instances = []
        self.failed_action_instances = []
        self.current_action_type_completion_recorded = False
        self.current_action_type_start_time = None
        self.current_action_type_start_distance_m = None
        self.current_action_type_start_sample_count = None
        self.ee_distance_tracking_frame_id = self.config['frames']['base_frame']
        self.ee_distance_total_m = 0.0
        self.ee_distance_sample_count = 0
        self.ee_distance_skipped_sample_count = 0
        self.ee_distance_skipped_frame_mismatch_count = 0
        self.ee_distance_last_position = None
        self.ee_distance_last_frame_id = None
        
        # ====== Graph Representation ======
        # Create initial belief state graph from rck using automatic JSON generation
        self.rck_json = PolygonKnowledgeToGraphConverter.polygon_knw_to_json(self.rck, frame_name="robot_frame_0")
        self.rpk_json = PolygonKnowledgeToGraphConverter.polygon_knw_to_json(self.rpk, frame_name="object_frame_0", polygon_id="polygon_0")
        
        self.rpk_graph = Graph.create_graph_from_json(self.rpk_json)             # Prior knowledge graph (static)
        self.rck_graph = Graph.create_simplified_graph_from_json(self.rck_json)  # Current knowledge graph (belief, updated after each action)
        
        # ====== State Variables ======
        self.state_of_execution = Util.StateOfExecution.IDLE
        self.ee_pose_frame_id = None
        self.current_position = [0.0, 0.0, 0.0]
        self.current_orientation = [0.0, 0.0, 0.0, 1.0]
        self.first_state_update_received = False
        self.current_ms_frame = self.config['frames']['base_frame']
        self.take_user_input_for_orientation = self.config['motion']['take_user_input_for_orientation']
        self.prev_action_instance = ActionInstance(action_type=None, edge_index=None)
        self.orientation_input = Util.OrientationInput(self.config['motion']['default_orientation'])
        self.offset_above_surface = self.config['motion']['offset_above_surface']
        self.offset_below_surface = self.config['motion']['offset_below_surface']
        self.diameter_of_end_effector = self.config['motion']['diameter_of_end_effector']
        self.offset_due_to_camera = self.config["motion"].get("offset_due_to_camera", 0.0)
        self.offset_from_edge_while_sliding_against_vertical_surface = self.config['motion'].get(
            'offset_from_edge_while_sliding_against_vertical_surface',
            0.0
        )
        self.touch_velocity = self.config['motion']['touch_velocity']
        self.slide_velocity = self.config['motion']['slide_velocity']
        self.slide_offset_from_edge = self.config['motion']['slide_offset_from_edge']
        self.offset_from_edge_while_moving_from_outside_to_edge = self.config['motion'].get(
            'offset_from_edge_while_moving_from_outside_to_edge',
            self.slide_offset_from_edge
        )
        self.force_against_surface = self.config['motion']['force_against_surface']
        self.angle_increment_radians = self.config['motion']['angle_increment_radians']
        self.force_along_edge_to_find_slope = self.config['motion']['force_along_edge_to_find_slope']
        self.length_action_list = 0
        self.next_action_idx = 0
        self.desired_orientation = None
        self.sliding_motion_detected = False
        self.dir_of_sliding_motion_2d = None
        self.distance_threshold_for_motion_detection = self.config['motion']['distance_threshold_for_motion_detection']
        self.sliding_variables_initialized = False
        self.reorient_while_sliding_against_edge = False
        self.contact_established_with_vertical_surface = False
        self.action_name_str = None
        self.current_marker_frame_name = None
        self.current_edge_of_interest_origin = None
        self.current_edge_of_interest_orientation = None
        self.current_marker_id = -1
        self.marker_id_for_edges = [None for _ in range(self.n_sides)] # used only for sliding along edge. Thus, even if edge uv is known, this might not be populated until prior to sliding after initial motion along edge.
        self.plane_origin_position = None
        self.plane_orientation = None
        self.exploration_complete = False
        self.current_ref_edge_index = None
        self.get_new_action_list_bool = True
        self.established_first_contact = True
        self.plane_slope_estimated = False # trigger default motion to estimate the slope of plane
        self.sliding_against_edge_sm_active = False
        self.prev_action_spec = None
        self.current_action_spec = None
        self.last_direction_of_motion = None
        self.last_direction_of_force_while_sliding_against_edge = None
        self.collect_points_on_edge_bool = False
        self.debug_log = False
        
        # Motion tracking
        self.motion_indices_to_collect_points = []
        self.collected_points = []
        self.points_on_plane = []
        self.points_on_edge = []
        self.noisy_points_on_edge = [[] for _ in range(self.n_sides)]
        self.current_action_type = None
        
        # ====== Logging Setup ======
        self.logs_dir = os.path.join(os.path.dirname(__file__), 'rck_logs')
        os.makedirs(self.logs_dir, exist_ok=True)
        self.rck_json_path = os.path.join(self.logs_dir, 'rck_knowledge.json')
        self.rck_history_dir = os.path.join(self.logs_dir, 'rck_history')
        os.makedirs(self.rck_history_dir, exist_ok=True)
        logging_config = self.config.get('logging', {})
        self.rck_save_mode = logging_config.get('rck_save_mode', 'final_only')
        if self.rck_save_mode not in {'final_only', 'on_update'}:
            self.get_logger().warn(
                f"Unknown rck_save_mode '{self.rck_save_mode}'. Using 'final_only'."
            )
            self.rck_save_mode = 'final_only'
        self.rck_snapshot_count = 0
        
        # ====== Initialization ======
        
        # Sync rck with graph
        self._sync_knowledge_to_graph()

        # Start main loop
        self.create_timer(0.001, self.main_loop)
        self.get_logger().info("Reasoner Node initialized with PolygonKnowledge framework")
    
    def _load_config(self):
        """
        Load configuration from YAML file.
        
        Returns:
            Dictionary containing configuration parameters
        """
        # Try multiple locations for the config file
        current_file = os.path.abspath(__file__)
        config_paths = [
            # Installation path (when package is installed)
            os.path.join(os.path.dirname(current_file), 'config', 'reasoner_config.yaml'),
            # Source path option 1 (during development from source)
            os.path.normpath(os.path.join(os.path.dirname(current_file), '..', '..', '..', 'config', 'reasoner_config.yaml')),
            # Source path option 2 (alternative structure)
            os.path.normpath(os.path.join(os.path.dirname(current_file), '..', 'config', 'reasoner_config.yaml')),
        ]
        
        for config_path in config_paths:
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                    self.config_source_path = os.path.abspath(config_path)
                    self.get_logger().info(f"Loaded configuration from {config_path}")
                    return config
                except Exception as e:
                    self.get_logger().warn(f"Error loading config file from {config_path}: {e}")
        
        self.get_logger().warn(f"Config file not found in any of: {[os.path.abspath(p) for p in config_paths]}")
        return None

    def _initialize_prior_knowledge(self):
        """
        Initialize prior knowledge (rpk) from polygon type.
        This represents what we expect about the polygon geometry.
        """
        if self.experiment_id in range(1,10):
            pass
        elif self.experiment_id == 11:
            pass
        elif self.experiment_id == 12:
            pass
        else:
            self.get_logger().warn(f"Unknown experiment ID {self.experiment_id} - using default prior knowledge")
    
    def _sync_knowledge_to_graph(self):
        """
        Synchronize current PolygonKnowledge (rck) to NetworkX graph.
        """
        PolygonKnowledgeToGraphConverter.polygon_knw_to_graph(
            self.rck_graph, 
            self.rck
        ) # this updates the values in self.rck_graph.data_structure based on current rck values
        
        # Only render visualization if enabled (to avoid issues in headless environments)
        if self.enable_graph_visualization:
            Graph.render_graph_visualization(self.rck_graph)
        # since rpk is static, it is not updated here
    
    def _save_rck_to_json(self, snapshot_label: str, mark_final: bool = False):
        """
        Save current knowledge (rck) to the latest JSON path and timestamped history.
        """
        if mark_final and self.final_rck_saved:
            return

        try:
            snapshot_time = datetime.now().astimezone()
            if mark_final:
                self.execution_end_time = snapshot_time

            # Generate JSON from current rck
            rck_json = PolygonKnowledgeToGraphConverter.polygon_knw_to_json(
                self.rck, 
                frame_name="robot_frame_0",
                polygon_id="polygon_0",
                ref_frame_pose=self._ref_frame_pose_to_json(),
                noisy_points_on_edge=self.noisy_points_on_edge
            )

            duration_seconds = (
                snapshot_time - self.execution_start_time
            ).total_seconds()
            rck_json["execution"] = {
                "start_time": self.execution_start_time.isoformat(),
                "snapshot_time": snapshot_time.isoformat(),
                "end_time": self.execution_end_time.isoformat() if self.execution_end_time else None,
                "duration_seconds": duration_seconds,
                "step_count": self.step_count,
                "dof": self.dof,
                "current_action_type": self.current_action_type.name if self.current_action_type else None,
                "current_ref_edge_index": self.current_ref_edge_index,
                "snapshot_label": snapshot_label,
                "rck_save_mode": self.rck_save_mode,
                "logging_schema_version": 2,
                "reproducibility": self._execution_reproducibility_metadata(),
                "motion_primitives": {
                    "completed_action_type_count": self.completed_action_type_count,
                    "action_type_instances": self.completed_action_type_instances,
                    "bootstrap_instances": self.bootstrap_action_instances,
                    "failed_instances": self.failed_action_instances,
                },
                "end_effector_distance": {
                    "total_distance_m": self.ee_distance_total_m,
                    "tracking_frame_id": self.ee_distance_tracking_frame_id,
                    "sample_count": self.ee_distance_sample_count,
                    "skipped_sample_count": self.ee_distance_skipped_sample_count,
                    "skipped_frame_mismatch_count": self.ee_distance_skipped_frame_mismatch_count,
                }
            }

            if mark_final:
                timestamp = snapshot_time.strftime("%Y%m%d_%H%M%S")
                history_filename = f'rck_knowledge_{timestamp}.json'
            else:
                self.rck_snapshot_count += 1
                timestamp = snapshot_time.strftime("%Y%m%d_%H%M%S_%f")
                history_filename = (
                    f'rck_knowledge_{self.rck_snapshot_count:04d}_{snapshot_label}_{timestamp}.json'
                )
            history_file = os.path.join(self.rck_history_dir, history_filename)

            with open(self.rck_json_path, 'w') as f:
                json.dump(rck_json, f, indent=4)
            with open(history_file, 'w') as f:
                json.dump(rck_json, f, indent=4)

            if mark_final:
                self.final_rck_saved = True
            self.get_logger().info(f"Saved rck snapshot to {history_file}")
        except Exception as e:
            self.get_logger().warn(f"Error saving rck to JSON: {e}")

    def _save_final_rck_to_json(self):
        """
        Save final current knowledge (rck) once at the end of execution.
        """
        self._save_rck_to_json(snapshot_label='final', mark_final=True)

    def _save_rck_snapshot_if_configured(self, snapshot_label: str):
        """
        Save an intermediate RCK snapshot when configured to log after updates.
        """
        if self.rck_save_mode == 'on_update':
            self._save_rck_to_json(snapshot_label=snapshot_label)

    def _start_action_type_instance(self):
        """
        Mark the cumulative EE distance at the start of a high-level ActionType.
        """
        self.current_action_type_completion_recorded = False
        self.current_action_type_start_time = datetime.now().astimezone()
        self.current_action_type_start_distance_m = self.ee_distance_total_m
        self.current_action_type_start_sample_count = self.ee_distance_sample_count
        self.current_action_log = {
            "instance_index": len(self.completed_action_type_instances) + 1,
            "action_type": self.current_action_type.name if self.current_action_type else None,
            "ref_edge_index_before_reindex": (
                int(self.current_ref_edge_index)
                if self.current_ref_edge_index is not None else None
            ),
            "ref_edge_index": (
                int(self.current_ref_edge_index)
                if self.current_ref_edge_index is not None else None
            ),
            "ref_edge_index_after_reindex": None,
            "action_spec": self._action_spec_to_json(self.current_action_spec),
            "selection_provenance": self._json_safe(self.current_selection_metadata),
            "rck_rearranged_at_start": bool(self.rck_rearranged),
            "dof_before": self.dof,
            "knowledge_before": self._knowledge_snapshot(),
            "primitives": [],
            "fit_diagnostics": [],
            "knowledge_stage_trace": [],
            "knowledge_update_provenance": [],
            "knowledge_updates_before_reindex": [],
            "knowledge_updates_after_reindex": [],
            "reindex": None,
        }

    @staticmethod
    def _json_safe(value):
        """Convert numpy/enumeration values to values accepted by ``json``."""
        if isinstance(value, np.ndarray):
            return [ReasonerNode._json_safe(v) for v in value.tolist()]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, Enum):
            return value.name
        if isinstance(value, dict):
            return {str(k): ReasonerNode._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [ReasonerNode._json_safe(v) for v in value]
        return value

    def _knowledge_snapshot_for(self, knowledge):
        """Return a compact, index-preserving snapshot of polygon knowledge."""
        fields = (
            "slopes", "lengths", "edge_unit_vectors", "corners",
            "is_reflexive_angle", "corner_angles", "dihedrals",
            "internal_points_on_edge",
        )
        return {
            "n_sides": int(knowledge.n_sides),
            **{field: self._json_safe(copy.deepcopy(getattr(knowledge, field)))
               for field in fields},
        }

    def _knowledge_snapshot(self):
        """Return a compact, index-preserving snapshot of the current RCK."""
        return self._knowledge_snapshot_for(self.rck)

    @staticmethod
    def _knowledge_diff(before, after):
        """Describe changed indexed attributes without storing duplicate states."""
        changes = []
        if not before or not after:
            return changes
        for field in before:
            if field == "n_sides" or field not in after:
                continue
            old_values = before[field]
            new_values = after[field]
            if not isinstance(old_values, list) or not isinstance(new_values, list):
                if old_values != new_values:
                    changes.append({"field": field, "before": old_values, "after": new_values})
                continue
            for index, (old, new) in enumerate(zip(old_values, new_values)):
                if old != new:
                    changes.append({
                        "field": field,
                        "index": index,
                        "before": old,
                        "after": new,
                    })
        return changes

    @staticmethod
    def _action_spec_to_json(action_spec):
        if action_spec is None:
            return None
        return {
            "direction": getattr(action_spec.direction, "name", str(action_spec.direction)),
            "mode": getattr(action_spec.mode, "name", str(action_spec.mode)),
            "stop": getattr(action_spec.stop, "name", str(action_spec.stop)),
        }

    def _ensure_action_log(self):
        """Create a trace record for bootstrap actions with no ActionType yet."""
        if self.current_action_log is None:
            self._start_action_type_instance()

    def _begin_primitive_log(self, motion_specification, ms_json):
        self._ensure_action_log()
        arm_name = ms_json.get("arm_name")
        arm_spec = ms_json.get(arm_name, {}) if arm_name else {}
        action_name = arm_spec.get("action_name")
        if isinstance(action_name, str):
            primitive_family = (
                "slide" if action_name.startswith("slide") or action_name.startswith("find_edge")
                else "touch" if action_name.startswith("touch")
                else "move" if action_name.startswith("move")
                else "yaw" if action_name.startswith("yaw")
                else action_name
            )
        else:
            primitive_family = None
        primitive = {
            "primitive_index": len(self.current_action_log["primitives"]) + 1,
            "family": primitive_family,
            "action_name": action_name,
            "reference_edge_before_reindex": self.current_action_log.get(
                "ref_edge_index_before_reindex"
            ),
            "reference_edge": (
                int(self.current_ref_edge_index)
                if self.current_ref_edge_index is not None else None
            ),
            "frame_name": arm_spec.get("frame_name"),
            "start_time": datetime.now().astimezone().isoformat(),
            "start_sample_count": self.ee_distance_sample_count,
            "knowledge_before": self._knowledge_snapshot(),
            "knowledge_updates_before_reindex": [],
            "knowledge_updates_after_reindex": [],
            "reference_edge_after_reindex": None,
            "rck_rearranged_at_start": bool(self.rck_rearranged),
            "terminal_outcome": None,
            "guard_observation": None,
            "status": "running",
        }
        self.current_action_log["primitives"].append(primitive)
        self.active_primitive_log = primitive

    def _finish_primitive_log(self, status, result=None, error=None):
        primitive = self.active_primitive_log
        if primitive is None:
            return
        primitive["end_time"] = datetime.now().astimezone().isoformat()
        primitive["end_sample_count"] = self.ee_distance_sample_count
        primitive["status"] = status
        primitive["duration_seconds"] = (
            datetime.fromisoformat(primitive["end_time"]) -
            datetime.fromisoformat(primitive["start_time"])
        ).total_seconds()
        primitive["distance_start_m"] = self.current_action_type_start_distance_m
        primitive["distance_end_m"] = self.ee_distance_total_m
        if self.current_action_type_start_distance_m is not None:
            primitive["distance_from_action_start_m"] = max(
                0.0, self.ee_distance_total_m - self.current_action_type_start_distance_m
            )
        if result is not None:
            primitive["disjunction_indices"] = self._json_safe(
                list(getattr(result, "disjunction_indices", []) or [])
            )
            primitive["result_action_name"] = getattr(result, "ms_action_name", None)
        if self.last_guard_diagnostic is not None:
            primitive["terminal_outcome"] = self._json_safe(self.last_guard_diagnostic)
            primitive["guard_observation"] = self._json_safe(self.last_guard_diagnostic)
        if self.current_action_log is not None:
            primitive["fit_diagnostics"] = self._json_safe(
                self.current_action_log.get("fit_diagnostics", [])
            )
        if error is not None:
            primitive["error"] = str(error)
        primitive["knowledge_after"] = self._knowledge_snapshot()
        primitive["rck_rearranged_at_end"] = bool(self.rck_rearranged)
        self.active_primitive_log = None

        # Direct evidence is committed inside ``on_action_succeeded`` before
        # this callback returns. Capture it after the primitive has completed.
        if self.current_action_log is not None:
            if self.current_action_type_completion_recorded:
                if "knowledge_before_reindex" not in self.current_action_log:
                    self._capture_action_updates("before_reindex")
            elif (
                self.current_action_type is None
                and self.next_action_idx >= self.length_action_list
                and self.current_action_log not in self.bootstrap_action_instances
            ):
                self.current_action_log.update({
                    "action_type": "BOOTSTRAP",
                    "end_time": primitive.get("end_time"),
                    "status": status,
                })
                self.bootstrap_action_instances.append(self.current_action_log)
            elif (
                status != "succeeded"
                and self.current_action_log not in self.completed_action_type_instances
                and self.current_action_log not in self.failed_action_instances
            ):
                self.current_action_log.update({
                    "end_time": primitive.get("end_time"),
                    "status": status,
                })
                self.failed_action_instances.append(self.current_action_log)

    @staticmethod
    def _terminal_outcome_metadata(result):
        """Map controller disjunction IDs to compact semantic guard labels."""
        if result is None:
            return None
        action_name = getattr(result, "ms_action_name", None)
        disjunctions = [int(value) for value in (getattr(result, "disjunction_indices", []) or [])]
        labels = {
            "slide_against_vertical_surface_until_corner": {
                1: "convex_90_transition",
                2: "convex_270_transition",
                3: "reflex_transition",
            },
            "slide_against_edge_until_corner": {
                1: "ambiguous_or_reflex_transition",
                2: "non_reflex_transition",
            },
            "slide_until_edge": {
                1: "adjacent_face_contact",
                2: "outer_edge_transition",
                3: "bounded_no_edge",
            },
            "find_edge_by_sliding": {
                1: "adjacent_face_contact",
                2: "outer_edge_transition",
                3: "bounded_no_edge",
            },
        }.get(action_name, {})
        return {
            "action_name": action_name,
            "disjunction_indices": disjunctions,
            "semantic_labels": [labels.get(index, f"disjunction_{index}") for index in disjunctions],
        }

    def _record_knowledge_stage(self, stage):
        """Store compact provenance checkpoints without raw sensor streams."""
        if self.current_action_log is None:
            return
        entry = {
            "stage": stage,
            "time": datetime.now().astimezone().isoformat(),
            "knowledge": self._knowledge_snapshot(),
            "dof": find_dof(self.rck),
        }
        self.current_action_log.setdefault("knowledge_stage_trace", []).append(entry)
        self.knowledge_stage_trace.append(entry)

    def _record_fit_diagnostics(self, fit_type, points, *, direction=None, origin=None, orientation=None):
        """Record fit quality statistics while discarding the raw trajectory."""
        if self.current_action_log is None or points is None:
            return
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or len(pts) == 0:
            return
        record = {"fit_type": fit_type, "sample_count": int(len(pts))}
        if direction is not None and len(pts) >= 2:
            xy = pts[:, :2]
            d = np.asarray(direction, dtype=float)[:2]
            norm = float(np.linalg.norm(d))
            if norm > 1e-12:
                d /= norm
                center = np.mean(xy, axis=0)
                residuals = np.abs((xy - center)[:, 0] * d[1] - (xy - center)[:, 1] * d[0])
                record.update({
                    "inlier_count_estimate": int(np.count_nonzero(residuals <= 0.0015)),
                    "residual_rms_m": float(np.sqrt(np.mean(residuals ** 2))),
                    "residual_mean_m": float(np.mean(residuals)),
                    "residual_max_m": float(np.max(residuals)),
                })
        if origin is not None and orientation is not None and len(pts) >= 3:
            normal = R.from_quat(orientation).as_matrix()[:, 2]
            distances = np.abs((pts[:, :3] - np.asarray(origin, dtype=float)) @ normal)
            record.update({
                "inlier_count_estimate": int(np.count_nonzero(distances <= 0.005)),
                "residual_rms_m": float(np.sqrt(np.mean(distances ** 2))),
                "residual_mean_m": float(np.mean(distances)),
                "residual_max_m": float(np.max(distances)),
            })
        self.current_action_log.setdefault("fit_diagnostics", []).append(record)

    def _capture_action_updates(self, phase):
        """Capture attribute changes and index state at a reasoning boundary."""
        if self.current_action_log is None:
            return
        current = self._knowledge_snapshot()
        self.current_action_log["knowledge_after"] = current
        self.current_action_log["dof_after"] = find_dof(self.rck)
        self.current_action_log.setdefault("knowledge_update_provenance", []).append({
            "phase": phase,
            "source": (
                "action_conditioned_terminal_evidence"
                if phase == "before_reindex"
                else "geometric_propagation_and_prior_alignment"
            ),
            "terminal_outcome": self._json_safe(self.last_guard_diagnostic),
            "update_count": 0,
        })
        if phase == "before_reindex":
            updates = self._knowledge_diff(
                self.current_action_log["knowledge_before"], current
            )
            self.current_action_log["knowledge_before_reindex"] = current
            self.current_action_log["knowledge_updates_before_reindex"] = updates
            self.current_action_log["knowledge_update_provenance"][-1]["update_count"] = len(updates)
            if self.current_action_log.get("primitives"):
                self.current_action_log["primitives"][-1]["knowledge_updates_before_reindex"] = updates
        elif phase == "after_reindex":
            prior = self.current_action_log.get("knowledge_before_reindex", self.current_action_log["knowledge_before"])
            updates = self._knowledge_diff(prior, current)
            self.current_action_log["knowledge_after_reindex"] = current
            self.current_action_log["knowledge_updates_after_reindex"] = updates
            self.current_action_log["knowledge_update_provenance"][-1]["update_count"] = len(updates)
            if self.current_action_log.get("primitives"):
                self.current_action_log["primitives"][-1]["knowledge_updates_after_reindex"] = updates
            self.current_action_log["ref_edge_index_after_reindex"] = (
                int(self.current_ref_edge_index)
                if self.current_ref_edge_index is not None else None
            )
            for primitive in self.current_action_log.get("primitives", []):
                primitive["reference_edge_after_reindex"] = (
                    int(self.current_ref_edge_index)
                    if self.current_ref_edge_index is not None else None
                )
                primitive["rck_rearranged_after_action"] = True

    def _execution_reproducibility_metadata(self):
        config_bytes = json.dumps(self.config, sort_keys=True, default=str).encode("utf-8")
        prior_bytes = json.dumps(self._json_safe(self._knowledge_snapshot_for(self.rpk)), sort_keys=True).encode("utf-8")
        try:
            git_commit = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=os.path.dirname(__file__), capture_output=True, text=True,
                timeout=1, check=False,
            ).stdout.strip() or None
        except Exception:
            git_commit = None
        return {
            "run_id": self.execution_start_time.strftime("%Y%m%dT%H%M%S%f%z"),
            "software_git_commit": git_commit,
            "source_file": os.path.abspath(__file__),
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "config_source_path": self.config_source_path,
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "prior_knowledge_sha256": hashlib.sha256(prior_bytes).hexdigest(),
            "initial_current_knowledge_sha256": self.initial_current_knowledge_sha256,
            "random_seed": self.config.get("experiment", {}).get("random_seed"),
            "initial_end_effector_pose": self.initial_end_effector_pose,
            "hardware": self.config.get("hardware", {}),
            "experiment_id": self.experiment_id,
            "n_sides": self.n_sides,
            "frames": self.config.get("frames", {}),
            "motion_parameters": self.config.get("motion", {}),
            "knowledge_propagation": self.config.get("knowledge_propagation", {}),
        }

    def _record_completed_action_type(self):
        """
        Record one completed high-level ActionType after its final motion goal succeeds.
        """
        if self.current_action_type is None:
            return

        if self.current_action_type_completion_recorded:
            return

        action_type_name = self.current_action_type.name
        end_time = datetime.now().astimezone()
        duration_seconds = (
            (end_time - self.current_action_type_start_time).total_seconds()
            if self.current_action_type_start_time is not None
            else None
        )
        start_distance_m = (
            self.current_action_type_start_distance_m
            if self.current_action_type_start_distance_m is not None
            else self.ee_distance_total_m
        )
        distance_traversed_m = max(0.0, self.ee_distance_total_m - start_distance_m)
        self.completed_action_type_count += 1
        action_record = self.current_action_log or {
            "instance_index": self.completed_action_type_count,
            "action_type": action_type_name,
            "primitives": [],
            "knowledge_before": self._knowledge_snapshot(),
        }
        action_record.update({
            "instance_index": self.completed_action_type_count,
            "action_type": action_type_name,
            "ref_edge_index": (
                int(self.current_ref_edge_index)
                if self.current_ref_edge_index is not None else None
            ),
            "start_time": (
                self.current_action_type_start_time.isoformat()
                if self.current_action_type_start_time is not None
                else None
            ),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration_seconds,
            "distance_traversed_m": distance_traversed_m,
            "distance_start_m": start_distance_m,
            "distance_end_m": self.ee_distance_total_m,
            "tracking_frame_id": self.ee_distance_tracking_frame_id,
            "start_sample_count": self.current_action_type_start_sample_count,
            "end_sample_count": self.ee_distance_sample_count,
        })
        if self.current_action_log is None:
            self.current_action_log = action_record
        if action_record not in self.completed_action_type_instances:
            self.completed_action_type_instances.append(action_record)
        self.current_action_type_completion_recorded = True
        self.get_logger().info(
            f"Completed action type {action_type_name}; distance traversed: "
            f"{distance_traversed_m:.4f} m; total completed action types: "
            f"{self.completed_action_type_count}"
        )

    def _ref_frame_pose_to_json(self):
        """
        Serialize the estimated plane/reference frame pose into JSON-safe values.
        """
        if self.plane_origin_position is None or self.plane_orientation is None:
            return None

        position = [float(value) for value in self.plane_origin_position]
        quaternion = [float(value) for value in self.plane_orientation]

        return {
            "description": "Estimated plane/reference frame pose in eddie_base_link",
            "frame_id": "eddie_base_link",
            "child_frame_id": "robot_frame_0",
            "position": {
                "x": position[0],
                "y": position[1],
                "z": position[2],
                "array": position
            },
            "orientation": {
                "quaternion": {
                    "x": quaternion[0],
                    "y": quaternion[1],
                    "z": quaternion[2],
                    "w": quaternion[3],
                    "array": quaternion
                }
            },
            "source": "slide_to_explore_plane"
        }
        
    def handle_sliding_against_unknown_surface(self):
        """
        Sliding requires dynamic action updation:
        1. apply force in the opposite direction in which end-effector detected contact loss.
        2. incrementally increase angle of force until a motion is detected crossing a threshold.
        3. Now specify edge tracing by applying perpendicular force and velocity in the direction of edge. 
        4. Once it traverses minimum distace, recalculate and update the edge unit vector
        5. Use this direction to specify sliding motion against edge
        """
        
        # initialization
        if not self.sliding_variables_initialized:
            self.desired_orientation = copy.deepcopy(self.current_orientation)
            self.points_on_edge = [(self.current_position[0], self.current_position[1])]
            self.sliding_motion_detected = False
            self.dir_of_sliding_motion_2d = None
            self.sliding_variables_initialized = True
            self.collect_points_on_edge_bool = True
            self.reorient_while_sliding_against_edge = False
            self.contact_established_with_vertical_surface = False
        
        # distance traversed check
        if not self.sliding_motion_detected:
            if len(self.points_on_edge) > 1:
                x0, y0 = self.points_on_edge[0][:2]
                x1, y1 = self.points_on_edge[-1][:2]
                distance_traversed_2d = math.hypot(x1 - x0, y1 - y0)
                if self.debug_log: print(f"Distance traversed while sliding: {distance_traversed_2d}")
            else:
                distance_traversed_2d = 0.0
            
            # on detection of first sliding motion
            if distance_traversed_2d > self.distance_threshold_for_motion_detection:
                self.sliding_motion_detected = True
                self.dir_of_sliding_motion_2d = Util.unit_vector_from_points_2d(points=self.points_on_edge)
                if self.debug_log: print(f"Direction of sliding motion (2D): {self.dir_of_sliding_motion_2d}")

                angle_with_direction_of_force = Util.get_ccw_angle(self.dir_of_sliding_motion_2d, self.last_direction_of_motion[0:2])
                if self.debug_log: print(f"Last direction of motion/force (2D): {self.last_direction_of_motion[0:2]}")
                if self.debug_log: print(f"Angle between direction of sliding motion and last direction of motion/force: {math.degrees(angle_with_direction_of_force)} degrees")
                if angle_with_direction_of_force < math.pi/2:
                    if self.current_action_spec.mode == Mode.AGAINST_EDGE:
                        self.get_logger().info("Sliding motion detected in the CCK direction")
                    elif self.current_action_spec.mode == Mode.AGAINST_VERTICAL:
                        self.get_logger().info("Sliding motion detected in the CK direction. Flipping direction to get correct edge unit vector")
                        self.dir_of_sliding_motion_2d = [-self.dir_of_sliding_motion_2d[0], -self.dir_of_sliding_motion_2d[1]] # invert direction to match CCK direction
                elif angle_with_direction_of_force > 3*math.pi/2:
                    if self.current_action_spec.mode == Mode.AGAINST_EDGE:
                        self.get_logger().info("Sliding motion detected in the CK direction. Flipping direction to get correct edge unit vector")
                        self.dir_of_sliding_motion_2d = [-self.dir_of_sliding_motion_2d[0], -self.dir_of_sliding_motion_2d[1]] # invert direction to match CCK direction
                    elif self.current_action_spec.mode == Mode.AGAINST_VERTICAL:
                        self.get_logger().info("Sliding motion detected in the CCK direction")
                else:
                    self.get_logger().warn("Sliding motion detected but direction is ambiguous. There could be a contact loss. Terminating execution")
                    self.stop_execution = True
                    return
                
                print(f"Final direction of sliding motion (2D): {self.dir_of_sliding_motion_2d}")
                
                # get orientation of new marker frame, where x axis is along the direction of sliding motion and z axis is same as base_link frame (assuming planar surface)
                edge_orientation = Util.get_quat_of_align_frame_to_edge(self.plane_orientation, self.dir_of_sliding_motion_2d)
                self.create_and_publish_marker_pose(
                    position=self.points_on_edge[-1], 
                    orientation=edge_orientation, 
                    marker_type="marker", 
                    frame="marker_frame_0")
                self.current_edge_of_interest_origin = self.points_on_edge[-1]
                self.current_edge_of_interest_orientation = edge_orientation
                
                if self.marker_id_for_edges[self.current_ref_edge_index] is None:
                    self.marker_id_for_edges[self.current_ref_edge_index] = self.current_marker_id
                
                self.points_on_edge = [] # reset for next round of accumulation if needed
                # Note: this is with an assumption that the direction of force is against the edge and contact is always established, which is enabled by previous motion
        
        # continue probing if first sliding is still not detected
        if not self.sliding_motion_detected:
            if self.current_action_spec.mode == Mode.AGAINST_EDGE:
                self.get_logger().info("Handling sliding against edge to estimate edge unit vector")
                
                if self.current_action_spec.direction == Direction.CCK:
                    self.get_logger().info("Handling sliding in counter-clockwise direction")
                    angle_increment_direction = -1 # in the frame of base_link
                else: # Direction.CK:
                    self.get_logger().info("Handling sliding in clockwise direction")
                    angle_increment_direction = 1  # in the frame of base_link

            elif self.current_action_spec.mode == Mode.AGAINST_VERTICAL:
                self.get_logger().info("Handling sliding against vertical surface to estimate slope of plane")
                if self.current_action_spec.direction == Direction.CCK:
                    self.get_logger().info("Handling sliding in counter-clockwise direction")
                    angle_increment_direction = 1 # in the frame of base_link
                else: # Direction.CK:
                    self.get_logger().info("Handling sliding in clockwise direction")
                    angle_increment_direction = -1  # in the frame of base_link
            else:
                raise ValueError(f"Unhandled mode: {self.current_action_spec.mode}")
            
            if self.current_action_spec.mode == Mode.AGAINST_VERTICAL:
                force_in_z_direction = -self.force_against_surface
                position_in_z_direction = None
            else:
                force_in_z_direction = None
                position_in_z_direction = -self.offset_below_surface

            # Rotate last_direction_of_motion by angle_increment_radians
            dx, dy, _ = self.last_direction_of_motion
            
            angle = self.angle_increment_radians * angle_increment_direction
            cos_angle = math.cos(angle)
            sin_angle = math.sin(angle)

            new_dx = dx * cos_angle - dy * sin_angle
            new_dy = dx * sin_angle + dy * cos_angle
            
            norm = math.hypot(new_dx, new_dy)
            if norm > 1e-8:
                new_dx /= norm
                new_dy /= norm
            
            self.last_direction_of_motion = [new_dx, new_dy, 0.0]
            
            fx = self.force_along_edge_to_find_slope * new_dx
            fy = self.force_along_edge_to_find_slope * new_dy

            force_vector = [fx, fy, force_in_z_direction]
            self.action_name_str = "find_edge_by_sliding"
            
            ms_to_execute = Util.make_action_goal_slide(
                position=[None, None, position_in_z_direction],
                force=force_vector,
                orientation=self.desired_orientation,
                action_name=self.action_name_str,
                frame_name="marker_frame_0",
                time=1.0
            )
            self.send_goal(ms_to_execute)

        if self.sliding_motion_detected:
            self.get_logger().info("Sliding motion detected, determining final sliding motion specification to trace along the edge")
            
            if self.current_action_spec.stop == Stop.UNTIL_CORNER:
                if self.current_action_spec.mode == Mode.AGAINST_EDGE:
                    self.action_name_str = "slide_against_edge_until_corner"
                elif self.current_action_spec.mode == Mode.AGAINST_VERTICAL:
                    self.action_name_str = "slide_against_vertical_surface_until_corner"
            elif self.current_action_spec.stop == Stop.VECTOR_ONLY:
                # only when it is vector only, the time parameter in motion spec is used
                self.action_name_str = "slide_against_surface_vector_only"

            if self.marker_id_for_edges[self.current_ref_edge_index] is None:
                self.get_logger().warn(f"Marker ID for edge {self.current_ref_edge_index} is not set.")
            self.current_marker_frame_name = f"marker_frame_{self.marker_id_for_edges[self.current_ref_edge_index]}"
            if self.current_action_spec.mode == Mode.AGAINST_EDGE:                
                # rotate dir of velocity by 90 degrees to get foce vector for sliding against edge
                dir_of_force = (-self.dir_of_sliding_motion_2d[1], self.dir_of_sliding_motion_2d[0]) # 90 degree rotation in the frame of base_link
                
                if self.current_action_spec.direction == Direction.CCK:
                    self.desired_orientation, desired_yaw = Util.resolve_orientation(dir_of_motion=[1.0, 0.0], orientation_input=Util.OrientationInput.RIGHT_OF_DIR_MOTION)
                    
                    if self.reorient_while_sliding_against_edge == False:
                        self.collect_points_on_edge_bool = False
                        self.reorient_while_sliding_against_edge = True
                        # attain desired yaw
                        ms_to_execute = Util.make_action_goal_yaw(
                            position=[None, None, -self.offset_below_surface],
                            yaw=desired_yaw,
                            frame_name=self.current_marker_frame_name
                        )
                    else:
                        self.collect_points_on_edge_bool = True
                        ms_to_execute = Util.make_action_goal_slide(
                            position=[None, None, -self.offset_below_surface],
                            velocity=[self.slide_velocity, None, None],
                            force=[None, self.force_against_surface, None],
                            orientation=self.desired_orientation,
                            action_name=self.action_name_str,
                            frame_name=self.current_marker_frame_name,
                            time=3.0
                        )
                        self.last_direction_of_motion = [self.dir_of_sliding_motion_2d[0], self.dir_of_sliding_motion_2d[1], 0.0]
                        self.last_direction_of_force_while_sliding_against_edge = [dir_of_force[0], dir_of_force[1], 0.0]
                elif self.current_action_spec.direction == Direction.CK:
                    self.desired_orientation, desired_yaw = Util.resolve_orientation(dir_of_motion=[1.0, 0.0], orientation_input=Util.OrientationInput.LEFT_OF_DIR_MOTION)
                    
                    if self.reorient_while_sliding_against_edge == False:
                        self.collect_points_on_edge_bool = False
                        self.reorient_while_sliding_against_edge = True
                        ms_to_execute = Util.make_action_goal_yaw(
                            position=[None, None, -self.offset_below_surface],
                            yaw=desired_yaw,
                            frame_name=self.current_marker_frame_name
                        )
                    else:
                        self.collect_points_on_edge_bool = True
                        ms_to_execute = Util.make_action_goal_slide(
                            position=[None, None, -self.offset_below_surface],
                            velocity=[self.slide_velocity, None, None],
                            force=[None, self.force_against_surface, None],
                            orientation=self.desired_orientation,
                            action_name=self.action_name_str,
                            frame_name=self.current_marker_frame_name,
                            time=3.0
                        )
                        self.last_direction_of_motion = [-self.dir_of_sliding_motion_2d[0], -self.dir_of_sliding_motion_2d[1], 0.0]
                        self.last_direction_of_force_while_sliding_against_edge = [dir_of_force[0], dir_of_force[1], 0.0]
                self.send_goal(ms_to_execute)
            
            elif self.current_action_spec.mode == Mode.AGAINST_VERTICAL:
                dir_of_force = (self.dir_of_sliding_motion_2d[1], -self.dir_of_sliding_motion_2d[0]) # -90 degree rotation in the frame of base_link
                
                if self.current_action_spec.direction == Direction.CCK:
                    self.desired_orientation, desired_yaw = Util.resolve_orientation(dir_of_motion=[1.0, 0.0], orientation_input=Util.OrientationInput.LEFT_OF_DIR_MOTION)
                    
                    if self.reorient_while_sliding_against_edge == False:
                        self.collect_points_on_edge_bool = False
                        self.reorient_while_sliding_against_edge = True
                        ms_to_execute = Util.make_action_goal_yaw(
                            position=[None, None, self.offset_below_surface],
                            yaw=desired_yaw,
                            frame_name=self.current_marker_frame_name
                        )
                        
                    elif self.reorient_while_sliding_against_edge == True and self.contact_established_with_vertical_surface == False:
                        self.contact_established_with_vertical_surface = True
                        ms_to_execute = Util.make_action_goal_slide(
                            velocity=[None, -self.slide_velocity, None],
                            force=[None, None, -self.force_against_surface/2],
                            orientation=self.desired_orientation,
                            action_name="slide_until_edge",
                            frame_name=self.current_marker_frame_name,
                        )
                    else:
                        self.collect_points_on_edge_bool = True
                        ms_to_execute = Util.make_action_goal_slide(
                            velocity=[self.slide_velocity, None, None],
                            force=[None, -self.force_against_surface/1.2, -self.force_against_surface/1.2],
                            orientation=self.desired_orientation,
                            action_name=self.action_name_str,
                            frame_name=self.current_marker_frame_name,
                            time=3.0
                        )
                        self.last_direction_of_motion = [self.dir_of_sliding_motion_2d[0], self.dir_of_sliding_motion_2d[1], 0.0]
                        self.last_direction_of_force_while_sliding_against_edge = [dir_of_force[0], dir_of_force[1], 0.0]
                        
                elif self.current_action_spec.direction == Direction.CK:
                    self.desired_orientation, desired_yaw = Util.resolve_orientation(dir_of_motion=[-1.0, 0.0], orientation_input=Util.OrientationInput.RIGHT_OF_DIR_MOTION)
                    
                    if self.reorient_while_sliding_against_edge == False:
                        self.collect_points_on_edge_bool = False
                        self.reorient_while_sliding_against_edge = True
                        ms_to_execute = Util.make_action_goal_yaw(
                            position=[None, None, self.offset_below_surface],
                            yaw=desired_yaw,
                            frame_name=self.current_marker_frame_name
                        )
                    elif self.reorient_while_sliding_against_edge == True and self.contact_established_with_vertical_surface == False:
                        self.contact_established_with_vertical_surface = True
                        ms_to_execute = Util.make_action_goal_slide(
                            velocity=[None, -self.slide_velocity, None],
                            force=[None, None, -self.force_against_surface/2],
                            orientation=self.desired_orientation,
                            action_name="slide_until_edge",
                            frame_name=self.current_marker_frame_name,
                        )
                    else:
                        self.collect_points_on_edge_bool = True
                        ms_to_execute = Util.make_action_goal_slide(
                            velocity=[-self.slide_velocity, None, None],
                            force=[None, -self.force_against_surface/1.2, -self.force_against_surface/1.2],
                            orientation=self.desired_orientation,
                            action_name=self.action_name_str,
                            frame_name=self.current_marker_frame_name,
                            time=3.0
                        )
                        self.last_direction_of_motion = [-self.dir_of_sliding_motion_2d[0], -self.dir_of_sliding_motion_2d[1], 0.0]
                        self.last_direction_of_force_while_sliding_against_edge = [dir_of_force[0], dir_of_force[1], 0.0]
                self.send_goal(ms_to_execute)
    
    def publish_corners_and_estimate_plane(self):
        corner_points = []
        # collect corner points
        for i in range(self.n_sides):
            if self.rck.corners[i] is not None:
                point = self.rck.corners[i]
                # Ensure point is 3D (x, y, z); if only 2D, add z=0
                if len(point) == 2:
                    corner_points.append((point[0], point[1], 0.0))
                else:
                    corner_points.append(point)

                self.create_and_publish_marker_pose(
                    position=point, 
                    orientation=self.current_orientation, 
                    marker_type="corner", 
                    frame="marker_frame_0"
                )
        
        est_plane_origin_position, est_plane_orientation = Util.pose_from_points(points=corner_points, use_ransac=False)
        self.create_and_publish_marker_pose(
            position=est_plane_origin_position, 
            orientation=est_plane_orientation, 
            marker_type="plane", 
            frame="marker_frame_0"
        )
    
    def main_loop(self):
        """
        Main reasoning loop executed on timer.
        """
        # exit condition
        if self.dof == 0:
            self.publish_corners_and_estimate_plane()
            self.exploration_complete = True
            self.get_logger().info("Exploration complete - all parameters known")
            self._save_final_rck_to_json()
            rclpy.shutdown()
            return
        
        if self.stop_execution or self.state_of_execution == Util.StateOfExecution.FAILED:
            self.get_logger().error("Error detected - halting main loop")
            # TODO: check if default motion spec to be sent to stop motion of robot
            rclpy.shutdown()
            return
        
        if self.first_state_update_received is False:
            self.get_logger().info("Waiting for first state update...")
            return
        
        if (not self.state_of_execution == Util.StateOfExecution.EXECUTING and 
            not self.state_of_execution == Util.StateOfExecution.WAITING_FOR_SERVER and
            self.sliding_against_edge_sm_active):
            # this is continuously executed when sliding against edge state machine is active
            self.handle_sliding_against_unknown_surface()
        
        if self.get_new_action_list_bool:
            if self.current_action_type is not None:
                self.prev_action_instance = replace(self.prev_action_instance,
                                                    action_type=self.current_action_type, 
                                                    edge_index=self.current_ref_edge_index)
                self.prev_action_spec = ACTION_TO_SPEC[self.prev_action_instance.action_type]
            

            self.action_list = self._generate_action_list()
            
            if self.sliding_against_edge_sm_active == True:
                self.sliding_variables_initialized = False
                self.handle_sliding_against_unknown_surface()
                self.get_new_action_list_bool = False
                return
            elif not self.action_list:
                self.get_logger().warn("No valid action generated to execute")
                self.stop_execution = True
                return
            
            self.length_action_list = len(self.action_list)
            self.get_new_action_list_bool = False
        
        if (not self.state_of_execution == Util.StateOfExecution.EXECUTING and 
            not self.state_of_execution == Util.StateOfExecution.WAITING_FOR_SERVER and 
            not self.sliding_against_edge_sm_active and
            self.length_action_list > 0):
            if self.next_action_idx < self.length_action_list:
                ms_to_execute = self.action_list[self.next_action_idx]
                self.get_logger().info(f"Executing action {self.next_action_idx + 1}/{self.length_action_list}")
                self.next_action_idx += 1 # Note: in on_action_succeeded, if this idx is equal to length of action list, it is assumed that all actions in the list have been executed
                self.send_goal(ms_to_execute)
            
            elif self.next_action_idx == self.length_action_list:
                self.get_logger().info("All actions in current list executed, preparing to generate new action list")
                self.get_new_action_list_bool = True
                self.next_action_idx = 0
                # propagate current knowledge
                self._propagate_knowledge(knowledge="rck")

                # check if unique pattern is found for action selection
                self.unique_pattern_found_in_rck = find_unique_pattern(self.rck)

                if self.unique_pattern_found_in_rpk and not self.rpk_rck_matching_idx_found:
                    self.get_logger().info(f"Unique_pattern_found_in_rpk: {self.unique_pattern_found_in_rpk}, unique_pattern_found_in_rck: {self.unique_pattern_found_in_rck}")
                    self.get_logger().info("Attempting to match rck with rpk...")
                    self.rpk_rck_matching_idx_found, self.rpk_first_idx_in_rck = get_unique_pattern_ref_index(self.rck, self.rpk)
                if not self.rpk_rck_matching_idx_found:
                    if self.corner_coordinates_available_in_rpk:
                        print("Attempting to match rck with rpk using corner coordinates...")
                        self.rpk_rck_matching_idx_found, self.rpk_first_idx_in_rck = get_unique_pattern_ref_index(self.rck, self.rpk, match_corner_coordinates=True)
                    elif self.unique_pattern_found_in_rpk:
                        print("Attempting to find unique pattern in individual parameters...")
                        self.rpk_rck_matching_idx_found, self.rpk_first_idx_in_rck = get_unique_pattern_ref_index(
                            self.rck,
                            self.rpk,
                            find_match_in_individual_parameters=True,
                            validate_individual_match_across_fields=self.validate_individual_match_across_fields)
                
                if self.rpk_rck_matching_idx_found and not self.rck_rearranged:
                    self._capture_action_updates("before_reindex")
                    rearrange_rck_using_prior_knowledge(self.rck, self.rpk_first_idx_in_rck)
                    self.marker_id_for_edges[:] = self.marker_id_for_edges[self.rpk_first_idx_in_rck:] + self.marker_id_for_edges[:self.rpk_first_idx_in_rck] # rearrange marker ids in the same way as rck
                    self.noisy_points_on_edge[:] = self.noisy_points_on_edge[self.rpk_first_idx_in_rck:] + self.noisy_points_on_edge[:self.rpk_first_idx_in_rck]
                    self._reindex_edge_references_after_rck_rearrangement()
                    self.rck_rearranged = True
                    fill_missing_parameters(self.rck, self.rpk, self.rpk_rck_matching_idx_found)
                    self._propagate_knowledge(knowledge="rck")
                    self._sync_knowledge_to_graph()
                    self._capture_action_updates("after_reindex")
                    
                # find dof after propagation to check if exploration is complete
                self.dof = find_dof(self.rck)
                print(f"[main loop] Degrees of freedom after propagation: {self.dof}")
                
                # Update visualization
                self._sync_knowledge_to_graph()
                self._save_rck_snapshot_if_configured('post_action_list_update')

    def _reindex_edge_references_after_rck_rearrangement(self) -> None:
        """Rotate edge-indexed execution state into the RPK-aligned RCK frame."""
        shift = self.rpk_first_idx_in_rck
        if shift is None:
            raise RuntimeError("Cannot remap edge references without an RCK/RPK match index")

        def remap_edge_index(edge_index: Optional[int]) -> Optional[int]:
            if edge_index is None:
                return None
            return (edge_index - shift) % self.n_sides

        self.current_ref_edge_index = remap_edge_index(self.current_ref_edge_index)
        self.prev_action_instance = replace(
            self.prev_action_instance,
            edge_index=remap_edge_index(self.prev_action_instance.edge_index),
        )
        if self.current_action_log is not None:
            self.current_action_log["reindex"] = {
                "shift": int(shift),
                "reference_edge_before": self.current_action_log.get("ref_edge_index"),
                "reference_edge_after": self.current_ref_edge_index,
            }
    
    def _propagate_knowledge(self, knowledge: Literal["rck", "rpk"] = "rck",
                             min_points_to_remove_outliers: Optional[int] = None,
                             inlier_distance_threshold: Optional[float] = None):
        """
        Propagate/resolve current knowledge using core_algorithm rules.
        This fills in unknown values based on known ones.
        
        Parameters are loaded from config if not explicitly provided.
        """
        if min_points_to_remove_outliers is None:
            min_points_to_remove_outliers = self.config['knowledge_propagation']['min_points_to_remove_outliers']
        
        if inlier_distance_threshold is None:
            inlier_distance_threshold = self.config['knowledge_propagation']['inlier_distance_threshold']
        
        if knowledge == "rpk":
            knowledge_obj = self.rpk
        elif knowledge == "rck":
            knowledge_obj = self.rck
        else:
            self.get_logger().warn(f"Unknown knowledge type {knowledge} for propagation")
            return
        
        propagate_parameters(
            knowledge_obj,
            min_points_to_remove_outlers=min_points_to_remove_outliers,
            inlier_distance_threshold=inlier_distance_threshold
        )
            
    def _validate_edge_uv_and_normalize(self, edge_uv):
        if edge_uv is None:
            self.get_logger().warn(f"Edge unit vector for edge {self.current_ref_edge_index} is unknown, cannot determine direction of motion. Stopping execution.")
            self.stop_execution = True
            return
        norm = math.hypot(edge_uv[0], edge_uv[1])
        if norm != 1:
            rclpy.logging.get_logger(__name__).warn(f"Edge unit vector for edge {self.current_ref_edge_index} is not normalized: {edge_uv}. Normalizing it for further calculations.")
            if norm == 0:
                self.get_logger().warn(f"Edge unit vector for edge {self.current_ref_edge_index} has zero length, cannot determine direction of motion. Stopping execution.")
                self.stop_execution = True
                return
            edge_uv = (edge_uv[0]/norm, edge_uv[1]/norm)
        
        return edge_uv
    
    def _generate_action_list(self) -> List[str]:
        """
        Generate set of next actions
        
        Returns:
            List of motion specifications in the form of strings to execute
        """
        
        if not self.established_first_contact:
            self.desired_orientation = self.current_orientation
            self.get_logger().info("No contact established yet: generating default action to establish contact with surface")
            
            # pre-jnt config could also depend on initial belief of slope of plane wrt eddie_base_link, and can be followed by touch table to establish first contact
            action_list = [
                # TODO: move to pre-jnt-angle-configuration, followed by touch table
            ]
        
        # TODO: for extending to multiple surfaces, introduces indices and access plane slope information via reference index
        if not self.plane_slope_estimated:
            self.action_name_str = "slide_to_explore_plane"
            self.get_logger().info("Plane slope estimated: generating default action to estimate plane slope")
            
            # ideally, this depends on the initial belief of slope of plane wrt eddie_base_link
            
            ## log desired orientation for debugging
            self.get_logger().info(f"Desired orientation for plane slope estimation action: {self.desired_orientation}")
            self.desired_orientation = self.current_orientation
            self.desired_velocity = [0.0, -self.slide_velocity, None]
            self.last_direction_of_motion = [0.0, -1.0, 0.0]
            action_list = [
                Util.make_action_goal_slide(velocity=[self.slide_velocity,  0.0,  None], 
                                            force = [None, None, -self.force_against_surface], 
                                            orientation = self.desired_orientation, 
                                            action_name=self.action_name_str, 
                                            frame_name="eddie_base_link", 
                                            time=2.0),                
                Util.make_action_goal_slide(velocity=[0.0,   self.slide_velocity, None], 
                                            force = [None, None, -self.force_against_surface], 
                                            orientation = self.desired_orientation, 
                                            action_name=self.action_name_str, 
                                            frame_name="eddie_base_link", 
                                            time=3.0),
                Util.make_action_goal_slide(velocity=[-self.slide_velocity, 0.0,  None], 
                                            force = [None, None, -self.force_against_surface], 
                                            orientation = self.desired_orientation, 
                                            action_name=self.action_name_str, 
                                            frame_name="eddie_base_link", 
                                            time=2.0),
                Util.make_action_goal_slide(velocity=self.desired_velocity, 
                                            force = [None, None, -self.force_against_surface], 
                                            orientation = self.desired_orientation, 
                                            action_name=self.action_name_str, 
                                            frame_name="eddie_base_link", 
                                            time=3.0)
            ] # use points collected from these motions to estimate slope
            self.motion_indices_to_collect_points = [0, 1, 2, 3] # collect points from all 4 motions for slope estimation
            self.current_marker_frame_name = "eddie_base_link"
            return action_list
        
        try:
            # get next action recommendation from action selection algorithm
            self.current_selection_metadata = {
                "policy": "legacy-next-action",
                "rationale": (
                    "Selected by core_algorithm.next_action using current RCK, "
                    "previous action, and re-index state."
                ),
                "plan": None,
            }
            self.current_action_type, self.current_ref_edge_index = next_action(self.rck, self.prev_action_instance, self.rck_rearranged)
            self.current_selection_metadata["decision"] = self._json_safe(
                get_last_action_selection_trace()
            )
            self.current_action_spec = ACTION_TO_SPEC[self.current_action_type] if self.current_action_type is not None else None
            if self.current_action_type is not None:
                self._start_action_type_instance()

            # if next_action is None, check if knowledge is complete
            if self.current_action_type is None:
                self.get_logger().info("No action recommendation from action selection algorithm. Checking if knowledge is complete.")
                self._propagate_knowledge(knowledge="rck") # propagate knowledge before checking dof to fill in any values that can be resolved based on current knowledge
                self._sync_knowledge_to_graph()
                self.dof = find_dof(self.rck)
                print(f"Degrees of freedom: {self.dof}")
                if self.dof == 0:
                    self.exploration_complete = True
                    self.publish_corners_and_estimate_plane()
                    self.get_logger().info("Exploration complete - all parameters known")
                    self._save_final_rck_to_json()
                    rclpy.shutdown()
                else:
                    self.get_logger().warn("Knowledge is not complete but no action recommendation found. There might be an issue with action selection algorithm or the way knowledge is represented.")
                return
            self.get_logger().info(
                f"Next action: {self.current_action_type.name} with reference edge being: {self.current_ref_edge_index}"
            )
            self.get_logger().info(f"Current step count: {self.step_count}, current action type: {self.current_action_type.name}, current reference edge index: {self.current_ref_edge_index}")
            self.step_count += 1
            
            # get default orientation for this action type from config
            action_type_name = self.current_action_type.name
            default_orientation_val = 0  # fallback
            if 'action_defaults' in self.config and action_type_name in self.config['action_defaults']:
                default_orientation_val = self.config['action_defaults'][action_type_name].get('default_orientation', 0)
            
            # take user input for orientation if required for current action
            if self.take_user_input_for_orientation and self.current_action_type is not None:
                user_input = input("Enter orientation for this action (e.g., 0: in direction of motion, 1: against direction of motion, 2: right of motion, 3: left of motion): ")
                if user_input in ['0', '1', '2', '3']:
                    self.orientation_input = Util.OrientationInput(int(user_input))
                    self.get_logger().info(f"User input for orientation: {self.orientation_input}")
                else:
                    self.get_logger().warn("Invalid input for orientation, using default orientation from config for this action type")
                    self.orientation_input = Util.OrientationInput(default_orientation_val)
            elif self.current_action_type is None:
                self.get_logger().warn("No action could be determined")
                self.stop_execution = True
                return
            else:
                # use config-based default orientation for this action type
                self.orientation_input = Util.OrientationInput(default_orientation_val)
                self.get_logger().debug(f"Using default orientation {default_orientation_val} for action type {action_type_name}")
            
            # TODO: currently adding optional user input to confirm action execution. Remove once action selection is reliable
            attempt = 0
            num_attempts_allowed = 2
            while attempt < num_attempts_allowed:
                # answer = input("Do you want to continue? (yes/no): ").strip().lower()
                answer = "yes"
                if answer in ("yes", "y"):
                    print("Continuing...")
                    break
                elif answer in ("no", "n"):
                    print("Stopping.")
                    self.stop_execution = True
                    return
                else:
                    attempt += 1
                    if attempt < num_attempts_allowed: print(f"Invalid input. You have {num_attempts_allowed - attempt} attempts left.")
                    else:
                        print("Invalid input. Aborting.")
                        self.stop_execution = True
                        return
            
            ############# Action execution logic based on action type #############
            
            # use self.orientation_input to determine desired orientation for action execution
            if self.current_action_type == ActionType.SLIDE_OVER_SURFACE_PERPENDICULAR_TO_EDGE_GIVEN_ONE_POINT:
                if 'action_defaults' in self.config and action_type_name in self.config['action_defaults']:
                    offset_from_edge = self.config['action_defaults'][action_type_name].get('offset_from_edge', 0.05)
                else:
                    offset_from_edge = 0.05
                
                if self.current_marker_frame_name is None:
                    self.current_marker_frame_name = "marker_frame_0"
                
                # get closest point on edge to current end-effector position
                points_on_edge = self.rck.internal_points_on_edge[self.current_ref_edge_index]
                if len(points_on_edge) == 0:
                    self.get_logger().warn(f"No points on edge {self.current_ref_edge_index} to determine closest point. Stopping execution.")
                    self.stop_execution = True
                    return
                
                min_dist_idx = min(range(len(points_on_edge)),
                                    key=lambda i: (self.current_position[0] - points_on_edge[i][0])**2 +
                                                (self.current_position[1] - points_on_edge[i][1])**2)
                point_of_interest_on_edge = points_on_edge[min_dist_idx]
                
                # find a point perpendicular to edge, but on the plane using edge_unit_vector
                edge_uv = self.rck.edge_unit_vectors[self.current_ref_edge_index]
                edge_uv = self._validate_edge_uv_and_normalize(edge_uv)
                perp_direction = (-edge_uv[1], edge_uv[0]) # (-uy, ux) when rotated by 90 degrees in anticlockwise direction, but since edge unit vector is anticlockwise
                opp_to_perp_direction = (edge_uv[1], -edge_uv[0])
                point_on_plane = (point_of_interest_on_edge[0] + perp_direction[0]*offset_from_edge, point_of_interest_on_edge[1] + perp_direction[1]*offset_from_edge)
                
                # determine desired orientation based on direction of motion and user input
                self.desired_orientation, desired_yaw = Util.resolve_orientation(opp_to_perp_direction, self.orientation_input)
                self.desired_velocity = [self.slide_velocity*opp_to_perp_direction[0], self.slide_velocity*opp_to_perp_direction[1], None]
                
                self.action_name_str = "slide_until_edge"
                self.last_direction_of_motion = [opp_to_perp_direction[0], opp_to_perp_direction[1], 0.0]
                action_list = [
                        Util.make_action_goal_move(position=[self.current_position[0], self.current_position[1], self.offset_above_surface], 
                                                   orientation = self.current_orientation, 
                                                   frame_name="marker_frame_0"),

                        # attain desired yaw
                        Util.make_action_goal_yaw(position=[self.current_position[0], self.current_position[1], self.offset_above_surface], 
                                                  yaw = desired_yaw,
                                                  frame_name="marker_frame_0"),

                        # move above point on plane by offset_above_surface
                        Util.make_action_goal_move(position=[point_on_plane[0], point_on_plane[1], self.offset_above_surface], 
                                                   orientation = self.desired_orientation, 
                                                   frame_name="marker_frame_0"),
                        
                        
                        # touch table
                        Util.make_action_goal_touch(velocity=[0, 0, -self.touch_velocity], 
                                                    orientation=self.desired_orientation,
                                                    frame_name="marker_frame_0"),
                        
                        # slide until edge is reached
                        Util.make_action_goal_slide(velocity=self.desired_velocity, 
                                                    force = [None, None, -self.force_against_surface], 
                                                    orientation = self.desired_orientation, 
                                                    action_name=self.action_name_str, 
                                                    frame_name="marker_frame_0")
                ]
                self.current_marker_frame_name = "marker_frame_0"
                return action_list
            
            elif self.current_action_type == ActionType.SLIDE_OVER_SURFACE_UNTIL_EDGE:
                # slide in default direction (positive x-axis of marker frame 0) until edge is reached.
                # This action is executed after plane slope estimation
                self.desired_orientation, desired_yaw = Util.resolve_orientation(dir_of_motion=(1, 0), orientation_input=self.orientation_input)
                self.desired_velocity = [self.slide_velocity, 0.0, None]
                
                self.last_direction_of_motion = [1.0, 0.0, 0.0]
                
                # Note: current_position is in the frame of previous motion specification.
                # Thus, instead of using current_position, using [0, 0, 0]
                if self.action_name_str=="slide_to_explore_plane":
                    desired_pos_while_setting_yaw = [0.0, 0.0, self.offset_above_surface]
                    self.current_marker_frame_name = "marker_frame_0"
                else:
                    desired_pos_while_setting_yaw = [self.current_position[0], self.current_position[1], self.offset_above_surface]
                
                self.action_name_str = "slide_until_edge"
                action_list = [
                        # attain desired yaw
                        Util.make_action_goal_yaw(position=desired_pos_while_setting_yaw, 
                                                  yaw = desired_yaw,
                                                  frame_name="marker_frame_0"),
                        
                        # touch table
                        Util.make_action_goal_touch(velocity=[0, 0, -self.touch_velocity], 
                                                    orientation=self.desired_orientation, 
                                                    frame_name="marker_frame_0"),
                        
                        # slide until edge is reached
                        Util.make_action_goal_slide(velocity=self.desired_velocity, 
                                                    force = [None, None, -self.force_against_surface], 
                                                    orientation = self.desired_orientation, 
                                                    action_name=self.action_name_str, 
                                                    frame_name="marker_frame_0")
                ]
                self.current_marker_frame_name = "marker_frame_0"
                return action_list
            
            elif self.current_action_spec.mode in [Mode.AGAINST_VERTICAL, Mode.AGAINST_EDGE]:
                self.sliding_against_edge_sm_active = True
                rclpy.logging.get_logger(__name__).info(f"Action type {self.current_action_type.name} is in mode {self.current_action_spec.mode.name}, enabling sliding against edge state machine")
                return 
            
            elif self.current_action_type in [
                ActionType.SLIDE_OVER_SURFACE_PARALLEL_TO_EDGE_CCK,
                ActionType.SLIDE_OVER_SURFACE_PARALLEL_TO_EDGE_WITHIN_RANGE_CCK
            ]:
                edge_uv = self.rck.edge_unit_vectors[self.current_ref_edge_index]
                edge_uv = self._validate_edge_uv_and_normalize(edge_uv)
                perp_direction = (-edge_uv[1], edge_uv[0]) # (-uy, ux) when rotated by 90 degrees in anticlockwise direction, but since edge unit vector is anticlockwise
                opp_direction = (-edge_uv[0], -edge_uv[1])
                
                points_on_edge = self.rck.internal_points_on_edge[self.current_ref_edge_index]
                # print points on edge for debugging
                if self.debug_log: print(f"Points on edge {self.current_ref_edge_index}: {points_on_edge}")
                
                if len(points_on_edge) == 0:
                    self.get_logger().warn(f"No points on edge {self.current_ref_edge_index} to determine point for sliding parallel to edge. Stopping execution.")
                    self.stop_execution = True
                    return
                point_on_edge = points_on_edge[-1]
                if self.debug_log: print(f"Selected point on edge for sliding parallel to edge: {point_on_edge}")
                point_on_plane = (point_on_edge[0] + perp_direction[0]*self.slide_offset_from_edge + opp_direction[0]*self.slide_offset_from_edge, 
                                  point_on_edge[1] + perp_direction[1]*self.slide_offset_from_edge + opp_direction[1]*self.slide_offset_from_edge)
                if self.debug_log: print("offset point on plane for sliding parallel to edge: ", point_on_plane)
                self.desired_orientation, desired_yaw = Util.resolve_orientation(dir_of_motion=edge_uv, orientation_input=self.orientation_input)
                self.desired_velocity = [self.slide_velocity*edge_uv[0], self.slide_velocity*edge_uv[1], None]
                
                self.last_direction_of_motion = [edge_uv[0], edge_uv[1], 0.0]
                
                if self.current_action_type == ActionType.SLIDE_OVER_SURFACE_PARALLEL_TO_EDGE_WITHIN_RANGE_CCK:
                    self.action_name_str = "slide_parallel_to_edge_within_range"
                else:
                    self.action_name_str = "slide_until_edge"
                action_list = [
                    # move above the surface
                    Util.make_action_goal_move(position=[self.current_position[0], self.current_position[1], self.offset_above_surface], 
                                                orientation = self.current_orientation, 
                                                frame_name="marker_frame_0"),
                    Util.make_action_goal_move(position=[point_on_plane[0], point_on_plane[1], self.offset_above_surface],
                                                orientation = self.current_orientation,
                                                frame_name="marker_frame_0"),
                    
                    # attain desired yaw
                    Util.make_action_goal_yaw(position=[point_on_plane[0], point_on_plane[1], self.offset_above_surface],
                                                yaw = desired_yaw,
                                                frame_name="marker_frame_0"),
                    
                    # move to point on plane offset from edge to slide parallel to edge
                    Util.make_action_goal_move(position=[point_on_plane[0], point_on_plane[1], self.offset_above_surface], 
                                                orientation = self.desired_orientation,
                                                frame_name="marker_frame_0"),
                    
                    # touch table
                    Util.make_action_goal_touch(velocity=[0, 0, -self.touch_velocity], 
                                                orientation=self.desired_orientation, 
                                                frame_name="marker_frame_0"),
                    
                    # slide parallel to edge in one direction
                    Util.make_action_goal_slide(velocity=self.desired_velocity, 
                                                force = [None, None, -self.force_against_surface], 
                                                orientation = self.desired_orientation, 
                                                action_name=self.action_name_str, 
                                                frame_name="marker_frame_0")
                ]
                self.current_marker_frame_name = "marker_frame_0"
                return action_list
            
            elif self.current_action_type in [
                ActionType.SLIDE_OVER_SURFACE_PARALLEL_TO_EDGE_CK,
                ActionType.SLIDE_OVER_SURFACE_PARALLEL_TO_EDGE_WITHIN_RANGE_CK
            ]:
                edge_uv = self.rck.edge_unit_vectors[self.current_ref_edge_index]                    
                edge_uv = self._validate_edge_uv_and_normalize(edge_uv)
                perp_direction = (-edge_uv[1], edge_uv[0]) # (-uy, ux) when rotated by 90 degrees in anticlockwise direction, but since edge unit vector is anticlockwise
                edge_uv_ck = [-edge_uv[0], -edge_uv[1]]
                
                points_on_edge = self.rck.internal_points_on_edge[self.current_ref_edge_index]
                if len(points_on_edge) == 0:
                    self.get_logger().warn(f"No points on edge {self.current_ref_edge_index} to determine point for sliding parallel to edge. Stopping execution.")
                    self.stop_execution = True
                    return
                point_on_edge = points_on_edge[0]
                point_on_plane = (point_on_edge[0] + perp_direction[0]*self.slide_offset_from_edge + edge_uv[0]*self.slide_offset_from_edge, 
                                  point_on_edge[1] + perp_direction[1]*self.slide_offset_from_edge + edge_uv[1]*self.slide_offset_from_edge)

                self.desired_orientation, desired_yaw = Util.resolve_orientation(dir_of_motion=edge_uv_ck, orientation_input=self.orientation_input)
                self.desired_velocity = [self.slide_velocity*edge_uv_ck[0], self.slide_velocity*edge_uv_ck[1], None]
                
                self.last_direction_of_motion = [edge_uv_ck[0], edge_uv_ck[1], 0.0]
                
                if self.current_action_type == ActionType.SLIDE_OVER_SURFACE_PARALLEL_TO_EDGE_WITHIN_RANGE_CK:
                    self.action_name_str = "slide_parallel_to_edge_within_range"
                else:
                    self.action_name_str = "slide_until_edge"
                action_list = [
                    Util.make_action_goal_move(position=[self.current_position[0], self.current_position[1], self.offset_above_surface], 
                                                orientation = self.current_orientation, 
                                                frame_name="marker_frame_0"),
                    # attain desired yaw
                    Util.make_action_goal_yaw(position=[self.current_position[0], self.current_position[1], self.offset_above_surface], 
                                                yaw = desired_yaw,
                                                frame_name="marker_frame_0"),
                    
                    # move to a point offset from edge to slide parallel to edge
                    Util.make_action_goal_move(position=[point_on_plane[0], point_on_plane[1], self.offset_above_surface], 
                                                orientation = self.desired_orientation,
                                                frame_name="marker_frame_0"),
                    # touch table
                    Util.make_action_goal_touch(velocity=[0, 0, -self.touch_velocity], 
                                                orientation=self.desired_orientation, 
                                                frame_name="marker_frame_0"),
                    
                    # slide parallel to edge in one direction
                    Util.make_action_goal_slide(velocity=self.desired_velocity, 
                                                force = [None, None, -self.force_against_surface], 
                                                orientation = self.desired_orientation, 
                                                action_name=self.action_name_str, 
                                                frame_name="marker_frame_0")
                ]
                self.current_marker_frame_name = "marker_frame_0"
                return action_list
            
            
            elif self.current_action_type == ActionType.SLIDE_OVER_SURFACE_PARALLEL_FROM_OUTSIDE_TO_EDGE_CCK:
                edge_uv = self.rck.edge_unit_vectors[self.current_ref_edge_index]
                edge_uv = self._validate_edge_uv_and_normalize(edge_uv)
                edge_uv_ck = [-edge_uv[0], -edge_uv[1]]
                
                corner_coordinate = self.rck.corners[self.current_ref_edge_index]
                
                # check if corner is already known
                corner_known = False
                if corner_coordinate is not None:
                    self.get_logger().info(f"Corner coordinate for adjacent edge {self.current_ref_edge_index} is known: {corner_coordinate}")
                    corner_known = True
                else:
                    self.get_logger().info(f"Corner coordinate for adjacent edge {self.current_ref_edge_index} is unknown")
                
                # get internal point closest to the corner
                points_on_edge = self.rck.internal_points_on_edge[self.current_ref_edge_index]
                if len(points_on_edge) == 0:
                    self.get_logger().warn(f"No points on edge {self.current_ref_edge_index} to determine point for sliding parallel to edge. Stopping execution.")
                    self.stop_execution = True
                    return
                perp_direction = (-edge_uv[1], edge_uv[0]) # (-uy, ux) when rotated by 90 degrees in anticlockwise direction, but since edge unit vector is anticlockwise
                opp_to_perp_direction = (edge_uv[1], -edge_uv[0])
                
                if not corner_known:
                    point_on_edge = points_on_edge[0]
                else:
                    point_on_edge = corner_coordinate
                
                point_on_plane = (point_on_edge[0] + perp_direction[0]*self.slide_offset_from_edge, 
                                  point_on_edge[1] + perp_direction[1]*self.slide_offset_from_edge)
                
                point_closest_to_start_of_edge = (point_on_plane[0] + edge_uv_ck[0]*0.15, point_on_plane[1] + edge_uv_ck[1]*0.15) # point on plane further away from edge in direction of edge unit vector
                point_to_traverse_in_opp_direction = (point_closest_to_start_of_edge[0] + opp_to_perp_direction[0]*0.15, point_closest_to_start_of_edge[1] + opp_to_perp_direction[1]*0.15) # point on plane with offset to best_edge
                    
                
                self.desired_orientation, desired_yaw = Util.resolve_orientation(dir_of_motion=edge_uv, orientation_input=self.orientation_input)
                self.desired_velocity = [self.slide_velocity*edge_uv[0], self.slide_velocity*edge_uv[1], None]
                
                self.last_direction_of_motion = [edge_uv[0], edge_uv[1], 0.0]
                self.action_name_str = "slide_until_edge"
                action_list = [
                    ## case 1: prev: slide until corner and reflexive and dihedral is 90/unknown
                    ## case 2: prev: slide against vertical until corner, and dihedral is 270
                    ## case 3: when a corner of best edge is known, and arm is not at adjacent edge
                    Util.make_action_goal_move(position=[self.current_position[0], self.current_position[1], self.offset_above_surface], 
                                                orientation = self.current_orientation, 
                                                frame_name="marker_frame_0"),
                    # attain desired yaw
                    Util.make_action_goal_yaw(position=[self.current_position[0], self.current_position[1], self.offset_above_surface], 
                                            yaw = desired_yaw,
                                            frame_name="marker_frame_0"),
                    
                    # move to a point offset from edge to slide parallel to edge
                    Util.make_action_goal_move(position=[point_on_plane[0], point_on_plane[1], self.offset_above_surface], 
                                                orientation = self.desired_orientation,
                                                frame_name="marker_frame_0"),
                    Util.make_action_goal_move(position=[point_closest_to_start_of_edge[0], point_closest_to_start_of_edge[1], self.offset_above_surface], 
                                                orientation = self.desired_orientation,
                                                frame_name="marker_frame_0"),
                    Util.make_action_goal_move(position=[point_to_traverse_in_opp_direction[0], point_to_traverse_in_opp_direction[1], self.offset_above_surface], 
                                                orientation = self.desired_orientation,
                                                frame_name="marker_frame_0"),
                    
                    # touch table
                    Util.make_action_goal_touch(velocity=[0, 0, -self.touch_velocity], 
                                                orientation=self.desired_orientation, 
                                                frame_name="marker_frame_0"),
                    
                    # slide parallel to edge in one direction
                    Util.make_action_goal_slide(velocity=self.desired_velocity, 
                                                force = [None, None, -self.force_against_surface], 
                                                orientation = self.desired_orientation, 
                                                action_name=self.action_name_str, 
                                                frame_name="marker_frame_0")
                ]
                
                if self.current_position[2] > 0.01:
                    # remove first move action in action list
                    action_list = action_list[1:]
                self.current_marker_frame_name = "marker_frame_0"
                return action_list
            
            elif self.current_action_type == ActionType.SLIDE_OVER_SURFACE_PARALLEL_FROM_OUTSIDE_TO_EDGE_CK:
                edge_uv = self.rck.edge_unit_vectors[self.current_ref_edge_index]
                edge_uv = self._validate_edge_uv_and_normalize(edge_uv)
                edge_uv_ck = [-edge_uv[0], -edge_uv[1]]
                
                next_edge_idx = (self.current_ref_edge_index + 1) % self.rck.n_sides
                corner_coordinate = self.rck.corners[next_edge_idx]
                
                # check if corner is already known
                corner_known = False
                if corner_coordinate is not None:
                    self.get_logger().info(f"Corner coordinate for next edge {next_edge_idx} is known: {corner_coordinate}")
                    corner_known = True
                else:
                    self.get_logger().info(f"Corner coordinate for next edge {next_edge_idx} is unknown")
                
                # get internal point closest to the corner
                points_on_edge = self.rck.internal_points_on_edge[self.current_ref_edge_index]
                if len(points_on_edge) == 0:
                    self.get_logger().warn(f"No points on edge {self.current_ref_edge_index} to determine point for sliding parallel to edge. Stopping execution.")
                    self.stop_execution = True
                    return
                perp_direction = (-edge_uv[1], edge_uv[0]) # (-uy, ux) when rotated by 90 degrees in anticlockwise direction, but since edge unit vector is anticlockwise
                opp_to_perp_direction = (edge_uv[1], -edge_uv[0])
                
                if not corner_known:
                    point_on_edge = points_on_edge[-1]
                else:
                    point_on_edge = corner_coordinate
                
                point_on_plane = (point_on_edge[0] + perp_direction[0]*self.slide_offset_from_edge, point_on_edge[1] + perp_direction[1]*self.slide_offset_from_edge)
                
                point_further_away_in_dir_of_edge = (point_on_plane[0] + edge_uv[0]*0.15, point_on_plane[1] + edge_uv[1]*0.15) # point on plane further away from edge in direction of edge unit vector
                point_to_traverse_in_opp_direction = (point_further_away_in_dir_of_edge[0] + opp_to_perp_direction[0]*0.20, point_further_away_in_dir_of_edge[1] + opp_to_perp_direction[1]*0.20) # point on plane with offset to best_edge
                    
                
                self.desired_orientation, desired_yaw = Util.resolve_orientation(dir_of_motion=edge_uv_ck, orientation_input=self.orientation_input)
                self.desired_velocity = [self.slide_velocity*edge_uv_ck[0], self.slide_velocity*edge_uv_ck[1], None]
                
                self.action_name_str = "slide_until_edge"
                self.last_direction_of_motion = [edge_uv_ck[0], edge_uv_ck[1], 0.0]
                action_list = [
                    ## case 1: prev: slide until corner and reflexive and dihedral is 90/unknown
                    ## case 2: prev: slide against vertical until corner, and dihedral is 270
                    ## case 3: when a corner of best edge is known, and arm is not at adjacent edge
                    
                    # move above surface
                    Util.make_action_goal_move(position=[self.current_position[0], self.current_position[1], self.offset_above_surface], 
                                                orientation = self.current_orientation, 
                                                frame_name="marker_frame_0"),
                    
                    # move with an offset from edge
                    Util.make_action_goal_move(position=[point_on_plane[0], point_on_plane[1], self.offset_above_surface], 
                                                orientation = self.current_orientation,
                                                frame_name="marker_frame_0"),

                    # attain desired yaw
                    Util.make_action_goal_yaw(position=[point_on_plane[0], point_on_plane[1], self.offset_above_surface], 
                                                yaw = desired_yaw,
                                                frame_name="marker_frame_0"),

                    # move along the edge in the direction opposite to the final slide
                    Util.make_action_goal_move(position=[point_further_away_in_dir_of_edge[0], point_further_away_in_dir_of_edge[1], self.offset_above_surface], 
                                                orientation = self.desired_orientation,
                                                frame_name="marker_frame_0"),
                    
                    # move to other side of edge to get point
                    Util.make_action_goal_move(position=[point_to_traverse_in_opp_direction[0], point_to_traverse_in_opp_direction[1], self.offset_above_surface], 
                                                orientation = self.desired_orientation,
                                                frame_name="marker_frame_0"),
                    
                    
                    # touch table
                    Util.make_action_goal_touch(velocity=[0, 0, -self.touch_velocity], 
                                                orientation=self.desired_orientation, 
                                                frame_name="marker_frame_0"),
                    
                    # slide parallel to edge in one direction
                    Util.make_action_goal_slide(velocity=self.desired_velocity, 
                                                force = [None, None, -self.force_against_surface], 
                                                orientation = self.desired_orientation, 
                                                action_name=self.action_name_str, 
                                                frame_name="marker_frame_0")
                ]
                
                if self.current_position[2] > 0.01:
                    # remove first move action in action list
                    action_list = action_list[1:]
                self.current_marker_frame_name = "marker_frame_0"
                return action_list
            
            elif (self.current_action_type == ActionType.MOVE_PARALLEL_FROM_OUTSIDE_TO_EDGE_UNTIL_CONTACT_CK or
                  self.current_action_type == ActionType.MOVE_PARALLEL_FROM_OUTSIDE_TO_EDGE_UNTIL_CONTACT_CCK):
                if self.prev_action_instance.action_type is None:
                    self.get_logger().warn(f"No previous action instance to determine reference edge for orientation. Stopping execution.")
                    self.stop_execution = True
                    return
                
                if self.prev_action_spec.stop == Stop.UNTIL_CORNER:
                    cw_motion = False
                    if self.current_action_type == ActionType.MOVE_PARALLEL_FROM_OUTSIDE_TO_EDGE_UNTIL_CONTACT_CK:
                        cw_motion = True
                    # case 1: if prev action was move until corner and corner angle is non-reflexive and dihedral is 270: dir matters, ref-edge is adjacent edge
                    edge_uv = self.rck.edge_unit_vectors[self.current_ref_edge_index]
                    edge_uv = self._validate_edge_uv_and_normalize(edge_uv)
                    edge_uv_ck = [-edge_uv[0], -edge_uv[1]]
                    
                    if cw_motion:
                        desired_direction_of_motion = edge_uv_ck
                    else:
                        desired_direction_of_motion = edge_uv

                    perp_direction = (-edge_uv[1], edge_uv[0])
                    point_to_move_to = (
                        self.current_position[0] + perp_direction[0] * self.offset_from_edge_while_moving_from_outside_to_edge,
                        self.current_position[1] + perp_direction[1] * self.offset_from_edge_while_moving_from_outside_to_edge
                    )
                    
                    self.desired_velocity = [self.slide_velocity*desired_direction_of_motion[0], self.slide_velocity*desired_direction_of_motion[1], 0.0]
                    
                    self.action_name_str = "touch_edge"
                    self.last_direction_of_motion = [desired_direction_of_motion[0], desired_direction_of_motion[1], 0.0]
                    self.desired_orientation, desired_yaw = Util.resolve_orientation(
                        dir_of_motion=desired_direction_of_motion,
                        orientation_input=self.orientation_input)
                    current_yaw = -float(R.from_quat(self.current_orientation).as_euler('zyx', degrees=True)[0])
                    yaw_error = abs((current_yaw - desired_yaw + 180.0) % 360.0 - 180.0)

                    action_list = [
                        Util.make_action_goal_move(position=[point_to_move_to[0], point_to_move_to[1], -self.offset_below_surface],
                                                    orientation = self.current_orientation,
                                                    frame_name="marker_frame_0")
                    ]
                    if yaw_error > 5.0:
                        action_list.append(
                            Util.make_action_goal_yaw(position=[point_to_move_to[0], point_to_move_to[1], -self.offset_below_surface],
                                                      yaw=desired_yaw,
                                                      frame_name="marker_frame_0")
                        )
                    action_list.append(
                        Util.make_action_goal_touch(velocity=self.desired_velocity,
                                                    orientation=self.desired_orientation,
                                                    frame_name="marker_frame_0",
                                                    action_name=self.action_name_str)
                    )
                    self.current_marker_frame_name = "marker_frame_0"
                    return action_list
                
                elif self.prev_action_spec.stop in (Stop.UNTIL_EDGE_CONTACT,
                                                    Stop.UNTIL_EDGE_CONTACT_WITHIN_RANGE):
                    # case 2: if prev action was slide until edge and if dihedral is 270, then move back to get point: dir doesn't matter. ref-edge is edge to contact
                    # get perpendicular dir vector to edge_uv
                    edge_uv = self.rck.edge_unit_vectors[self.current_ref_edge_index]
                    if edge_uv is None:
                        prev_desired_velocity = self.desired_velocity
                        # replace None with 0.0
                        prev_desired_velocity = [v if v is not None else 0.0 for v in prev_desired_velocity]
                        norm_of_prev_desired_velocity = np.linalg.norm(prev_desired_velocity)
                        # prev_dir_of_motion = (prev_desired_velocity[0]/norm_of_prev_desired_velocity, prev_desired_velocity[1]/norm_of_prev_desired_velocity)
                        # opp_to_prev_dir_of_motion = (-prev_dir_of_motion[0], -prev_dir_of_motion[1])
                        opp_to_prev_dir_of_motion = (-prev_desired_velocity[0]/norm_of_prev_desired_velocity, -prev_desired_velocity[1]/norm_of_prev_desired_velocity)
                        desired_dir_of_vel = opp_to_prev_dir_of_motion
                        self.get_logger().warn(f"Edge unit vector for edge {self.current_ref_edge_index} is None. \
                                               Cannot determine edge direction for moving parallel to edge until contact. \
                                               Using opposite to previous desired velocity direction {opp_to_prev_dir_of_motion} \
                                               for determining point on the edge.")
                    else:
                        edge_uv = self._validate_edge_uv_and_normalize(edge_uv)
                        """
                        perp_direction = (-edge_uv[1], edge_uv[0]) # (-uy, ux) when rotated by 90 degrees in anticlockwise direction, but since edge unit vector is anticlockwise
                        opp_to_perp_direction = (edge_uv[1], -edge_uv[0])
                        desired_dir_of_vel = opp_to_perp_direction
                        """
                        opp_to_edge_uv = (-edge_uv[0], -edge_uv[1])
                        desired_dir_of_vel = opp_to_edge_uv
                    
                    # offset current position opposite to the desired_dir_of_vel to get reliable free space
                    offset_distance = 0.02 # 2cm
                    
                    offset_position = (self.current_position[0] - desired_dir_of_vel[0]*offset_distance, self.current_position[1] - desired_dir_of_vel[1]*offset_distance)
                    
                    self.desired_velocity = [self.touch_velocity*desired_dir_of_vel[0], self.touch_velocity*desired_dir_of_vel[1], 0.0]
                    
                    self.action_name_str = "touch_edge"
                    self.last_direction_of_motion = [desired_dir_of_vel[0], desired_dir_of_vel[1], 0.0]
                    action_list = [
                        Util.make_action_goal_move(position=[offset_position[0], offset_position[1], 0.0],
                                                    orientation = self.current_orientation,
                                                    frame_name="marker_frame_0"),
                        Util.make_action_goal_move(position=[offset_position[0], offset_position[1], -self.offset_below_surface],
                                                    orientation = self.current_orientation,
                                                    frame_name="marker_frame_0"),
                        Util.make_action_goal_touch(velocity=self.desired_velocity,
                                                    orientation=self.current_orientation,
                                                    frame_name="marker_frame_0",
                                                    action_name=self.action_name_str)
                    ]
                    self.current_marker_frame_name = "marker_frame_0"
                    return action_list
                
                else:
                    self.get_logger().warn(f"[MOVE_PARALLEL_FROM_OUTSIDE_TO_EDGE_UNTIL_CONTACT_CK] Previous action stop condition is not suitable for determining motions for current action. Stopping execution.")
                    self.stop_execution = True
                    return
            
            else:
                self.get_logger().warn(f"Unknown action type from action selection")
                return []
        
        except Exception as e:
            self.get_logger().error(f"Error generating action list: {e}")
            self.get_logger().error(traceback.format_exc())
            return []

    # ========== ROS 2 related Methods ==========
    
    def send_goal(self, motion_specification: str):
        """
        Send a goal to the motion specification action server.
        
        Args:
            motion_specification: Motion specification string
        """
        if not self.client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("Motion specification server not available")
            self.state_of_execution = Util.StateOfExecution.WAITING_FOR_SERVER
            self.stop_execution = True
            return
        
        # # Debug: print motion_specification before parsing
        # self.get_logger().info(f"DEBUG: motion_specification = {motion_specification}")
        
        ms_json = ast.literal_eval(motion_specification)
        arm_name = ms_json["arm_name"]
        self.current_ms_frame = ms_json[arm_name]["frame_name"]
        self._begin_primitive_log(motion_specification, ms_json)
        
        goal_msg = MotionSpecification.Goal()
        goal_msg.motion_specification = str(motion_specification)
        
        ms_parsed = ast.literal_eval(motion_specification)
        
        self.get_logger().info("Sending goal:\n" + json.dumps(ms_parsed, indent=4))
        self.rck.print_knowledge("Robot Current Knowledge (rck) [Before sending goal]")
        
        send_goal_future = self.client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        self.state_of_execution = Util.StateOfExecution.EXECUTING
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """
        Handle response from action server after sending goal.
        
        :param future: Future object from send_goal_async
        """
        
        try:
            goal_handle = future.result()
        except Exception as e:
            self.get_logger().error(f"Send goal failed: {e}")
            return
        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected")
            return
        self.get_logger().info("Goal accepted, waiting for result")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)
    
    def feedback_callback(self, feedback_msg):
        """
        Handle feedback from motion server during action execution.
        
        :param feedback_msg: Feedback message from action server in the frame of motion specification
        """
        
        try:
            tcp_position = feedback_msg.feedback.tcp_position
        except AttributeError:
            self.get_logger().debug("No tcp_position in feedback")
    
    def result_callback(self, future):
        try:
            result = future.result()
        except Exception as e:
            self.get_logger().error(f"Result failed: {e}")
            self._finish_primitive_log("failed", error=e)
            return

        if result.status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("Goal succeeded")
            try:
                self.last_guard_diagnostic = self._terminal_outcome_metadata(result.result)
                self._record_knowledge_stage("before_action_interpretation")
                self.on_action_succeeded(result.result)
            finally:
                # Knowledge updates happen inside on_action_succeeded, so the
                # primitive record is closed afterwards and captures the
                # post-result state without storing high-rate signals.
                self._finish_primitive_log("succeeded", result.result)
                self.last_guard_diagnostic = None
            self.state_of_execution = Util.StateOfExecution.COMPLETED
        else:
            self.get_logger().error(f"Goal failed with status {result.status}")
            self.last_guard_diagnostic = self._terminal_outcome_metadata(result.result)
            self._finish_primitive_log("failed", result=result.result)
            self.last_guard_diagnostic = None
            self.state_of_execution = Util.StateOfExecution.FAILED

    def on_action_succeeded(self, result):
        """
        Handle result from completed action.
        
        Updates robot knowledge and triggers next action.
        
        :param result: Result from motion specification action
        """
        self.get_logger().info(f"Action completed with result: {result}")
        
        if result is None:
            self.get_logger().warn("No result data received from action")
            return

        elif result.ms_action_name == "slide_to_explore_plane":
            self.points_on_plane.extend(self.collected_points)
            if self.next_action_idx < self.length_action_list:
                self.get_logger().info("Collecting points for plane slope estimation")
                return
            else:
                self.plane_slope_estimated = True
                self.plane_origin_position, self.plane_orientation = Util.pose_from_points(points=self.points_on_plane, use_ransac=True)
                self._record_fit_diagnostics(
                    "plane",
                    self.points_on_plane,
                    origin=self.plane_origin_position,
                    orientation=self.plane_orientation,
                )
                
                self.get_logger().info(f"Estimated plane pose from points: position={self.plane_origin_position}, orientation={self.plane_orientation}")
                self.create_and_publish_marker_pose(position=self.plane_origin_position, 
                                                    orientation=self.plane_orientation, 
                                                    marker_type="marker", 
                                                    frame="eddie_base_link")
                self.get_logger().info("Plane slope estimation complete, proceeding to next action list generation")

        elif result.ms_action_name == "find_edge_by_sliding":
            collected_points_2d = [(p[0], p[1]) for p in self.collected_points]
            print("completed an instance of sliding to find edge. Number of points collected: ", len(self.collected_points))
            self.points_on_edge.extend(collected_points_2d)
        
        elif result.ms_action_name in ["slide_against_edge_until_corner", 
                                       "slide_against_vertical_surface_until_corner", 
                                       "slide_against_surface_vector_only"]:
            self._record_completed_action_type()

            # reset relevant flags
            self.sliding_against_edge_sm_active = False
            self.collect_points_on_edge_bool = False
            self.get_new_action_list_bool = True
            self.next_action_idx = 0
            
            # get adjacent edge indices
            next_edge_idx = (self.current_ref_edge_index + 1) % self.rck.n_sides
            prev_edge_idx = (self.current_ref_edge_index - 1) % self.rck.n_sides
            
            # get direction from collected points
            collected_points_2d = [(p[0], p[1]) for p in self.collected_points]
            noisy_points_2d = collected_points_2d
            e_uv_before_resampling = Util.unit_vector_from_points_2d(points=collected_points_2d)
            if self.debug_log: print("edge uv before resampling: ", e_uv_before_resampling)
            if self.debug_log: print("collected points 2d: ", collected_points_2d)
            try:
                internal_points_2d = Util.resample_line_points(collected_points_2d)
            except RuntimeError as e:
                self.get_logger().warn(
                    f"Could not resample collected edge points ({e}). "
                    "Using raw collected points for this RCK update."
                )
                internal_points_2d = np.asarray(collected_points_2d, dtype=float)
            if self.debug_log: print("collected points 2d after resampling: ", internal_points_2d)
            radius_of_ee = self.diameter_of_end_effector * 0.5
            
            edge_uv = Util.unit_vector_from_points_2d(points=internal_points_2d)
            print("edge uv from points: ", edge_uv)
            self._record_fit_diagnostics("edge_trace", collected_points_2d, direction=edge_uv)
            
            # make it cck
            angle_with_direction_of_force = Util.get_ccw_angle(self.last_direction_of_motion[0:2], 
                                                               self.last_direction_of_force_while_sliding_against_edge[0:2])
            
            if self.current_action_spec.mode == Mode.AGAINST_EDGE:
                if angle_with_direction_of_force < math.pi:
                    self.get_logger().info("Sliding motion detected in the CCK direction")
                elif angle_with_direction_of_force > math.pi:
                    self.get_logger().info("Sliding motion detected in the CK direction. Flipping direction to get correct edge unit vector")
                    edge_uv = [-edge_uv[0], -edge_uv[1]] # invert direction to match CCK direction
                internal_points_2d = Util.offset_points(internal_points_2d, edge_uv, radius_of_ee, side='left')
                noisy_points_2d = Util.offset_points(noisy_points_2d, edge_uv, radius_of_ee, side='left')
            elif self.current_action_spec.mode == Mode.AGAINST_VERTICAL:
                if angle_with_direction_of_force < math.pi:
                    self.get_logger().info("Sliding motion detected in the CK direction. Flipping direction to get correct edge unit vector")
                    edge_uv = [-edge_uv[0], -edge_uv[1]] # invert direction to match CCK direction
                elif angle_with_direction_of_force > math.pi:
                    self.get_logger().info("Sliding motion detected in the CCK direction")
                vertical_surface_offset_distance = (radius_of_ee + self.offset_due_to_camera)
                internal_points_2d = Util.offset_points(
                    internal_points_2d,
                    edge_uv,
                    vertical_surface_offset_distance,
                    side='right'
                )
                noisy_points_2d = Util.offset_points(
                    noisy_points_2d,
                    edge_uv,
                    vertical_surface_offset_distance,
                    side='right'
                )
            # update internal points by taking offset into account and edge unit vector for the current reference edge based on sliding motion
            self.noisy_points_on_edge[self.current_ref_edge_index].extend([tuple(pt) if isinstance(pt, (list, tuple, np.ndarray)) else pt for pt in noisy_points_2d])
            self.rck.internal_points_on_edge[self.current_ref_edge_index] = [tuple(pt) if isinstance(pt, (list, tuple, np.ndarray)) else pt for pt in internal_points_2d]
            self.rck.edge_unit_vectors[self.current_ref_edge_index] = (edge_uv[0], edge_uv[1])
            rclpy.logging.get_logger("Reasoner").info(f"Updated internal points and edge unit vector. Edge unit vector for edge {self.current_ref_edge_index} is now {self.rck.edge_unit_vectors[self.current_ref_edge_index]}")

            # based on disjunction ids, update dihedral angle and reflexivity of corner angle            
            disjunction_id_for_dih_90_non_reflexive = 1     # while sliding against vertical surface
            disjunction_id_for_dih_270_non_reflexive = 2    # while sliding against vertical surface
            disjunction_id_for_dih_unknown_reflexive = 3    # while sliding against vertical surface
            disjunction_id_for_edge_reflexive = 1           # while sliding against edge
            disjunction_id_for_edge_non_reflexive = 2       # while sliding against edge
            
            if self.current_action_spec.direction == Direction.CK:
                edge_idx_of_interest_reflexivity = self.current_ref_edge_index
                edge_idx_of_interest_dihedral = prev_edge_idx
            elif self.current_action_spec.direction == Direction.CCK:
                edge_idx_of_interest_reflexivity = next_edge_idx
                edge_idx_of_interest_dihedral = next_edge_idx
            
            if self.current_action_spec.stop == Stop.UNTIL_CORNER:
                if self.current_action_spec.mode == Mode.AGAINST_VERTICAL: # valid action_name: slide_against_vertical_surface_until_corner
                    self.get_logger().info(f"Updated edge unit vector for edge {self.current_ref_edge_index} to {self.rck.edge_unit_vectors[self.current_ref_edge_index]} based on sliding motion against edge")
                    if disjunction_id_for_dih_90_non_reflexive in result.disjunction_indices:
                        self.rck.dihedrals[edge_idx_of_interest_dihedral] = 90.0
                        self.rck.is_reflexive_angle[edge_idx_of_interest_reflexivity] = False
                        rclpy.logging.get_logger("Reasoner").info(f"Updated dihedral angle of edge {edge_idx_of_interest_dihedral} to 90 degrees and non-reflexive angle type based on result of action {self.current_action_type.name}")
                    elif disjunction_id_for_dih_270_non_reflexive in result.disjunction_indices:
                        self.rck.dihedrals[edge_idx_of_interest_dihedral] = 270.0
                        self.rck.is_reflexive_angle[edge_idx_of_interest_reflexivity] = False
                        rclpy.logging.get_logger("Reasoner").info(f"Updated dihedral angle of edge {edge_idx_of_interest_dihedral} to 270 degrees and non-reflexive angle type based on result of action {self.current_action_type.name}")
                    elif disjunction_id_for_dih_unknown_reflexive in result.disjunction_indices:
                        self.rck.is_reflexive_angle[edge_idx_of_interest_reflexivity] = True
                        rclpy.logging.get_logger("Reasoner").info(f"Updated reflexive angle type based on result of action {self.current_action_type.name}")
                
                elif self.current_action_spec.mode == Mode.AGAINST_EDGE: # valid action_name: slide_against_edge_until_corner
                    self.get_logger().info(f"Updated edge unit vector for edge {self.current_ref_edge_index} to {self.rck.edge_unit_vectors[self.current_ref_edge_index]} based on sliding motion against vertical surface")
                    if disjunction_id_for_edge_reflexive in result.disjunction_indices:
                        # do nothing because, it might be due to a reflexive corner or due to the contact with the next edge's perpendicular surface
                        rclpy.logging.get_logger("Reasoner").info(f"Result of action {self.current_action_type.name} is consistent with reflexive angle type for edge {edge_idx_of_interest_reflexivity}. Not updating angle type since it might be due to contact with next edge's perpendicular surface rather than reflexivity Or with dih=270 with non-reflexivity due to workspace limit.")
                        pass
                    elif disjunction_id_for_edge_non_reflexive in result.disjunction_indices:
                        self.rck.is_reflexive_angle[edge_idx_of_interest_reflexivity] = False
                        rclpy.logging.get_logger("Reasoner").info(f"Updated angle type of edge {edge_idx_of_interest_reflexivity} to non-reflexive based on result of action {self.current_action_type.name}")
            
            # propagate current knowledge
            self._record_knowledge_stage("after_direct_evidence")
            self._propagate_knowledge(knowledge="rck")
            self._record_knowledge_stage("after_geometric_propagation")

            # check if unique pattern is found for action selection
            self.unique_pattern_found_in_rck = find_unique_pattern(self.rck)

            if self.unique_pattern_found_in_rpk and not self.rpk_rck_matching_idx_found:
                self.get_logger().info(f"Unique_pattern_found_in_rpk: {self.unique_pattern_found_in_rpk}, unique_pattern_found_in_rck: {self.unique_pattern_found_in_rck}")
                self.get_logger().info("Attempting to match rck with rpk...")
                self.rpk_rck_matching_idx_found, self.rpk_first_idx_in_rck = get_unique_pattern_ref_index(self.rck, self.rpk)
            if not self.rpk_rck_matching_idx_found:
                if self.corner_coordinates_available_in_rpk:
                    print("Attempting to match rck with rpk using corner coordinates...")
                    self.rpk_rck_matching_idx_found, self.rpk_first_idx_in_rck = get_unique_pattern_ref_index(self.rck, self.rpk, match_corner_coordinates=True)
                elif self.unique_pattern_found_in_rpk:
                    print("Attempting to find unique pattern in individual parameters...")
                    self.rpk_rck_matching_idx_found, self.rpk_first_idx_in_rck = get_unique_pattern_ref_index(
                        self.rck,
                        self.rpk,
                        find_match_in_individual_parameters=True,
                        validate_individual_match_across_fields=self.validate_individual_match_across_fields)
            
            if self.rpk_rck_matching_idx_found and not self.rck_rearranged:
                self._capture_action_updates("before_reindex")
                rearrange_rck_using_prior_knowledge(self.rck, self.rpk_first_idx_in_rck)
                self.marker_id_for_edges[:] = self.marker_id_for_edges[self.rpk_first_idx_in_rck:] + self.marker_id_for_edges[:self.rpk_first_idx_in_rck] # rearrange marker ids in the same way as rck
                self.noisy_points_on_edge[:] = self.noisy_points_on_edge[self.rpk_first_idx_in_rck:] + self.noisy_points_on_edge[:self.rpk_first_idx_in_rck]
                self._reindex_edge_references_after_rck_rearrangement()
                self.rck_rearranged = True
                fill_missing_parameters(self.rck, self.rpk, self.rpk_rck_matching_idx_found)
                self._propagate_knowledge(knowledge="rck")
                self._record_knowledge_stage("after_prior_alignment_propagation")
                self._sync_knowledge_to_graph()
                self._capture_action_updates("after_reindex")
                
            # find dof after propagation to check if exploration is complete
            self.dof = find_dof(self.rck)
            print(f"[on action success]Degrees of freedom after propagation: {self.dof}")
            
            # Update visualization
            self._sync_knowledge_to_graph()
            self._save_rck_snapshot_if_configured('sliding_state_machine_update')
            
        ## Update dihedrals when moved until edges
        
        if self.current_action_type is None:
            self.get_logger().warn("Previous action instance type is None, since default motion for plane estimation is executed. Skipping checking with action_types for updating RCK.")
            self.collected_points = []
            self.motion_indices_to_collect_points = []
            return
        
        motion_spec = ACTION_TO_SPEC[self.current_action_type]
        
        # update knowledge when a set of motion specifications are executed
        if self.next_action_idx == self.length_action_list:
            print("Final action in list completed. Updating RCK based on result")
            self._record_completed_action_type()

            ref_edge_idx = self.current_ref_edge_index
            prev_edge_idx = (ref_edge_idx - 1) % self.n_sides
            next_edge_idx = (ref_edge_idx + 1) % self.n_sides
            
            edge_idx_of_interest = ref_edge_idx
            
            if (self.current_action_type in [ActionType.SLIDE_OVER_SURFACE_UNTIL_EDGE,
                                            ActionType.SLIDE_OVER_SURFACE_PERPENDICULAR_TO_EDGE_GIVEN_ONE_POINT,
                                            ActionType.SLIDE_OVER_SURFACE_PARALLEL_FROM_OUTSIDE_TO_EDGE_CCK,
                                            ActionType.SLIDE_OVER_SURFACE_PARALLEL_FROM_OUTSIDE_TO_EDGE_CK,
                                            ActionType.SLIDE_OVER_SURFACE_PARALLEL_TO_EDGE_CCK,
                                            ActionType.SLIDE_OVER_SURFACE_PARALLEL_TO_EDGE_CK,
                                            ActionType.SLIDE_OVER_SURFACE_PARALLEL_TO_EDGE_WITHIN_RANGE_CCK,
                                            ActionType.SLIDE_OVER_SURFACE_PARALLEL_TO_EDGE_WITHIN_RANGE_CK]):

                disjunction_id_for_dihedral_90 = 1
                disjunction_id_for_dihedral_270 = 2
                disjunction_id_for_no_edge = 3
                
                if self.current_action_type == ActionType.SLIDE_OVER_SURFACE_PARALLEL_TO_EDGE_WITHIN_RANGE_CCK:
                    if disjunction_id_for_no_edge in result.disjunction_indices:
                        self.rck.is_reflexive_angle[next_edge_idx] = True
                        rclpy.logging.get_logger("Reasoner").info(f"Updated angle type of edge {next_edge_idx} to reflexive based on result of action {self.current_action_type.name}")
                    elif disjunction_id_for_dihedral_90 in result.disjunction_indices:
                        self.rck.is_reflexive_angle[next_edge_idx] = False
                        self.rck.dihedrals[next_edge_idx] = 90.0
                        rclpy.logging.get_logger("Reasoner").info(f"Updated dihedral angle of edge {next_edge_idx} to 90 degrees and angle type to non-reflexive based on result of action {self.current_action_type.name}")
                    elif disjunction_id_for_dihedral_270 in result.disjunction_indices:
                        self.rck.is_reflexive_angle[next_edge_idx] = False
                        self.rck.dihedrals[next_edge_idx] = 270.0
                        rclpy.logging.get_logger("Reasoner").info(f"Updated dihedral angle of edge {next_edge_idx} to 270 degrees and angle type to non-reflexive based on result of action {self.current_action_type.name}")
                elif self.current_action_type == ActionType.SLIDE_OVER_SURFACE_PARALLEL_TO_EDGE_WITHIN_RANGE_CK:
                    if disjunction_id_for_no_edge in result.disjunction_indices:
                        self.rck.is_reflexive_angle[ref_edge_idx] = True
                        rclpy.logging.get_logger("Reasoner").info(f"Updated angle type of edge {ref_edge_idx} to reflexive based on result of action {self.current_action_type.name}")
                    elif disjunction_id_for_dihedral_90 in result.disjunction_indices:
                        self.rck.is_reflexive_angle[ref_edge_idx] = False
                        self.rck.dihedrals[prev_edge_idx] = 90.0
                        rclpy.logging.get_logger("Reasoner").info(f"Updated dihedral angle of edge {prev_edge_idx} to 90 degrees and angle type of edge {ref_edge_idx} to non-reflexive based on result of action {self.current_action_type.name}")
                    elif disjunction_id_for_dihedral_270 in result.disjunction_indices:
                        self.rck.is_reflexive_angle[ref_edge_idx] = False
                        self.rck.dihedrals[prev_edge_idx] = 270.0
                        rclpy.logging.get_logger("Reasoner").info(f"Updated dihedral angle of edge {prev_edge_idx} to 270 degrees and angle type of edge {ref_edge_idx} to non-reflexive based on result of action {self.current_action_type.name}")
                
                else:
                    if self.current_action_type in [ActionType.SLIDE_OVER_SURFACE_UNTIL_EDGE,
                                                    ActionType.SLIDE_OVER_SURFACE_PERPENDICULAR_TO_EDGE_GIVEN_ONE_POINT]:
                        edge_idx_of_interest = ref_edge_idx
                    elif self.current_action_type in [ActionType.SLIDE_OVER_SURFACE_PARALLEL_FROM_OUTSIDE_TO_EDGE_CCK,
                                                    ActionType.SLIDE_OVER_SURFACE_PARALLEL_TO_EDGE_CK]:
                        edge_idx_of_interest = prev_edge_idx
                    elif self.current_action_type in [ActionType.SLIDE_OVER_SURFACE_PARALLEL_FROM_OUTSIDE_TO_EDGE_CK,
                                                    ActionType.SLIDE_OVER_SURFACE_PARALLEL_TO_EDGE_CCK]:
                        edge_idx_of_interest = next_edge_idx
                    
                    if disjunction_id_for_dihedral_90 in result.disjunction_indices:
                        self.rck.dihedrals[edge_idx_of_interest] = 90.0
                        rclpy.logging.get_logger("Reasoner").info(f"Updated dihedral angle of edge {edge_idx_of_interest} to 90 degrees based on result of action {self.current_action_type.name}")
                        # Note: this is approximate point on edge, which will be refined/filtered while sliding along this edge
                        last_desired_velocity = self.desired_velocity
                        norm = math.hypot(last_desired_velocity[0], last_desired_velocity[1])
                        last_dir_of_motion = (last_desired_velocity[0]/norm, last_desired_velocity[1]/norm) if norm > 0 else (0, 0)

                        ee_offset_vector = self.diameter_of_end_effector * 0.5 * np.array(last_dir_of_motion)
                        point_on_edge = (self.current_position[0] + ee_offset_vector[0], self.current_position[1] + ee_offset_vector[1])
                        self.rck.internal_points_on_edge[edge_idx_of_interest].append(point_on_edge)
                        rclpy.logging.get_logger("Reasoner").info(f"Updated internal points on edge {edge_idx_of_interest} with point {point_on_edge} based on result of action {self.current_action_type.name}")
                    elif disjunction_id_for_dihedral_270 in result.disjunction_indices:
                        self.rck.dihedrals[edge_idx_of_interest] = 270.0
                        rclpy.logging.get_logger("Reasoner").info(f"Updated dihedral angle of edge {edge_idx_of_interest} to 270 degrees based on result of action {self.current_action_type.name}")

            elif (self.current_action_type == ActionType.MOVE_PARALLEL_FROM_OUTSIDE_TO_EDGE_UNTIL_CONTACT_CCK or
                  self.current_action_type == ActionType.MOVE_PARALLEL_FROM_OUTSIDE_TO_EDGE_UNTIL_CONTACT_CK):
                # point on edge
                
                if motion_spec.direction == Direction.CK:
                    edge_idx_of_interest = next_edge_idx
                elif motion_spec.direction == Direction.CCK:
                    edge_idx_of_interest = prev_edge_idx
                last_desired_velocity = self.desired_velocity
                norm = math.hypot(last_desired_velocity[0], last_desired_velocity[1])
                last_dir_of_motion = (last_desired_velocity[0]/norm, last_desired_velocity[1]/norm) if norm > 0 else (0, 0)

                ee_offset_vector = self.diameter_of_end_effector * 0.5 * np.array(last_dir_of_motion)
                point_on_edge = (self.current_position[0] + ee_offset_vector[0], self.current_position[1] + ee_offset_vector[1])
                self.rck.internal_points_on_edge[edge_idx_of_interest].append(point_on_edge)
                rclpy.logging.get_logger("Reasoner").info(f"Updated internal points on edge {edge_idx_of_interest} with point {point_on_edge} based on result of action {self.current_action_type.name}")

            else:
                self.get_logger().warn(f"No specific update logic for action type {self.current_action_type.name}.")
        
            # propagate knowledge based on new observations
            self._record_knowledge_stage("after_direct_evidence")
            self._propagate_knowledge()
            self._record_knowledge_stage("after_geometric_propagation")
            self._sync_knowledge_to_graph()
            self._save_rck_snapshot_if_configured('action_result_update')
            self.motion_indices_to_collect_points = []
                    
        # Clear collected points after each motion specification execution
        self.collected_points = []

    def _project_ee_position_to_estimated_plane(self, position, orientation):
        """
        Intersect the EE tool z-axis with the estimated plane.

        This corrects stored geometry points when the tool axis is not exactly
        perpendicular to the plane: a z-height error can otherwise appear as an
        x-y error in the edge samples when the orientation is not perfectly aligned.
        """
        
        if not self.plane_slope_estimated:
            return list(position)

        position_np = np.asarray(position, dtype=float)
        
        # since ee pose is measured in the plane frame, plane origin is at (0,0,0) and plane normal is along z axis
        plane_origin_np = np.zeros(3, dtype=float)
        plane_normal = np.array([0.0, 0.0, 1.0], dtype=float)
        
        ee_z_axis = R.from_quat(orientation).as_matrix()[:, 2]

        denom = float(np.dot(plane_normal, ee_z_axis))
        if abs(denom) < 1e-6:
            if self.debug_log:
                print("Skipping EE point projection: tool z-axis is nearly parallel to estimated plane")
            return position_np.tolist()

        distance_along_ee_z = float(np.dot(plane_normal, plane_origin_np - position_np) / denom)
        projected_position = position_np + distance_along_ee_z * ee_z_axis

        if self.debug_log:
            print(f"Projected EE point from {position_np.tolist()} to {projected_position.tolist()}")

        return projected_position.tolist()

    def _ee_position_in_distance_tracking_frame(self, msg: PoseStamped):
        """
        Return the raw EE callback position expressed in the configured distance frame.
        """
        source_frame = msg.header.frame_id or ""
        position = [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ]

        if source_frame == self.ee_distance_tracking_frame_id:
            return position, self.ee_distance_tracking_frame_id

        if not source_frame.startswith("marker_frame"):
            return None, None

        # i.e., if source frame is not in tracking frame (base_link)
        # and still plane details are not known, then we cannot transform
        if self.plane_origin_position is None or self.plane_orientation is None:
            return None, None

        position_in_marker_0 = position
        if source_frame != "marker_frame_0":
            if (
                self.current_edge_of_interest_origin is None or
                self.current_edge_of_interest_orientation is None
            ):
                return None, None

            # transform the point to marker_frame_0 (frame of the plane of interest)
            position_in_marker_0 = Util.transform_points_local_to_global(
                points=[position],
                frame_position_wrt_global=self.current_edge_of_interest_origin,
                frame_orientation_wrt_global=self.current_edge_of_interest_orientation,
            )[0]

        # if source frame starts with marker_frame and it is marker_frame_0
        position_in_tracking_frame = Util.transform_points_local_to_global(
            points=[position_in_marker_0],
            frame_position_wrt_global=self.plane_origin_position,
            frame_orientation_wrt_global=self.plane_orientation,
        )[0]
        return [float(value) for value in position_in_tracking_frame], self.ee_distance_tracking_frame_id

    def _record_ee_distance_sample(self, msg: PoseStamped):
        """
        Accumulate EE path length from consecutive callback poses in one frame.
        """
        try:
            position, frame_id = self._ee_position_in_distance_tracking_frame(msg)
        except Exception as exc:
            self.ee_distance_skipped_sample_count += 1
            if self.debug_log:
                self.get_logger().warn(f"Skipping EE distance sample: {exc}")
            return

        if position is None or frame_id is None:
            self.ee_distance_skipped_sample_count += 1
            return

        position_np = np.asarray(position, dtype=float)
        if not np.all(np.isfinite(position_np)):
            self.ee_distance_skipped_sample_count += 1
            return

        if self.ee_distance_last_position is None:
            self.ee_distance_last_position = position_np
            self.ee_distance_last_frame_id = frame_id
            self.ee_distance_sample_count += 1
            return

        if frame_id != self.ee_distance_last_frame_id:
            self.ee_distance_skipped_frame_mismatch_count += 1
            self.ee_distance_last_position = position_np
            self.ee_distance_last_frame_id = frame_id
            self.ee_distance_sample_count += 1
            return

        self.ee_distance_total_m += float(
            np.linalg.norm(position_np - self.ee_distance_last_position)
        )
        self.ee_distance_last_position = position_np
        self.ee_distance_sample_count += 1
    
    def ee_callback(self, msg: PoseStamped):
        """
        Callback for end-effector pose updates in the frame of eddie_base_link.
        
        Args:
            msg: PoseStamped message with current EE pose
        """
        
        if self.initial_end_effector_pose is None:
            self.initial_end_effector_pose = {
                "frame_id": msg.header.frame_id,
                "position": [
                    float(msg.pose.position.x),
                    float(msg.pose.position.y),
                    float(msg.pose.position.z),
                ],
                "orientation_xyzw": [
                    float(msg.pose.orientation.x),
                    float(msg.pose.orientation.y),
                    float(msg.pose.orientation.z),
                    float(msg.pose.orientation.w),
                ],
            }
        self.first_state_update_received = True
        self._record_ee_distance_sample(msg)
        
        self.ee_pose_frame_id = msg.header.frame_id        
        self.current_position = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        self.current_orientation = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]

        # transform point from edge frame to plane frame
        if (self.ee_pose_frame_id.startswith("marker_frame") and 
            self.plane_slope_estimated):
            if self.ee_pose_frame_id != "marker_frame_0":
                if self.debug_log: print(f"Transforming current pose from frame {self.ee_pose_frame_id} to global frame using plane origin and orientation")
                if self.debug_log: print(f"Current position before transformation: {self.current_position}, Current orientation before transformation: {self.current_orientation}")
                if self.debug_log: print(f"Current edge of interest origin: {self.current_edge_of_interest_origin}, Current edge of interest orientation: {self.current_edge_of_interest_orientation}")
                self.current_position = Util.transform_points_local_to_global(
                                        points=[self.current_position],
                                        frame_position_wrt_global=self.current_edge_of_interest_origin,
                                        frame_orientation_wrt_global=self.current_edge_of_interest_orientation
                                        )[0]
                self.current_orientation = Util.transform_quaternion_local_to_global(
                    quaternion_local=self.current_orientation,
                    frame_orientation_wrt_global=self.current_edge_of_interest_orientation
                )
                
            else:
                if self.debug_log: print(f"The current pose is not transformed. Collected point is in frame {self.ee_pose_frame_id} ")
        
            self.current_position = self._project_ee_position_to_estimated_plane(
                self.current_position,
                self.current_orientation
            )
        else:
            if self.debug_log: print(f"The current pose is not transformed. Either plane slope is not estimated yet or current marker frame {self.ee_pose_frame_id} does not start with 'marker_frame' ")
        
        current_motion_idx = self.next_action_idx - 1
        if (current_motion_idx in self.motion_indices_to_collect_points or
            self.collect_points_on_edge_bool):
            if self.collect_points_on_edge_bool:
                if self.debug_log: print(f"Collecting point {self.current_position}")
            self.collected_points.append(self.current_position)

    def create_and_publish_marker_pose(self, 
                                       marker_type="marker", 
                                       frame=None, 
                                       position=None, 
                                       orientation=None):

        pose_stamped = PoseStamped()

        if position is not None:
            # if position is list of two elements, set z as zero
            if len(position) == 2:
                position = list(position) + [0.0]
            x, y, z = position
        else:
            x, y, z = self.current_position


        if orientation is None:
            qx, qy, qz, qw = self.current_orientation
        else:
            qx, qy, qz, qw = orientation

        # Set header
        pose_stamped.header.stamp = self.get_clock().now().to_msg()

        if frame:
            pose_stamped.header.frame_id = frame
        else:
            pose_stamped.header.frame_id = self.current_ms_frame

        # Set position
        pose_stamped.pose.position.x = x
        pose_stamped.pose.position.y = y
        #pose_stamped.pose.position.z = z

        if self.current_ms_frame == "marker_frame_0":
            pose_stamped.pose.position.z = 0.0
        else:
            pose_stamped.pose.position.z = z

        # Set orientation
        pose_stamped.pose.orientation.x = qx
        pose_stamped.pose.orientation.y = qy
        pose_stamped.pose.orientation.z = qz
        pose_stamped.pose.orientation.w = qw
        
        # convert quaternion to euler angles for logging
        r = R.from_quat([qx, qy, qz, qw])
        euler_angles = r.as_euler('zyx', degrees=True)


        if marker_type == "marker":
            self.current_marker_id += 1
            self.publisher_marker.publish(pose_stamped)
            print(f"SENDING MARKER {self.current_marker_id} at position {position} and orientation {euler_angles} in frame {pose_stamped.header.frame_id}")
        elif marker_type == "corner":
            self.publisher_corner.publish(pose_stamped)
            print(f"SENDING CORNER at position {position} and orientation {euler_angles} in frame {pose_stamped.header.frame_id}")
        elif marker_type == "plane":
            self.publisher_plane.publish(pose_stamped)
            print(f"SENDING PLANE at position {position} and orientation {euler_angles} in frame {pose_stamped.header.frame_id}")
        return

    def destroy_node(self):
        """
        Properly clean up ROS 2 resources including ActionClient and subscriptions.
        This prevents the "Maximum number of clients reached" error for graph visualization.
        """
        try:
            # Destroy ActionClient if it exists
            if hasattr(self, 'client') and self.client is not None:
                # Wait for any pending operations to complete
                if hasattr(self.client, '_goal_future') and self.client._goal_future is not None:
                    try:
                        self.client._goal_future.result(timeout_sec=0.5)
                    except Exception:
                        pass
                self.client = None
        except Exception as e:
            self.get_logger().warn(f"Error destroying ActionClient: {e}")
        
        # Call parent's destroy_node to clean up other ROS resources
        super().destroy_node()


def main(args=None):
    """
    Main entry point for the Reasoner Node.
    
    Args:
        args: Command line arguments
    """
    rclpy.init(args=args)
    
    reasoner_node = ReasonerNode()
    
    try:
        rclpy.spin(reasoner_node)
    except KeyboardInterrupt:
        reasoner_node.get_logger().info("Shutdown requested")
    finally:
        reasoner_node.destroy_node()
        try:
            rclpy.shutdown()
        except RuntimeError:
            # Context already shut down by node destruction
            pass


if __name__ == '__main__':
    main()
