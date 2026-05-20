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
                      polygon_id: str = "polygon_0") -> dict:
        """
        Convert PolygonKnowledge to Graph JSON format suitable for create_graph_from_json().
        
        Architecture:
        - Nodes reference data via string IDs (e.g., "pt_0_x", "line_segment_0_length")
        - Data_structure contains actual values indexed by these IDs
        - This separation allows the graph module to update values independently
        
        :param pk: PolygonKnowledge instance
        :param frame_name: Reference frame name
        :param polygon_id: ID for the polygon node

        :return: JSON-compatible dict with frame, nodes, data_structure, and edges
        """
        n_sides = pk.n_sides
        nodes = []
        data_structure = []
        edges = []
        
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
        
        return {
            "frame": {
                "name": frame_name
            },
            "nodes": nodes,
            "data_structure": data_structure,
            "edges": edges
        }

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
        self.dof = None
        self.stop_execution = False
        self.last_motion_progress_log_time = 0.0
        
        # ====== Visualization Control ======
        # Set to False to suppress graph visualization (helps in headless/non-interactive environments)
        self.enable_graph_visualization = False
        
        # ====== Knowledge Structures ======
        self.experiment_id = self.config['experiment']['id']
        self.n_sides = self.config['experiment']['n_sides']
        
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
        self.step_count = 0
        
        # ====== Graph Representation ======
        # Create initial belief state graph from rck using automatic JSON generation
        self.rck_json = PolygonKnowledgeToGraphConverter.polygon_knw_to_json(self.rck, frame_name="robot_frame_0")
        self.rpk_json = PolygonKnowledgeToGraphConverter.polygon_knw_to_json(self.rpk, frame_name="object_frame_0", polygon_id="polygon_0")
        
        self.rpk_graph = Graph.create_graph_from_json(self.rpk_json)             # Prior knowledge graph (static)
        self.rck_graph = Graph.create_simplified_graph_from_json(self.rck_json)  # Current knowledge graph (belief, updated after each action)
        
        # ====== State Variables ======
        self.state_of_execution = Util.StateOfExecution.IDLE
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
        self.touch_velocity = self.config['motion']['touch_velocity']
        self.slide_velocity = self.config['motion']['slide_velocity']
        self.slide_offset_from_edge = self.config['motion']['slide_offset_from_edge']
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
        self.current_action_type = None
        
        # ====== Logging Setup ======
        self.logs_dir = os.path.join(os.path.dirname(__file__), 'rck_logs')
        os.makedirs(self.logs_dir, exist_ok=True)
        self.rck_json_path = os.path.join(self.logs_dir, 'rck_knowledge.json')
        self.rck_history_dir = os.path.join(self.logs_dir, 'rck_history')
        os.makedirs(self.rck_history_dir, exist_ok=True)
        
        # ====== Initialization ======
        
        # Sync rck with graph
        self._sync_knowledge_to_graph()
        
        # Save initial rck state
        self._save_rck_to_json()
        
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
    
    def _save_rck_to_json(self):
        """
        Save current knowledge (rck) to JSON file in logs directory.
        Creates both:
        1. A main file (rck_knowledge.json) that keeps getting updated
        2. Timestamped backup files in rck_history directory for tracking evolution
        """
        try:
            # Generate JSON from current rck
            rck_json = PolygonKnowledgeToGraphConverter.polygon_knw_to_json(
                self.rck, 
                frame_name="robot_frame_0",
                polygon_id="polygon_0"
            )
            
            # Save main file (always updated)
            with open(self.rck_json_path, 'w') as f:
                json.dump(rck_json, f, indent=4)
            
            # Save timestamped backup for history tracking
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            history_file = os.path.join(self.rck_history_dir, f'rck_knowledge_{timestamp}.json')
            with open(history_file, 'w') as f:
                json.dump(rck_json, f, indent=4)
            
            self.get_logger().debug(f"Saved rck to {self.rck_json_path}")
        except Exception as e:
            self.get_logger().warn(f"Error saving rck to JSON: {e}")
        
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
                self.dir_of_sliding_motion_2d = Util.unit_vector_from_points_2d(self.points_on_edge)
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
                force_in_z_direction = self.force_against_surface
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
                    self.desired_orientation, _ = Util.resolve_orientation(dir_of_motion=[1.0, 0.0], orientation_input=Util.OrientationInput.RIGHT_OF_DIR_MOTION)
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
                    self.desired_orientation, _ = Util.resolve_orientation(dir_of_motion=[-1.0, 0.0], orientation_input=Util.OrientationInput.LEFT_OF_DIR_MOTION)
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
                    self.desired_orientation, _ = Util.resolve_orientation(dir_of_motion=[1.0, 0.0], orientation_input=Util.OrientationInput.LEFT_OF_DIR_MOTION)
                    ms_to_execute = Util.make_action_goal_slide(
                        velocity=[self.slide_velocity, None, None],
                        force=[None, -self.force_against_surface, None],
                        orientation=self.desired_orientation,
                        action_name=self.action_name_str,
                        frame_name=self.current_marker_frame_name,
                        time=3.0
                    )
                    self.last_direction_of_motion = [self.dir_of_sliding_motion_2d[0], self.dir_of_sliding_motion_2d[1], 0.0]
                    self.last_direction_of_force_while_sliding_against_edge = [dir_of_force[0], dir_of_force[1], 0.0]
                elif self.current_action_spec.direction == Direction.CK:
                    self.desired_orientation, _ = Util.resolve_orientation(dir_of_motion=[-1.0, 0.0], orientation_input=Util.OrientationInput.RIGHT_OF_DIR_MOTION)
                    ms_to_execute = Util.make_action_goal_slide(
                        velocity=[-self.slide_velocity, None, None],
                        force=[None, -self.force_against_surface, None],
                        orientation=self.desired_orientation,
                        action_name=self.action_name_str,
                        frame_name=self.current_marker_frame_name,
                        time=3.0
                    )
                    self.last_direction_of_motion = [-self.dir_of_sliding_motion_2d[0], -self.dir_of_sliding_motion_2d[1], 0.0]
                    self.last_direction_of_force_while_sliding_against_edge = [dir_of_force[0], dir_of_force[1], 0.0]
                self.send_goal(ms_to_execute)
    
    def main_loop(self):
        """
        Main reasoning loop executed on timer.
        """
        # exit condition
        if self.dof == 0:
            self.exploration_complete = True
            self.get_logger().info("Exploration complete - all parameters known")
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

                if self.unique_pattern_found_in_rpk and self.unique_pattern_found_in_rck and not self.rpk_rck_matching_idx_found:
                    self.get_logger().info(f"Unique_pattern_found_in_rpk: {self.unique_pattern_found_in_rpk}, unique_pattern_found_in_rck: {self.unique_pattern_found_in_rck}")
                    self.get_logger().info("Attempting to match rck with rpk...")
                    self.rpk_rck_matching_idx_found, self.rpk_first_idx_in_rck = get_unique_pattern_ref_index(self.rck, self.rpk)
                if not self.rpk_rck_matching_idx_found:
                    if self.corner_coordinates_available_in_rpk:
                        print("Attempting to match rck with rpk using corner coordinates...")
                        self.rpk_rck_matching_idx_found, self.rpk_first_idx_in_rck = get_unique_pattern_ref_index(self.rck, self.rpk, match_corner_coordinates=True)
                    else:
                        print("Attempting to find unique pattern in individual parameters...")
                        self.rpk_rck_matching_idx_found, self.rpk_first_idx_in_rck = get_unique_pattern_ref_index(self.rck, self.rpk, find_match_in_individual_parameters=True)
                
                if self.rpk_rck_matching_idx_found and not self.rck_rearranged:
                    rearrange_rck_using_prior_knowledge(self.rck, self.rpk_first_idx_in_rck)
                    self.marker_id_for_edges[:] = self.marker_id_for_edges[self.rpk_first_idx_in_rck:] + self.marker_id_for_edges[:self.rpk_first_idx_in_rck] # rearrange marker ids in the same way as rck
                    self.rck_rearranged = True
                    fill_missing_parameters(self.rck, self.rpk, self.rpk_rck_matching_idx_found)
                    self._propagate_knowledge(knowledge="rck")
                    self._sync_knowledge_to_graph()
                    # Save updated rck to logs
                    self._save_rck_to_json()
                    
                # find dof after propagation to check if exploration is complete
                self.dof = find_dof(self.rck)
                print(f"[main loop] Degrees of freedom after propagation: {self.dof}")
                
                # Update visualization
                self._sync_knowledge_to_graph()
                # Save updated rck to logs
                self._save_rck_to_json()
    
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
            self.current_action_type, self.current_ref_edge_index = next_action(self.rck, self.prev_action_instance, self.rck_rearranged)
            self.current_action_spec = ACTION_TO_SPEC[self.current_action_type] if self.current_action_type is not None else None

            # if next_action is None, check if knowledge is complete
            if self.current_action_type is None:
                self.get_logger().info("No action recommendation from action selection algorithm. Checking if knowledge is complete.")
                self._propagate_knowledge(knowledge="rck") # propagate knowledge before checking dof to fill in any values that can be resolved based on current knowledge
                self._sync_knowledge_to_graph()
                # Save updated rck to logs
                self._save_rck_to_json()
                self.dof = find_dof(self.rck)
                print(f"Degrees of freedom: {self.dof}")
                if self.dof == 0:
                    self.exploration_complete = True
                    self.get_logger().info("Exploration complete - all parameters known")
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
                answer = input("Do you want to continue? (yes/no): ").strip().lower()
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
            
            elif self.current_action_type == ActionType.SLIDE_OVER_SURFACE_PARALLEL_TO_EDGE_CCK:
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
                self.action_name_str = "slide_until_edge"
                action_list = [
                    # move above the surface
                    Util.make_action_goal_move(position=[self.current_position[0], self.current_position[1], self.offset_above_surface], 
                                                orientation = self.current_orientation, 
                                                frame_name="marker_frame_0"),
                    
                    # attain desired yaw
                    Util.make_action_goal_yaw(position=[self.current_position[0], self.current_position[1], self.offset_above_surface], 
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
            
            elif self.current_action_type == ActionType.SLIDE_OVER_SURFACE_PARALLEL_TO_EDGE_CK:
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
                point_to_traverse_in_opp_direction = (point_further_away_in_dir_of_edge[0] + opp_to_perp_direction[0]*0.15, point_further_away_in_dir_of_edge[1] + opp_to_perp_direction[1]*0.15) # point on plane with offset to best_edge
                    
                
                self.desired_orientation, desired_yaw = Util.resolve_orientation(dir_of_motion=edge_uv_ck, orientation_input=self.orientation_input)
                self.desired_velocity = [self.slide_velocity*edge_uv_ck[0], self.slide_velocity*edge_uv_ck[1], None]
                
                self.action_name_str = "slide_until_edge"
                self.last_direction_of_motion = [edge_uv_ck[0], edge_uv_ck[1], 0.0]
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
                    Util.make_action_goal_move(position=[point_further_away_in_dir_of_edge[0], point_further_away_in_dir_of_edge[1], self.offset_above_surface], 
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
                    
                    points_on_edge = self.rck.internal_points_on_edge[self.current_ref_edge_index]
                    if len(points_on_edge) == 0:
                        self.get_logger().warn(f"No points on edge {self.current_ref_edge_index} to determine point for moving parallel to edge until contact. Stopping execution.")
                        self.stop_execution = True
                        return
                    if cw_motion:
                        point_on_edge = points_on_edge[-1]
                        desired_direction_of_motion = edge_uv_ck
                    else:
                        point_on_edge = points_on_edge[0]
                        desired_direction_of_motion = edge_uv
                        
                    point_to_move_to = (point_on_edge[0] + edge_uv[0]*self.slide_offset_from_edge, point_on_edge[1] + edge_uv[1]*self.slide_offset_from_edge) # point further away from edge in direction of edge unit vector
                    self.desired_velocity = [self.slide_velocity*desired_direction_of_motion[0], self.slide_velocity*desired_direction_of_motion[1], 0.0]
                    
                    self.action_name_str = "touch_edge"
                    self.last_direction_of_motion = [desired_direction_of_motion[0], desired_direction_of_motion[1], 0.0]
                    action_list = [
                        Util.make_action_goal_move(position=[point_to_move_to[0], point_to_move_to[1], - self.offset_below_surface],
                                                    orientation = self.current_orientation,
                                                    frame_name="marker_frame_0"),
                        Util.make_action_goal_touch(velocity=self.desired_velocity,
                                                    orientation=self.current_orientation,
                                                    frame_name="marker_frame_0",
                                                    action_name=self.action_name_str)
                    ]
                    self.current_marker_frame_name = "marker_frame_0"
                    return action_list
                
                elif self.prev_action_spec.stop == Stop.UNTIL_EDGE_CONTACT:
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
        
        goal_msg = MotionSpecification.Goal()
        goal_msg.motion_specification = str(motion_specification)
        
        ms_parsed = ast.literal_eval(motion_specification)
        
        self.get_logger().info("Sending goal:\n" + json.dumps(ms_parsed, indent=4))
        
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
            return

        if result.status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("Goal succeeded")
            self.on_action_succeeded(result.result)
            self.state_of_execution = Util.StateOfExecution.COMPLETED
        else:
            self.get_logger().error(f"Goal failed with status {result.status}")
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
            e_uv_before_resampling = Util.unit_vector_from_points_2d(collected_points_2d)
            if self.debug_log: print("edge uv before resampling: ", e_uv_before_resampling)
            if self.debug_log: print("collected points 2d: ", collected_points_2d)
            internal_points_2d = Util.resample_line_points(collected_points_2d)
            if self.debug_log: print("collected points 2d after resampling: ", internal_points_2d)
            radius_of_ee = self.diameter_of_end_effector * 0.5
            
            edge_uv = Util.unit_vector_from_points_2d(internal_points_2d)
            print("edge uv from points: ", edge_uv)
            
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
            elif self.current_action_spec.mode == Mode.AGAINST_VERTICAL:
                if angle_with_direction_of_force < math.pi:
                    self.get_logger().info("Sliding motion detected in the CK direction. Flipping direction to get correct edge unit vector")
                    edge_uv = [-edge_uv[0], -edge_uv[1]] # invert direction to match CCK direction
                elif angle_with_direction_of_force > math.pi:
                    self.get_logger().info("Sliding motion detected in the CCK direction")
                internal_points_2d = Util.offset_points(internal_points_2d, edge_uv, radius_of_ee, side='right')
            # update internal points by taking offset into account and edge unit vector for the current reference edge based on sliding motion
            self.rck.internal_points_on_edge[self.current_ref_edge_index] = [tuple(pt) if isinstance(pt, (list, tuple)) else pt for pt in internal_points_2d]
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
                        self.rck.is_reflexive_angle[edge_idx_of_interest_reflexivity] = True
                        rclpy.logging.get_logger("Reasoner").info(f"Updated angle type of edge {edge_idx_of_interest_reflexivity} to reflexive based on result of action {self.current_action_type.name}")
                    elif disjunction_id_for_edge_non_reflexive in result.disjunction_indices:
                        self.rck.is_reflexive_angle[edge_idx_of_interest_reflexivity] = False
                        rclpy.logging.get_logger("Reasoner").info(f"Updated angle type of edge {edge_idx_of_interest_reflexivity} to non-reflexive based on result of action {self.current_action_type.name}")
            
            # propagate current knowledge
            self._propagate_knowledge(knowledge="rck")

            # check if unique pattern is found for action selection
            self.unique_pattern_found_in_rck = find_unique_pattern(self.rck)

            if self.unique_pattern_found_in_rpk and self.unique_pattern_found_in_rck and not self.rpk_rck_matching_idx_found:
                self.get_logger().info(f"Unique_pattern_found_in_rpk: {self.unique_pattern_found_in_rpk}, unique_pattern_found_in_rck: {self.unique_pattern_found_in_rck}")
                self.get_logger().info("Attempting to match rck with rpk...")
                self.rpk_rck_matching_idx_found, self.rpk_first_idx_in_rck = get_unique_pattern_ref_index(self.rck, self.rpk)
            if not self.rpk_rck_matching_idx_found:
                if self.corner_coordinates_available_in_rpk:
                    print("Attempting to match rck with rpk using corner coordinates...")
                    self.rpk_rck_matching_idx_found, self.rpk_first_idx_in_rck = get_unique_pattern_ref_index(self.rck, self.rpk, match_corner_coordinates=True)
                else:
                    print("Attempting to find unique pattern in individual parameters...")
                    self.rpk_rck_matching_idx_found, self.rpk_first_idx_in_rck = get_unique_pattern_ref_index(self.rck, self.rpk, find_match_in_individual_parameters=True)
            
            if self.rpk_rck_matching_idx_found and not self.rck_rearranged:
                rearrange_rck_using_prior_knowledge(self.rck, self.rpk_first_idx_in_rck)
                self.marker_id_for_edges[:] = self.marker_id_for_edges[self.rpk_first_idx_in_rck:] + self.marker_id_for_edges[:self.rpk_first_idx_in_rck] # rearrange marker ids in the same way as rck
                self.rck_rearranged = True
                fill_missing_parameters(self.rck, self.rpk, self.rpk_rck_matching_idx_found)
                self._propagate_knowledge(knowledge="rck")
                self._sync_knowledge_to_graph()
                # Save updated rck to logs
                self._save_rck_to_json()
                
            # find dof after propagation to check if exploration is complete
            self.dof = find_dof(self.rck)
            print(f"[on action success]Degrees of freedom after propagation: {self.dof}")
            
            # Update visualization
            self._sync_knowledge_to_graph()
            # Save updated rck to logs
            self._save_rck_to_json()
            
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

            ref_edge_idx = self.current_ref_edge_index
            prev_edge_idx = (ref_edge_idx - 1) % self.n_sides
            next_edge_idx = (ref_edge_idx + 1) % self.n_sides
            
            edge_idx_of_interest = ref_edge_idx
            
            if (self.current_action_type in [ActionType.SLIDE_OVER_SURFACE_UNTIL_EDGE,
                                            ActionType.SLIDE_OVER_SURFACE_PERPENDICULAR_TO_EDGE_GIVEN_ONE_POINT,
                                            ActionType.SLIDE_OVER_SURFACE_PARALLEL_FROM_OUTSIDE_TO_EDGE_CCK,
                                            ActionType.SLIDE_OVER_SURFACE_PARALLEL_FROM_OUTSIDE_TO_EDGE_CK,
                                            ActionType.SLIDE_OVER_SURFACE_PARALLEL_TO_EDGE_CCK,
                                            ActionType.SLIDE_OVER_SURFACE_PARALLEL_TO_EDGE_CK]):

                disjunction_id_for_dihedral_90 = 1
                disjunction_id_for_dihedral_270 = 2
                
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
            self._propagate_knowledge()
            self._sync_knowledge_to_graph()
            # Save updated rck to logs
            self._save_rck_to_json()
            self.motion_indices_to_collect_points = []
                    
        # Clear collected points after each motion specification execution
        self.collected_points = []
    
    def ee_callback(self, msg: PoseStamped):
        """
        Callback for end-effector pose updates in the frame of eddie_base_link.
        
        Args:
            msg: PoseStamped message with current EE pose
        """
        
        self.first_state_update_received = True
        
        ee_pose_frame_id = msg.header.frame_id        
        self.current_position = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        self.current_orientation = [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]

        # transform point from edge frame to plane frame if necessary
        if (ee_pose_frame_id.startswith("marker_frame") and 
            self.plane_slope_estimated):
            if ee_pose_frame_id != "marker_frame_0":
                if self.debug_log: print(f"Transforming current pose from frame {ee_pose_frame_id} to global frame using plane origin and orientation")
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
                if self.debug_log: print(f"The current pose is not transformed. Collected point is in frame {ee_pose_frame_id} ")
        else:
            if self.debug_log: print(f"The current pose is not transformed. Either plane slope is not estimated yet or current marker frame {ee_pose_frame_id} does not start with 'marker_frame' ")
        
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


        if marker_type == "marker":
            self.current_marker_id += 1
            self.publisher_marker.publish(pose_stamped)
            print("SENDING MARKER")
        elif marker_type == "corner":
            self.publisher_corner.publish(pose_stamped)
            print("SENDING CORNER")
        elif marker_type == "plane":
            self.publisher_plane.publish(pose_stamped)
            print("SENDING PLANE")
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
                if self.client._goal_future is not None:
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
        rclpy.shutdown()


if __name__ == '__main__':
    main()