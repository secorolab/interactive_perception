import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import rclpy
from rclpy.node import Node
import tf2_ros
import geometry_msgs.msg
from rclpy.time import Time
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import tf2_geometry_msgs  # Import this for transforming geometry_msgs
import math
import random
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from builtin_interfaces.msg import Duration
import mapbox_earcut as earcut
import numpy as np
from aruco_interfaces.msg import ArucoMarkers
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
import geometry_msgs.msg
from geometry_msgs.msg import PointStamped
from tf2_ros import TransformException
from rclpy.duration import Duration
from geometry_msgs.msg import PoseStamped
from tf2_ros import StaticTransformBroadcaster
from scipy.spatial.transform import Rotation as R
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseStamped, PointStamped
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose, do_transform_point
from builtin_interfaces.msg import Time
import csv
import pandas as pd
import numpy as np
import math
from tf_transformations import quaternion_matrix

#Transform all to eddie_base_link

#Transform from: marker_frame_0
#List [position, orientation, frame_id]
estimated_pose = []

#Transform from: marker_frame_0
#List [position, frame_id]
estimated_corners = []

#eddie_base_link
#No need to transform
ground_truth_pose = []

#trandform from: ground_truth_object
#List: [position, frame_id]
ground_truth_corners = []



finished = False


def calculate_translational_error(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 +
                     (p2[1] - p1[1]) ** 2 +
                     (p2[2] - p1[2]) ** 2)



def calculate_rotational_error_quat(q_est, q_true):

    R_true = quaternion_matrix(q_true)
    R_est = quaternion_matrix(q_est)

    R_true = R_true[:3, :3]
    R_est = R_est[:3, :3]

    # Relative rotation
    R_diff = R_true.T @ R_est

    # Trace-based formula
    trace_val = np.trace(R_diff)
    cos_theta = np.clip((trace_val - 1) / 2, -1.0, 1.0)
    e_r = np.arccos(cos_theta)

    # output in radians
    return e_r


def calculate_chamfer_distance(points1, points2):

    points1 = np.array(points1)
    points2 = np.array(points2)

    # Distance from points1 to points2
    dist1 = np.min(np.linalg.norm(points1[:, None, :] - points2[None, :, :], axis=2), axis=1)

    # Distance from points2 to points1
    dist2 = np.min(np.linalg.norm(points2[:, None, :] - points1[None, :, :], axis=2), axis=1)

    cd = (np.mean(dist1) + np.mean(dist2)) / 2
    return cd


def calculate_hausdorff_distance(points1, points2):

    points1 = np.array(points1)
    points2 = np.array(points2)

    # Directed distance from points1 to points2
    dist1 = np.max(np.min(np.linalg.norm(points1[:, None, :] - points2[None, :, :], axis=2), axis=1))

    # Directed distance from points2 to points1
    dist2 = np.max(np.min(np.linalg.norm(points2[:, None, :] - points1[None, :, :], axis=2), axis=1))

    # Symmetric Hausdorff distance
    return max(dist1, dist2)


class Eval_Node(Node):

    def __init__(self):

        global ground_truth
        super().__init__('eval_node')

        self.br = tf2_ros.TransformBroadcaster(self)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.timer = self.create_timer(0.5, self.check)

        self.target_frame = "eddie_base_link"


        self.subscription = self.create_subscription(
            PoseStamped,
            '/corner_position',
            self.corner_callback,
            10
        )

        self.subscription = self.create_subscription(
            PoseStamped,
            '/plane_position',
            self.plane_callback,
            10
        )

        self.subscription = self.create_subscription(
            Point,
            '/ground_truth_corners',
            self.point_callback,
            10
        )

        self.subscription = self.create_subscription(
            PoseStamped,
            '/ground_truth_centroid',
            self.centroid_callback,
            10
        )

    def corner_callback(self, msg):
        position = msg.pose.position
        orientation = msg.pose.orientation
        frame_id = msg.header.frame_id
        estimated_corners.append([position, frame_id])

    def plane_callback(self, msg):
        position = msg.pose.position
        orientation = msg.pose.orientation
        frame_id = msg.header.frame_id
        estimated_pose.append([position, orientation, frame_id])

    def centroid_callback(self, msg):
        position = msg.pose.position
        orientation = msg.pose.orientation
        frame_id = msg.header.frame_id
        ground_truth_pose.append([position, orientation, frame_id])

    def point_callback(self, msg):
        position = msg
        ground_truth_corners.append(position)

    def transform_all(self):
        """Transform all available global variables to eddie_base_link."""
        global estimated_pose, estimated_corners, ground_truth_corners

        # Transform estimated_pose
        if estimated_pose:
            try:
                pose_point, pose_quat, src_frame = estimated_pose[0]

                position = [pose_point.x, pose_point.y, pose_point.z]
                orientation = [pose_quat.x, pose_quat.y, pose_quat.z, pose_quat.w]

                # Must pass three elements to match transform_pose_list
                transformed = self.transform_pose_list([position, orientation], src_frame)
                estimated_pose = transformed
                self.get_logger().info(f'Transformed estimated_pose ({src_frame}) -> eddie_base_link')

            except Exception as e:
                self.get_logger().warn(f'Failed to transform estimated_pose: {e}')
        # Transform estimated_corners
        if estimated_corners:
            try:
                transformed_corners = []
                for corner in estimated_corners:
                    # Each corner looks like [geometry_msgs.msg.Point, 'marker_frame_0']
                    point, src_frame = corner
                    position = [point.x, point.y, point.z]
                    transformed = self.transform_point_list([position, src_frame], src_frame)
                    transformed_corners.append(transformed)

                estimated_corners = transformed_corners
                self.get_logger().info(f'Transformed estimated_corners ({src_frame}) -> eddie_base_link')

            except Exception as e:
                self.get_logger().warn(f'Failed to transform estimated_corners: {e}')

        # Transform ground_truth_corners
        if ground_truth_corners:
            try:
                src_frame = 'ground_truth_object'
                transformed_corners = []
                for point in ground_truth_corners:
                    position = [point.x, point.y, point.z]
                    transformed = self.transform_point_list([position, src_frame], src_frame)
                    transformed_corners.append(transformed)

                ground_truth_corners = transformed_corners
                self.get_logger().info(f'Transformed ground_truth_corners ({src_frame}) -> eddie_base_link')

            except Exception as e:
                self.get_logger().warn(f'Failed to transform ground_truth_corners: {e}')

    def transform_pose_list(self, pose_list, source_frame):
        """Transform [position, orientation, frame_id] list where position/orientation are lists of floats."""
        position, orientation = pose_list

        ps = PoseStamped()
        ps.header.frame_id = source_frame
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = position
        ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = orientation

        print(ps.pose.position.x)

        transform = self.tf_buffer.lookup_transform(
            self.target_frame,
            source_frame,
            rclpy.time.Time(seconds=0)
        )

        new_ps = do_transform_pose(ps.pose, transform)

        ps = PoseStamped()
        ps.header.frame_id = self.target_frame
        ps.pose = new_ps

        new_position = [
            ps.pose.position.x,
            ps.pose.position.y,
            ps.pose.position.z
        ]
        new_orientation = [
            ps.pose.orientation.x,
            ps.pose.orientation.y,
            ps.pose.orientation.z,
            ps.pose.orientation.w
        ]

        print(ps)
        return [new_position, new_orientation, self.target_frame]

    def transform_point_list(self, point_list, source_frame):
        """Transform [position, frame_id] list."""
        position, _ = point_list
        ps = PointStamped()
        ps.header.frame_id = source_frame
        ps.point.x, ps.point.y, ps.point.z = position

        transform = self.tf_buffer.lookup_transform(
            self.target_frame,
            source_frame,
            rclpy.time.Time()
        )
        new_ps = do_transform_point(ps, transform)
        new_position = [new_ps.point.x, new_ps.point.y, new_ps.point.z]
        return [new_position, self.target_frame]



    def check(self):
        global estimated_corners, estimated_pose, ground_truth_pose, ground_truth_corners, finished

        #All parameters have been detected
        if len(estimated_corners) == 4 and estimated_pose and ground_truth_pose and len(ground_truth_corners) == 4 and finished is False:

            print("Conditions met transforming all variables")

            self.transform_all()

            estimated_corners = [
                [corner[0][0], corner[0][1], corner[0][2]]
                for corner in estimated_corners
            ]

            estimated_position = [estimated_pose[0][0], estimated_pose[0][1], estimated_pose[0][2]]
            estimated_quaternion = [estimated_pose[1][0], estimated_pose[1][1], estimated_pose[1][2], estimated_pose[1][3]]
            estimated_pose = [estimated_position, estimated_quaternion]
            ground_truth_corners = [
                [corner[0][0], corner[0][1], corner[0][2]]
                for corner in ground_truth_corners
            ]
            ground_truth_position = [ground_truth_pose[0][0].x, ground_truth_pose[0][0].y, ground_truth_pose[0][0].z]
            ground_truth_quaternion = [ground_truth_pose[0][1].x, ground_truth_pose[0][1].y, ground_truth_pose[0][1].z, ground_truth_pose[0][1].w]
            ground_truth_pose = [ground_truth_position, ground_truth_quaternion]


            translational_error = calculate_translational_error(estimated_position, ground_truth_position)
            rotational_error = calculate_rotational_error_quat(estimated_quaternion, ground_truth_quaternion)
            hausdorff_distance = calculate_hausdorff_distance(estimated_corners, ground_truth_corners)
            chamfer_distance = calculate_chamfer_distance(estimated_corners, ground_truth_corners)


            output_path = "results_experiment_1_run_1.csv"

            results = [
                ("Time ", 0.0),
                ("Number of contact points ", 0.0),
                ("Number of motion primitives performed ", 0.0),
                ("Estimated corners ", estimated_corners),
                ("Estimated pose ", estimated_pose),
                ("Ground truth pose ", ground_truth_pose),
                ("Ground truth corners ", ground_truth_corners),
                ("Translational error ", translational_error),
                ("Rotational error ", rotational_error),
                ("Chamfer distance ", chamfer_distance),
                ("Hausdorff distance ", hausdorff_distance)]


            df = pd.DataFrame(results)

            df = df.rename(columns={
                0: 'Category',
                1: 'Data'
            })

            df.to_csv(output_path, index=False)

            finished = True

            return



def main(args=None):
    rclpy.init(args=args)

    # Create the marker publisher node
    gt_publisher = Eval_Node()

    # Spin the node to keep it alive and publishing
    rclpy.spin(gt_publisher)

    # Clean up on shutdown
    gt_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()