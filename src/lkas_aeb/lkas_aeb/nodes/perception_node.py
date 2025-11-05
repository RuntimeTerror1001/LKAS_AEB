#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from carla_msgs.msg import CarlaEgoVehicleStatus
from lkas_aeb_msgs.msg import Obstacle, ObstacleArray, LaneInfo

from lkas_aeb.modules.perception.lane_detector import LaneDetector
from lkas_aeb.modules.perception.obstacle_detector import ObstacleDetector
from lkas_aeb.util.helpers import load_params

import os
from ament_index_python.packages import get_package_share_directory 

"""
PERCEPTION NODE
"""

class PerceptionNode(Node):
    """
    ROS2 Node for perception processing that combines lane detection and obstacle detection.
    
    Subscribes to:
        - /carla/hero/rgb_front/image: Camera image feed
        - /carla/hero/vehicle_status: Vehicle speed information
    
    Publishes:
        - /perception/lane_markers: Annotated lane detection image
        - /perception/obstacles: Annotated obstacle detection image
        - /perception/lane_info: Lane detection data (center, curvature, etc.)
        - /perception/obstacles_info: Obstacle detection data (positions, distances, etc.)
    """

    def __init__(self):
        super().__init__('perception_node')

        # ========================
        # PARAMETER LOADING
        # ========================
        # Load configuration file paths
        package_path = get_package_share_directory('lkas_aeb')
        lkas_params_path = os.path.join(package_path, 'config', 'params', 'lkas_params.yaml')
        aeb_params_path = os.path.join(package_path, 'config', 'params', 'aeb_params.yaml')

        # Declare ROS parameters
        self.declare_parameter('lkas_params_path', lkas_params_path)
        self.declare_parameter('aeb_params_path', aeb_params_path)
        
        # Load parameters from YAML files
        lkas_params = load_params(self.get_parameter('lkas_params_path').value, logger=self.get_logger())
        aeb_params = load_params(self.get_parameter('aeb_params_path').value, logger=self.get_logger())

        # Extract configuration sections
        lane_params = lkas_params['lane_detection']
        perception_params = aeb_params['perception']

        # ========================
        # STATE VARIABLES
        # ========================
        self.curr_vehicle_speed = 0.0
        self.last_image_time = None

        # ========================
        # DETECTOR INITIALIZATION
        # ========================
        self.lane_detector = LaneDetector({'lane_detection':lane_params})
        self.obstacle_detector = ObstacleDetector({'perception':perception_params})

        # ========================
        # ROS COMMUNICATION SETUP
        # ========================
        # Subscribers
        self.speed_sub = self.create_subscription(
            CarlaEgoVehicleStatus, '/carla/hero/vehicle_status', self.speed_cb, 10
        )
        self.cam_sub = self.create_subscription(
            Image, '/carla/hero/rgb_front/image', self.camera_cb, 10
        )

        # Publishers
        self.lane_img_pub = self.create_publisher(
            Image, '/perception/lane_markers', 10
        )
        self.obstacle_img_pub = self.create_publisher(
            Image, '/perception/obstacles', 10
        )
        self.lane_info_pub = self.create_publisher(
            LaneInfo, '/perception/lane_info', 10
        )
        self.obstacles_pub = self.create_publisher(
            ObstacleArray, '/perception/obstacles_info', 10
        )

        # OpenCV Bridge for image conversion
        self.bridge = CvBridge()
        self.get_logger().info("Perception Node Initialized.")

    def speed_cb(self, msg):
        """
        Callback for vehicle speed updates.
        
        Args:
            msg (CarlaEgoVehicleStatus): Vehicle status message containing speed
        """
        self.curr_vehicle_speed = msg.velocity

    def camera_cb(self, msg):
        """
        Main callback for processing camera images. Performs lane and obstacle detection.
        
        Args:
            msg (Image): ROS Image message from camera
        """
        try:    
            # ========================
            # IMAGE PREPROCESSING
            # ========================
            
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            curr_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9

            # ========================
            # LANE DETECTION PROCESSING
            # ========================
            detection_res  = self.lane_detector.detect_lanes(cv_image)
            
            # Handle different result formats for backward compatibility
            if len(detection_res) == 7:  # New format with confidence and distance
                lane_img, lane_center, left_curv, right_curv, lane_width, confidence, distance = detection_res
            else:  # Backward compatibility with old format
                lane_img, lane_center, left_curv, right_curv, lane_width = detection_res
                confidence = 50.0  # Default confidence
                distance = 15.0   # Default distance

            # Publish annotated lane detection image
            self.lane_img_pub.publish(self.bridge.cv2_to_imgmsg(lane_img, 'bgr8'))

            # ========================
            # LANE INFO MESSAGE CREATION
            # ========================
            lane_info = LaneInfo()
            lane_info.header.stamp = msg.header.stamp
            lane_info.header.frame_id = msg.header.frame_id

            # Set lane center coordinates if detected
            if lane_center:
                lane_info.center_x = float(lane_center[0])
                lane_info.center_y = float(lane_center[1])
                lane_info.detected = True
            else:
                lane_info.detected = False
            
            # Set lane geometry parameters
            lane_info.curvature_left = float(left_curv) if left_curv is not None else 0.0
            lane_info.curvature_right = float(right_curv) if right_curv is not None else 0.0
            lane_info.lane_width = float(lane_width) if lane_width is not None else 0.0
            lane_info.confidence = float(confidence)
            lane_info.vehicle_speed = float(self.curr_vehicle_speed)
            lane_info.distance = float(distance)

            # Calculate lateral offset from lane center
            if lane_info.detected:
                image_center = cv_image.shape[1] / 2
                lateral_offset = abs(lane_info.center_x - image_center)
                # Convert pixel offset to meters (rough estimate)
                if lane_width > 0:
                    lateral_offset_meters = (lateral_offset / cv_image.shape[1]) * lane_width
                    lane_info.lateral_offset = float(lateral_offset_meters)
                else:
                    lane_info.lateral_offset = 0.0
            else:
                lane_info.lateral_offset = 0.0

            self.lane_info_pub.publish(lane_info)

            # ========================
            # OBSTACLE DETECTION PROCESSING
            # ========================
            stamp = msg.header.stamp
            obstacle_img, obstacles = self.obstacle_detector.detect(cv_image, stamp)
            
            # Publish annotated obstacle image
            self.obstacle_img_pub.publish(self.bridge.cv2_to_imgmsg(obstacle_img, 'bgr8'))

            # ========================
            # OBSTACLE INFO ARRAY MESSAGE CREATION
            # ========================
            obstacle_array = ObstacleArray()
            obstacle_array.header.stamp = stamp
            obstacle_array.header.frame_id = 'camera'

            # Process each detected obstacle
            for obs in obstacles:
                # Obstacle : [x1, y1, x2, y2, dist, speed, cls_id, track_id]
                obstacle = Obstacle()
                obstacle.class_id = int(obs[6])
                obstacle.distance = float(obs[4])
                obstacle.speed = float(obs[5]) if obs[5] is not None else 0.0
                obstacle.bbox = [int(obs[0]), int(obs[1]), int(obs[2]), int(obs[3])]
                obstacle.relative_speed = obstacle.speed - self.curr_vehicle_speed
                obstacle.track_id = int(obs[7])
                obstacle_array.obstacles.append(obstacle)
            
            self.obstacles_pub.publish(obstacle_array)

            # ========================
            # PERFORMANCE MONITORING
            # ========================
            # if self.last_image_time is not None:
            #     processing_time = curr_time - self.last_image_time
            #     if processing_time > 0.1:  # Log if processing takes more than 100ms
            #        self.get_logger().warn(f'Slow processing: {processing_time:.3f}s')
            
            self.last_image_time = curr_time

        except Exception as e:
            self.get_logger().error(f'Processing Error: {str(e)}')

# ========================
# MAIN FUNCTION
# ========================

def main(args=None):
    """
    Main function to initialize and run the perception node.
    
    Args:
        args: Command line arguments (optional)
    """
    rclpy.init(args=args)
    node = PerceptionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()