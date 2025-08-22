#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
from carla_msgs.msg import CarlaEgoVehicleStatus #type:ignore
from lkas_aeb_msgs.msg import Obstacle, ObstacleArray, LaneInfo

from lkas_aeb.modules.perception.lane_detector import LaneDetector
from lkas_aeb.modules.perception.obstacle_detector import ObstacleDetector
from lkas_aeb.modules.perception.radar_preproc import rear_radars_to_obstacles
from lkas_aeb.modules.perception.rear_cam_adapter import rear_rgb_to_obstacles
from lkas_aeb.modules.perception.rear_sensor_fusion import fuse_rear
from lkas_aeb.util.helpers import load_params

import os
from ament_index_python.packages import get_package_share_directory 

"""
PERCEPTION NODE
"""

class NewPerceptionNode(Node):
    """
    ROS2 Node for perception processing that combines lane detection and obstacle detection.
    Now includes rear sensor fusion for control gating with PointCloud2 support.
    
    Subscribes to:
        - /carla/hero/rgb_front/image: Camera image feed
        - /carla/hero/vehicle_status: Vehicle speed information
        - /carla/hero/radar_rear_left: Left rear radar PointCloud2 measurements
        - /carla/hero/radar_rear_right: Right rear radar PointCloud2 measurements  
        - /carla/hero/rgb_rear/image: Rear camera image feed
    
    Publishes:
        - /perception/lane_markers: Annotated lane detection image
        - /perception/obstacles: Annotated obstacle detection image
        - /perception/lane_info: Lane detection data (center, curvature, etc.)
        - /perception/obstacles_info: Obstacle detection data (positions, distances, etc.)
        - /perception/obstacles_rear_fused: Fused rear obstacle data for control gating
    """

    def __init__(self):
        super().__init__('new_perception_node')

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

        # Store params for sensor fusion modules
        self.params = {
            'lane_detection': lane_params,
            'perception': perception_params,
            # Add radar-specific parameters
            'radar': {
                'radar_min_range': 1.0,
                'radar_max_range': 100.0,
                'radar_min_elevation_deg': -20,
                'radar_max_elevation_deg': 20,
                'radar_min_velocity': 0.5,
                'radar_cluster_eps': 2.0,
                'radar_cluster_min_samples': 2,
                'rear_roi_x_min': -60.0,
                'rear_roi_x_max': 10.0,
                'rear_roi_y_min': -8.0,
                'rear_roi_y_max': 8.0,
            }
        }

        # ========================
        # STATE VARIABLES
        # ========================
        self.curr_vehicle_speed = 0.0
        self.last_image_time = None

        # Rear sensor state cache
        self.radar_left = None
        self.radar_right = None
        self.cam_rear = None
        self.radar_freshness_threshold = 0.1  # 100ms

        # ========================
        # DETECTOR INITIALIZATION
        # ========================
        self.lane_detector = LaneDetector({'lane_detection':lane_params})
        self.obstacle_detector = ObstacleDetector({'perception':perception_params})
        # Rear camera detector (separate instance for rear processing)
        self.rear_detector = ObstacleDetector({'perception':perception_params})

        # ========================
        # ROS COMMUNICATION SETUP
        # ========================
        # Existing subscribers
        self.speed_sub = self.create_subscription(
            CarlaEgoVehicleStatus, '/carla/hero/vehicle_status', self.speed_cb, 10
        )
        self.cam_sub = self.create_subscription(
            Image, '/carla/hero/rgb_front/image', self.camera_cb, 10
        )

        # Updated rear sensor subscribers for PointCloud2
        self.radar_left_sub = self.create_subscription(
            PointCloud2, '/carla/hero/radar_rear_left', self.radar_left_cb, 10
        )
        self.radar_right_sub = self.create_subscription(
            PointCloud2, '/carla/hero/radar_rear_right', self.radar_right_cb, 10
        )
        self.rear_cam_sub = self.create_subscription(
            Image, '/carla/hero/rgb_rear/image', self.rear_camera_cb, 10
        )

        # Existing publishers (unchanged)
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

        # New rear fusion publisher
        self.pub_rear_fused = self.create_publisher(
            ObstacleArray, '/perception/obstacles_rear_fused', 10
        )

        # OpenCV Bridge for image conversion
        self.bridge = CvBridge()
        self.get_logger().info("Perception Node Initialized with PointCloud2 radar support.")

    def speed_cb(self, msg):
        """
        Callback for vehicle speed updates.
        
        Args:
            msg (CarlaEgoVehicleStatus): Vehicle status message containing speed
        """
        self.curr_vehicle_speed = msg.velocity

    def radar_left_cb(self, msg):
        """
        Callback for left rear radar PointCloud2 measurements.
        
        Args:
            msg (PointCloud2): Left rear radar point cloud
        """
        self.radar_left = msg
        self._try_rear_fusion()

    def radar_right_cb(self, msg):
        """
        Callback for right rear radar PointCloud2 measurements.
        
        Args:
            msg (PointCloud2): Right rear radar point cloud
        """
        self.radar_right = msg
        self._try_rear_fusion()

    def rear_camera_cb(self, msg):
        """
        Callback for rear camera images. Processes obstacles and updates cache.
        
        Args:
            msg (Image): ROS Image message from rear camera
        """
        try:        
            # Process rear camera obstacles
            self.cam_rear = rear_rgb_to_obstacles(msg, self.rear_detector, self.params)
            
            # Try fusion if radar data is available
            self._try_rear_fusion()
            
        except Exception as e:
            self.get_logger().error(f'Rear Camera Processing Error: {str(e)}')

    def _try_rear_fusion(self):
        """
        Attempt to perform rear sensor fusion if both radar measurements are fresh.
        """
        try:
            # Check if we have both radar measurements
            if self.radar_left is None or self.radar_right is None:
                return
            
            # Check radar freshness (within 100ms)
            current_time = self.get_clock().now()
            left_time = rclpy.time.Time.from_msg(self.radar_left.header.stamp)
            right_time = rclpy.time.Time.from_msg(self.radar_right.header.stamp)
            
            left_age = (current_time - left_time).nanoseconds / 1e9
            right_age = (current_time - right_time).nanoseconds / 1e9
            
            if left_age > self.radar_freshness_threshold or right_age > self.radar_freshness_threshold:
                return
            
            # Process radar data using updated function
            radar_params = {**self.params.get('radar', {}), **self.params.get('perception', {})}
            radar_rear = rear_radars_to_obstacles(self.radar_left, self.radar_right, radar_params)
            
            # Perform fusion with camera data (if available)
            fusion_params = {**self.params.get('perception', {}), **self.params.get('radar', {})}
            fused_rear = fuse_rear(radar_rear, self.cam_rear, fusion_params)
            
            # Update relative speeds based on current vehicle speed
            for obstacle in fused_rear.obstacles:
                if obstacle.speed != 0.0:
                    obstacle.relative_speed = obstacle.speed - self.curr_vehicle_speed
                elif obstacle.relative_speed != 0.0:
                    # If we only have relative speed from radar Doppler
                    obstacle.speed = obstacle.relative_speed + self.curr_vehicle_speed
            
            # Publish fused rear obstacles
            self.pub_rear_fused.publish(fused_rear)
            
            # Log fusion statistics
            radar_count = len(radar_rear.obstacles)
            camera_count = len(self.cam_rear.obstacles) if self.cam_rear else 0
            fused_count = len(fused_rear.obstacles)
            
            if radar_count > 0 or camera_count > 0:
                self.get_logger().debug(
                    f'Rear Fusion: Radar={radar_count}, Camera={camera_count}, Fused={fused_count}'
                )
            
        except Exception as e:
            self.get_logger().error(f'Rear Fusion Error: {str(e)}')

    def camera_cb(self, msg):
        """
        Main callback for processing camera images. Performs lane and obstacle detection.
        """
        try:
            # ========================
            # IMAGE PREPROCESSING
            # ========================
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            curr_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9

            # ========================
            # LANE DETECTION
            # ========================
            detection_res = self.lane_detector.detect_lanes(cv_image)

            # Handle both old/new detector return formats
            if len(detection_res) == 7:
                lane_img, lane_center, left_curv, right_curv, lane_width, confidence, distance = detection_res
            else:
                lane_img, lane_center, left_curv, right_curv, lane_width = detection_res[:5]
                confidence = detection_res[5] if len(detection_res) > 5 else 0.8
                distance   = detection_res[6] if len(detection_res) > 6 else 15.0

            # Publish annotated lane image
            self.lane_img_pub.publish(self.bridge.cv2_to_imgmsg(lane_img, 'bgr8'))

            # LaneInfo message
            lane_info = LaneInfo()
            lane_info.header.stamp = msg.header.stamp
            lane_info.header.frame_id = msg.header.frame_id

            if lane_center:
                lane_info.center_x = float(lane_center[0])
                lane_info.center_y = float(lane_center[1])
                lane_info.detected  = True
            else:
                lane_info.detected  = False

            lane_info.curvature_left  = float(left_curv)  if left_curv  is not None else 0.0
            lane_info.curvature_right = float(right_curv) if right_curv is not None else 0.0
            lane_info.lane_width      = float(lane_width) if lane_width is not None else 0.0
            lane_info.confidence      = float(confidence)
            lane_info.vehicle_speed   = float(self.curr_vehicle_speed)
            lane_info.distance        = float(distance)

            # Rough lateral offset estimate (px→m via lane width)
            if lane_info.detected and lane_info.lane_width > 0:
                image_center = cv_image.shape[1] / 2.0
                lateral_px   = abs(lane_info.center_x - image_center)
                lane_info.lateral_offset = float((lateral_px / cv_image.shape[1]) * lane_info.lane_width)
            else:
                lane_info.lateral_offset = 0.0

            self.lane_info_pub.publish(lane_info)

            # ========================
            # FRONT OBSTACLE DETECTION
            # ========================
            stamp = msg.header.stamp
            obstacle_img, obstacles = self.obstacle_detector.detect(cv_image, stamp)

            # Publish annotated obstacle image
            self.obstacle_img_pub.publish(self.bridge.cv2_to_imgmsg(obstacle_img, 'bgr8'))

            # Build ObstacleArray for front camera
            obstacle_array = ObstacleArray()
            obstacle_array.header.stamp = stamp
            obstacle_array.header.frame_id = 'camera'

            # Each item: (x1, y1, x2, y2, distance, speed, class_id, track_id)
            for obs in (obstacles or []):
                o = Obstacle()
                o.class_id = int(obs[6]) if len(obs) > 6 else 0
                o.distance = float(obs[4]) if len(obs) > 4 and obs[4] is not None else -1.0
                o.speed    = float(obs[5]) if len(obs) > 5 and obs[5] is not None else 0.0
                o.relative_speed = o.speed - self.curr_vehicle_speed
                o.track_id = int(obs[7]) if len(obs) > 7 else -1
                o.bbox = [int(obs[0]), int(obs[1]), int(obs[2]), int(obs[3])]

                # Optional, safe defaults for your extended fields
                o.sensor_type     = "camera_front"
                o.confidence      = 0.0
                o.position_3d     = [0.0, 0.0, 0.0]
                o.size_3d         = [0.0, 0.0, 0.0]
                o.track_age       = 0
                o.sensor_sources  = ["camera_front"]
                o.fusion_distance = o.distance
                o.point_count     = 0

                obstacle_array.obstacles.append(o)

            # Publish front obstacle array (keeps AEB compatibility)
            self.obstacles_pub.publish(obstacle_array)

            # ========================
            # PERFORMANCE LOGGING
            # ========================
            if self.last_image_time is not None:
                dt = curr_time - self.last_image_time
                if dt > 0.1:
                    self.get_logger().warn(f"Slow perception processing: {dt:.3f}s")
            self.last_image_time = curr_time

        except Exception as e:
            self.get_logger().error(f'Processing Error: {e}')

# ========================
# MAIN
# ========================

def main(args=None):
    rclpy.init(args=args)
    node = NewPerceptionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
