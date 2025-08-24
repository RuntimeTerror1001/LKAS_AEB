#!/usr/bin/env python3
"""
Refactored Perception Node

Consistent, modular perception pipeline using base classes and common utilities.
Integrates lane detection, obstacle detection, and multi-sensor fusion.
"""

import os
import time
from typing import Dict, Optional
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2
from carla_msgs.msg import CarlaEgoVehicleStatus #type:ignore
from lkas_aeb_msgs.msg import Obstacle, ObstacleArray, LaneInfo
from ament_index_python.packages import get_package_share_directory

# Import perception modules
from lkas_aeb.modules.perception.lane_detector import LaneDetector
from lkas_aeb.modules.perception.obstacle_detector import ObstacleDetector
from lkas_aeb.modules.perception.radar_preproc import RearRadarProcessor
from lkas_aeb.modules.perception.front_cam_adapter import FrontCameraProcessor
from lkas_aeb.modules.perception.rear_cam_adapter import RearCameraProcessor
from lkas_aeb.modules.perception.lidar_preproc import FrontLidarProcessor
from lkas_aeb.modules.perception.front_sensor_fusion import FrontSensorFusion
from lkas_aeb.modules.perception.rear_sensor_fusion import RearSensorFusion

# Import utilities
from lkas_aeb.util.helpers import load_params
from lkas_aeb.util.perception_utils import (
    validate_numeric, ProcessingError, time_function
)


class NewPerceptionNode(Node):
    """
    Refactored perception node with modular, consistent architecture.
    
    Features:
    - Modular sensor processing with consistent interfaces
    - Kalman filter-based tracking for all fusion modules
    - Centralized parameter management
    - Comprehensive error handling and performance monitoring
    - Consistent data validation throughout pipeline
    """
    
    def __init__(self):
        super().__init__('new_perception_node')
        
        # Initialize timing and state
        self._last_update_time = time.time()
        self._processing_stats = {}
        self.current_vehicle_speed = 0.0
        
        # Sensor data cache
        self._sensor_cache = {
            'front_camera': None,
            'front_lidar': None,
            'radar_left': None,
            'radar_right': None,
            'rear_camera': None
        }
        
        # Cache timestamps for freshness checking
        self._cache_timestamps = {key: 0.0 for key in self._sensor_cache.keys()}
        
        try:
            # Load and validate parameters
            self._load_parameters()
            
            # Initialize sensor processors
            self._initialize_processors()
            
            # Setup ROS communication
            self._setup_communication()
            
            self.get_logger().info("Refactored Perception Node initialized successfully")
            
        except Exception as e:
            self.get_logger().error(f"Failed to initialize perception node: {str(e)}")
            raise
    
    def _load_parameters(self) -> None:
        """Load and validate all perception parameters"""
        try:
            # Get package path
            package_path = get_package_share_directory('lkas_aeb')
            
            # Load parameter files
            perception_params_path = os.path.join(package_path, 'config', 'params', 'perception_params.yaml')
            
            # Declare ROS parameters
            self.declare_parameter('perception_params_path', perception_params_path)
            
            # Load parameter files
            self.perception_params = load_params(
                self.get_parameter('perception_params_path').value, 
                logger=self.get_logger()
            )
            
            # Extract sensor timeout
            self.sensor_timeout = self.perception_params.get('common', {}).get('sensor_timeout', 0.2)
            
            self.get_logger().info("Parameters loaded successfully")
            
        except Exception as e:
            self.get_logger().error(f"Failed to load parameters: {str(e)}")
            raise
    
    def _initialize_processors(self) -> None:
        """Initialize all sensor processing modules"""
        try:
            #self.obstacle_detector = ObstacleDetector({'perception': camera_params})
            
            # New modular processors

            # Front Camera Processor
            front_camera_params = self.perception_params.get('front_camera', {})
            self.front_camera_processor = FrontCameraProcessor(front_camera_params)
            
            # Front LiDAR processor
            lidar_params = self.perception_params.get('front_lidar', {})
            self.front_lidar_processor = FrontLidarProcessor(lidar_params)
            
            # Rear radar processor
            radar_params = self.perception_params.get('rear_radar', {})
            self.rear_radar_processor = RearRadarProcessor(radar_params)
            
            # Rear camera processor
            rear_camera_params = self.perception_params.get('rear_camera', {})
            self.rear_camera_processor = RearCameraProcessor(rear_camera_params)
            
            # Fusion modules
            
            # Front sensor fusion
            front_fusion_params = self.perception_params.get('front_fusion', {})
            front_intrinsics = self.perception_params.get('front_camera', {}).get('intrinsics', {})
            front_extrinsics = front_fusion_params.get('extrinsics', {}).get('T_cam_base', np.eye(4).flatten())
            
            self.front_fusion = FrontSensorFusion(
                front_fusion_params,
                front_intrinsics,
                np.array(front_extrinsics).reshape(4, 4)
            )
            
            # Rear sensor fusion
            rear_fusion_params = self.perception_params.get('rear_fusion', {})
            self.rear_fusion = RearSensorFusion(rear_fusion_params)
            
            self.get_logger().info("All sensor processors initialized")
            
        except Exception as e:
            self.get_logger().error(f"Failed to initialize processors: {str(e)}")
            raise
    
    def _setup_communication(self) -> None:
        """Setup ROS publishers and subscribers"""
        try:
            # Subscribers
            self.vehicle_status_sub = self.create_subscription(
                CarlaEgoVehicleStatus, '/carla/hero/vehicle_status',
                self._vehicle_status_callback, 10
            )
            
            self.front_camera_sub = self.create_subscription(
                Image, '/carla/hero/rgb_front/image',
                self._front_camera_callback, 10
            )
            
            self.front_lidar_sub = self.create_subscription(
                PointCloud2, '/carla/hero/lidar',
                self._front_lidar_callback, 10
            )
            
            self.radar_left_sub = self.create_subscription(
                PointCloud2, '/carla/hero/radar_rear_left',
                self._radar_left_callback, 10
            )
            
            self.radar_right_sub = self.create_subscription(
                PointCloud2, '/carla/hero/radar_rear_right',
                self._radar_right_callback, 10
            )
            
            self.rear_camera_sub = self.create_subscription(
                Image, '/carla/hero/rgb_rear/image',
                self._rear_camera_callback, 10
            )
            
            # Publishers
            self.lane_image_pub = self.create_publisher(
                Image, '/perception/lane_markers', 10
            )
            
            self.obstacle_image_pub = self.create_publisher(
                Image, '/perception/obstacles', 10
            )
            
            self.lane_info_pub = self.create_publisher(
                LaneInfo, '/perception/lane_info', 10
            )
            
            self.front_obstacles_pub = self.create_publisher(
                ObstacleArray, '/perception/obstacles_front_fused', 10
            )
            
            self.rear_obstacles_pub = self.create_publisher(
                ObstacleArray, '/perception/obstacles_rear_fused', 10
            )
            
            # Performance monitoring publisher
            self.stats_pub = self.create_publisher(
                ObstacleArray, '/perception/stats', 10  # Reuse ObstacleArray for stats
            )
            
            # OpenCV bridge
            self.cv_bridge = CvBridge()
            
            self.get_logger().info("ROS communication setup complete")
            
        except Exception as e:
            self.get_logger().error(f"Failed to setup communication: {str(e)}")
            raise
    
    # ========================================================================
    # SENSOR CALLBACKS
    # ========================================================================
    
    def _vehicle_status_callback(self, msg: CarlaEgoVehicleStatus) -> None:
        """Handle vehicle status updates"""
        self.current_vehicle_speed = validate_numeric(msg.velocity, 0.0)
    
    @time_function
    def _front_camera_callback(self, msg: Image) -> None:
        """Process front camera image"""
        try:
            self._sensor_cache['front_camera'] = msg
            self._cache_timestamps['front_camera'] = time.time()
            
            # Process camera data
            self._process_front_camera(msg)
            
            # Attempt front fusion if LiDAR data is available
            self._attempt_front_fusion()
            
        except Exception as e:
            self.get_logger().error(f"Front camera processing error: {str(e)}")
    
    @time_function
    def _front_lidar_callback(self, msg: PointCloud2) -> None:
        """Process front LiDAR data"""
        try:
            self._sensor_cache['front_lidar'] = msg
            self._cache_timestamps['front_lidar'] = time.time()
            
            # Attempt front fusion
            self._attempt_front_fusion()
            
        except Exception as e:
            self.get_logger().error(f"Front LiDAR processing error: {str(e)}")
    
    def _radar_left_callback(self, msg: PointCloud2) -> None:
        """Process left rear radar data"""
        try:
            self._sensor_cache['radar_left'] = msg
            self._cache_timestamps['radar_left'] = time.time()
            
            self._attempt_rear_fusion()
            
        except Exception as e:
            self.get_logger().error(f"Rear radar left processing error: {str(e)}")
    
    def _radar_right_callback(self, msg: PointCloud2) -> None:
        """Process right rear radar data"""
        try:
            self._sensor_cache['radar_right'] = msg
            self._cache_timestamps['radar_right'] = time.time()
            
            self._attempt_rear_fusion()
            
        except Exception as e:
            self.get_logger().error(f"Rear radar right processing error: {str(e)}")
    
    def _rear_camera_callback(self, msg: Image) -> None:
        """Process rear camera data"""
        try:
            self._sensor_cache['rear_camera'] = msg
            self._cache_timestamps['rear_camera'] = time.time()
            
            self._attempt_rear_fusion()
            
        except Exception as e:
            self.get_logger().error(f"Rear camera processing error: {str(e)}")
    
    # ========================================================================
    # PROCESSING FUNCTIONS
    # ========================================================================
    
    def _process_front_camera(self, msg: Image) -> None:
        """Process front camera for lane detection and obstacles"""
        try:
            # Convert image
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # Obstacle detection (for fusion)
            self._process_front_obstacle_detection(cv_image, msg.header)
            
        except Exception as e:
            raise ProcessingError(f"Front camera processing failed: {str(e)}")
    
    def _process_front_obstacle_detection(self, cv_image: np.ndarray, header) -> None:
        """Process front camera obstacle detection"""
        try:
            # Detect obstacles
            #obstacle_img, detections = self.obstacle_detector.detect(cv_image, header.stamp)

            obstacle_array = self.front_camera_processor.process(self.cv_bridge.cv2_to_imgmsg(cv_image, 'bgr8'))
            
            # Publish obstacle image
            # obstacle_img_msg = self.cv_bridge.cv2_to_imgmsg(obstacle_img, 'bgr8')
            # obstacle_img_msg.header = header
            # self.obstacle_image_pub.publish(obstacle_img_msg)
            
            self._front_camera_detections = obstacle_array
            
        except Exception as e:
            self.get_logger().error(f"Front obstacle detection error: {str(e)}")
    
    def _attempt_front_fusion(self) -> None:
        """Fixed front sensor fusion with proper error handling"""
        try:
            current_time = time.time()
            
            # Check if we have both camera and LiDAR data within timeout
            camera_fresh = (current_time - self._cache_timestamps['front_camera']) < self.sensor_timeout
            lidar_fresh = (current_time - self._cache_timestamps['front_lidar']) < self.sensor_timeout
            
            if not (camera_fresh and lidar_fresh):
                return
            
            camera_msg = self._sensor_cache['front_camera']
            lidar_msg = self._sensor_cache['front_lidar']
            
            if camera_msg is None or lidar_msg is None:
                return
            
            # Process camera to obstacles
            camera_obstacles = self.front_camera_processor.process(camera_msg)
            
            # Process LiDAR to obstacles
            lidar_obstacles = self.front_lidar_processor.process(lidar_msg)
            
            # Perform fusion with proper error handling
            try:
                fused_obstacles = self.front_fusion.process(camera_obstacles, lidar_obstacles)
                
                # Update relative speeds
                self._update_relative_speeds(fused_obstacles)
                
                # Publish fused result
                self.front_obstacles_pub.publish(fused_obstacles)
                
            except Exception as fusion_error:
                self.get_logger().warn(f"Front fusion processing error: {str(fusion_error)}")
                # Publish individual sensor data as fallback
                if len(lidar_obstacles.obstacles) > 0:
                    self.front_obstacles_pub.publish(lidar_obstacles)
                elif len(camera_obstacles.obstacles) > 0:
                    self.front_obstacles_pub.publish(camera_obstacles)
            
        except Exception as e:
            self.get_logger().error(f"Front fusion error: {str(e)}")
    
    def _attempt_rear_fusion(self) -> None:
        """Fixed rear sensor fusion with proper error handling"""
        try:
            current_time = time.time()
            
            # Check radar data freshness
            left_fresh = (current_time - self._cache_timestamps['radar_left']) < self.sensor_timeout
            right_fresh = (current_time - self._cache_timestamps['radar_right']) < self.sensor_timeout
            camera_available = self._sensor_cache['rear_camera'] is not None
            
            if not (left_fresh and right_fresh):
                return
            
            # Process radar data
            radar_obstacles = self.rear_radar_processor.process(
                self._sensor_cache['radar_left'],
                self._sensor_cache['radar_right']
            )
            
            # Process camera data if available
            camera_obstacles = ObstacleArray()
            if camera_available:
                try:
                    camera_obstacles = self.rear_camera_processor.process(
                        self._sensor_cache['rear_camera']
                    )
                except Exception as cam_error:
                    self.get_logger().warn(f"Rear camera processing error: {str(cam_error)}")
                    camera_obstacles = ObstacleArray()
            
            # Perform fusion with proper error handling
            try:
                fused_obstacles = self.rear_fusion.process(radar_obstacles, camera_obstacles)
                
                # Update relative speeds
                self._update_relative_speeds(fused_obstacles)
                
                # Publish fused result
                self.rear_obstacles_pub.publish(fused_obstacles)
                
            except Exception as fusion_error:
                self.get_logger().warn(f"Rear fusion processing error: {str(fusion_error)}")
                # Publish radar data as fallback
                if len(radar_obstacles.obstacles) > 0:
                    self.rear_obstacles_pub.publish(radar_obstacles)
            
        except Exception as e:
            self.get_logger().error(f"Rear fusion error: {str(e)}")
    
    # ========================================================================
    # UTILITY FUNCTIONS
    # ========================================================================
    
    def _create_lane_info_message(self, header, lane_center, left_curv, right_curv,
                                lane_width, confidence, distance, image_shape) -> LaneInfo:
        """Create LaneInfo message from detection results"""
        lane_info = LaneInfo()
        lane_info.header = header
        lane_info.header.frame_id = "base_link"
        
        # Lane center
        if lane_center is not None:
            lane_info.center_x = validate_numeric(lane_center[0], 0.0)
            lane_info.center_y = validate_numeric(lane_center[1], 0.0)
            lane_info.detected = True
        else:
            lane_info.detected = False
            lane_info.center_x = 0.0
            lane_info.center_y = 0.0
        
        # Curvature
        lane_info.curvature_left = validate_numeric(left_curv, 0.0)
        lane_info.curvature_right = validate_numeric(right_curv, 0.0)
        
        # Lane properties
        lane_info.lane_width = validate_numeric(lane_width, 0.0)
        lane_info.confidence = validate_numeric(confidence, 0.0, 0.0, 1.0)
        lane_info.distance = validate_numeric(distance, 0.0)
        lane_info.vehicle_speed = self.current_vehicle_speed
        
        # Lateral offset estimation
        if lane_info.detected and lane_info.lane_width > 0 and image_shape:
            image_center = image_shape[1] / 2.0
            lateral_px = abs(lane_info.center_x - image_center)
            lane_info.lateral_offset = (lateral_px / image_shape[1]) * lane_info.lane_width
        else:
            lane_info.lateral_offset = 0.0
        
        return lane_info
    
    def _convert_detections_to_obstacles(self, detections, header, sensor_type: str) -> ObstacleArray:
        """Convert detection results to ObstacleArray format"""
        obstacle_array = ObstacleArray()
        obstacle_array.header = header
        obstacle_array.header.frame_id = "base_link"
        
        if not detections:
            return obstacle_array
        
        for detection in detections:
            # Expected format: (x1, y1, x2, y2, distance, speed, class_id, track_id)
            if len(detection) < 4:
                continue
            
            obstacle = Obstacle()
            
            # Bounding box
            obstacle.bbox = [int(x) for x in detection[:4]]
            
            # Distance and speed
            if len(detection) > 4:
                obstacle.distance = validate_numeric(detection[4], -1.0, min_val=0.0)
            if len(detection) > 5:
                obstacle.speed = validate_numeric(detection[5], 0.0)
            if len(detection) > 6:
                obstacle.class_id = int(detection[6])
            if len(detection) > 7:
                obstacle.track_id = int(detection[7])
            
            # Sensor information
            obstacle.sensor_type = sensor_type
            obstacle.confidence = 0.7  # Default camera confidence
            obstacle.sensor_sources = [sensor_type]
            obstacle.fusion_distance = obstacle.distance
            obstacle.relative_speed = 0.0  # Will be updated later
            obstacle.position_3d = [0.0, 0.0, 0.0]  # Will be computed by fusion
            obstacle.size_3d = [0.0, 0.0, 0.0]
            obstacle.track_age = 0
            obstacle.point_count = 0
            
            obstacle_array.obstacles.append(obstacle)
        
        return obstacle_array
    
    def _update_relative_speeds(self, obstacle_array: ObstacleArray) -> None:
        """Update relative speeds based on current vehicle speed"""
        for obstacle in obstacle_array.obstacles:
            if obstacle.speed != 0.0:
                # Convert absolute speed to relative speed
                obstacle.relative_speed = obstacle.speed - self.current_vehicle_speed
            # If obstacle.relative_speed is already set (from radar), keep it
    
    def _check_processing_performance(self) -> None:
        """Monitor processing performance and log warnings"""
        current_time = time.time()
        dt = current_time - self._last_update_time
        
        # Check for slow processing
        mpt_cfg = self.perception_params.get('performance', {}).get('max_processing_time', 0.1)
        # Accept both a scalar or a per-module dict. Prefer 'global' if provided.
        if isinstance(mpt_cfg, dict):
            if 'global' in mpt_cfg:
                max_processing_time = float(mpt_cfg.get('global', 0.1) or 0.1)
            else:
                try:
                    max_processing_time = float(max(mpt_cfg.values()))
                except Exception:
                    max_processing_time = 0.1
        else:
            try:
                max_processing_time = float(mpt_cfg)
            except Exception:
                max_processing_time = 0.1
        if dt > max_processing_time:
            self.get_logger().warn(f"Slow perception processing detected: {dt:.3f}s")
        
        self._last_update_time = current_time
        
        # Publish performance statistics periodically
        if hasattr(self, '_last_stats_time'):
            stats_interval = 5.0  # Publish stats every 5 seconds
            if current_time - self._last_stats_time > stats_interval:
                self._publish_performance_stats()
                self._last_stats_time = current_time
        else:
            self._last_stats_time = current_time
    
    def _publish_performance_stats(self) -> None:
        """Publish performance statistics"""
        try:
            # Collect stats from all processors
            stats_msg = ObstacleArray()  # Reuse for stats
            stats_msg.header.stamp = self.get_clock().now().to_msg()
            stats_msg.header.frame_id = "performance_stats"
            
            # Create pseudo-obstacles to carry performance data
            modules = [
                ('front_camera', getattr(self, 'obstacle_detector', None)),
                ('front_lidar', getattr(self, 'front_lidar_processor', None)),
                ('front_fusion', getattr(self, 'front_fusion', None)),
                ('rear_radar', getattr(self, 'rear_radar_processor', None)),
                ('rear_camera', getattr(self, 'rear_camera_processor', None)),
                ('rear_fusion', getattr(self, 'rear_fusion', None))
            ]
            
            for name, module in modules:
                if module and hasattr(module, 'get_stats'):
                    stats = module.get_stats()
                    
                    # Create pseudo-obstacle with stats
                    stat_obs = Obstacle()
                    stat_obs.sensor_type = name
                    stat_obs.confidence = float(stats.get('processing_time', 0.0))
                    stat_obs.distance = float(stats.get('input_count', 0))
                    stat_obs.speed = float(stats.get('output_count', 0))
                    stat_obs.track_id = int(stats.get('errors', 0))
                    
                    stats_msg.obstacles.append(stat_obs)
            
            self.stats_pub.publish(stats_msg)
            
        except Exception as e:
            self.get_logger().error(f"Failed to publish performance stats: {str(e)}")
    
    def _cleanup_sensor_cache(self) -> None:
        """Clean up stale sensor data from cache"""
        current_time = time.time()
        cleanup_threshold = self.sensor_timeout * 3  # Keep data for 3x timeout
        
        for sensor_name, timestamp in self._cache_timestamps.items():
            if current_time - timestamp > cleanup_threshold:
                self._sensor_cache[sensor_name] = None
                self._cache_timestamps[sensor_name] = 0.0


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main(args=None):
    """Main entry point for the perception node"""
    rclpy.init(args=args)
    
    try:
        # Create and run the perception node
        node = NewPerceptionNode()

        # Run the node
        rclpy.spin(node)

    finally:
        # Cleanup
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()