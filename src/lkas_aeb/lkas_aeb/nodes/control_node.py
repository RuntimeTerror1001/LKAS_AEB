#!/usr/bin/env python3
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node
import carla
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from carla_msgs.msg import CarlaEgoVehicleControl, CarlaEgoVehicleStatus
from carla_common.transforms import ros_pose_to_carla_transform

from lkas_aeb_msgs.msg import LaneInfo, ObstacleArray
from lkas_aeb.util.helpers import load_params
from lkas_aeb.modules.control.pure_pursuit import PurePursuit
from lkas_aeb.modules.control.speed_pid import SpeedPID
from lkas_aeb.modules.control.aeb_controller import AEBController

"""
CONTROL NODE
"""

class ControlNode(Node):
    """
    ROS2 Node for vehicle control that integrates LKAS (Lane Keeping Assist System) 
    and AEB (Automatic Emergency Braking) functionality.
    
    Subscribes to:
        - /carla/hero/odometry: Vehicle pose and position
        - /perception/lane_info: Lane detection results
        - /perception/obstacles_info: Obstacle detection results
        - /carla/hero/vehicle_status: Vehicle speed and status
        - /carla/hero/waypoints: Path to follow
        - /goal_pose: Target destination
    
    Publishes:
        - /carla/hero/vehicle_control_cmd: Vehicle control commands
        - /carla/hero/goal_pose: Goal pose for path planning
    """

    def __init__(self):
        super().__init__('control_node')

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

        # Extract and combine configuration sections
        lkas_control_params = lkas_params['control']
        aeb_control_params = aeb_params['control']

        control_params = {
            'control': {
                **lkas_control_params,
                **aeb_control_params
            }
        }

        # ========================
        # CONTROLLER INITIALIZATION
        # ========================
        self.pure_pursuit = PurePursuit(control_params)
        self.speed_pid = SpeedPID(control_params)
        self.aeb_controller = AEBController(control_params, self.get_logger())

        # ========================
        # STATE VARIABLES
        # ========================
        # Vehicle State
        self.curr_speed = 0.0
        self.curr_pose = None
        self.current_time = None

        # Perception Data
        self.lane_info = None
        self.obstacles = []
        self.obstacles_stamp = None

        # Path Planning
        self.waypoints = []
        self.target_speed = 15.0
        self.has_active_goal = False
        self.curr_goal = None

        # Control History
        self.last_control_time = None

        # ========================
        # ROS COMMUNICATION SETUP
        # ========================
        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/carla/hero/odometry', self.odom_cb, 10
        )
        self.lane_info_sub = self.create_subscription(
            LaneInfo, '/perception/lane_info', self.lane_info_cb, 10
        )
        self.obstacles_sub = self.create_subscription(
            ObstacleArray, '/perception/fused_obstacles_front', self.obstacles_cb, 10
        )
        self.speed_sub = self.create_subscription(
            CarlaEgoVehicleStatus, '/carla/hero/vehicle_status', self.vehicle_status_cb, 10
        )
        self.path_sub = self.create_subscription(
            Path, 'carla/hero/waypoints', self.path_cb, 10
        )
        self.goal_sub = self.create_subscription(
            PoseStamped, '/goal_pose', self.goal_cb, 10
        )

        # Publisher
        self.control_pub = self.create_publisher(
            CarlaEgoVehicleControl, '/carla/hero/vehicle_control_cmd', 10
        )
        self.goal_pub = self.create_publisher(
            PoseStamped, '/carla/hero/goal_pose', 10
        )

        # Control Timer (20Hz)
        self.control_timer = self.create_timer(0.05, self.control_loop)

    def odom_cb(self, msg):
        """
        Callback for vehicle odometry updates.
        
        Args:
            msg (Odometry): Vehicle pose and position data
        """
        transform = ros_pose_to_carla_transform(msg.pose.pose)
        stamp = msg.header.stamp
        self.current_time = stamp.sec + stamp.nanosec/1e9
        self.curr_pose = transform
    
    def lane_info_cb(self, msg):
        """
        Callback for lane detection updates.
        
        Args:
            msg (LaneInfo): Lane detection results
        """
        self.lane_info = msg

    def obstacles_cb(self, msg):
        """
        Callback for obstacle detection updates.
        
        Args:
            msg (ObstacleArray): Array of detected obstacles
        """
        self.obstacles = msg.obstacles
        self.obstacles_stamp = msg.header.stamp

    def vehicle_status_cb(self, msg):
        """
        Callback for vehicle status updates.
        
        Args:
            msg (CarlaEgoVehicleStatus): Vehicle status including speed
        """
        self.curr_speed = msg.velocity

    def path_cb(self, msg):
        """
        Enhanced path processing with validation and smoothing.
        
        Args:
            msg (Path): Path containing waypoints to follow
        """
        if not msg.poses:
            return
            
        self.waypoints = []

        # Convert ROS poses to CARLA transforms
        raw_waypoints = []
        for pose in msg.poses:
            transform = ros_pose_to_carla_transform(pose.pose)
            raw_waypoints.append(transform)
        
        # Apply smoothing for paths with insufficient points
        if len(raw_waypoints) < 3:
            self.waypoints = raw_waypoints
        else:
            smoothed_waypoints = [raw_waypoints[0]]  # Keep first waypoint
            
            # Smooth intermediate waypoints using weighted average
            for i in range(1, len(raw_waypoints) - 1):
                prev = raw_waypoints[i-1]
                curr = raw_waypoints[i]
                next_wp = raw_waypoints[i+1]

                # Calculate smoothed position with weighted average
                # Give more weight to current point to preserve path shape
                x = (0.2 * prev.location.x + 0.6 * curr.location.x + 0.2 * next_wp.location.x)
                y = (0.2 * prev.location.y + 0.6 * curr.location.y + 0.2 * next_wp.location.y)
                z = (0.2 * prev.location.z + 0.6 * curr.location.z + 0.2 * next_wp.location.z)

                # Validate smoothed point isn't too far from original
                dist_from_original = np.hypot(x - curr.location.x, y - curr.location.y)
                if dist_from_original > 2.0:  # Max 2m deviation
                    smoothed_waypoints.append(curr)  # Use original if too much deviation
                else:
                    smoothed_waypoints.append(carla.Transform(carla.Location(x, y, z), curr.rotation))
            
            smoothed_waypoints.append(raw_waypoints[-1])  # Keep last waypoint
            self.waypoints = smoothed_waypoints
        
        self.pure_pursuit.update_path(self.waypoints)

    def goal_cb(self, msg):
        """
        Callback for new goal pose.
        
        Args:
            msg (PoseStamped): Target destination pose
        """
        self.has_active_goal = True
        self.curr_goal = msg

        self.goal_pub.publish(msg)
        self.get_logger().info('New Goal Received')

    def calculate_curve_speed_factor(self):
        """
        Calculate speed reduction factor for curves using lane curvature and path geometry.
        
        Args:
            None (uses internal lane_info and pure_pursuit data)
        
        Returns:
            float: Speed factor between 0.5 and 1.0 (1.0 = no reduction)
        """
        curve_factor = 1.0
        
        # Use lane curvature if available
        if self.lane_info and self.lane_info.detected:
            left_curv = self.lane_info.curvature_left
            right_curv = self.lane_info.curvature_right
            
            # Use the tighter curve for speed calculation
            min_curvature = min(left_curv, right_curv)
            if min_curvature < 500:  # Sharp curve
                curve_factor = max(0.5, min(1.0, min_curvature / 500))
        
        # Consider path curvature from pure pursuit
        pp_steering = self.pure_pursuit.calculate_steering(self.curr_speed)
        if abs(pp_steering) > 0.2:  # Significant steering required
            path_curve_factor = max(0.6, 1.0 - abs(pp_steering) / self.pure_pursuit.max_steer)
            curve_factor = min(curve_factor, path_curve_factor)
        
        return curve_factor
    
    def is_intersection(self):
        """
        Detect intersections using waypoint density analysis.
        
        Args:
            None (uses internal waypoints)
        
        Returns:
            bool: True if intersection detected, False otherwise
        """
        if len(self.waypoints) < 10:
            return False
        
        # Calculate distances between consecutive waypoints
        distances = []
        for i in range(1, min(10, len(self.waypoints))):
            wp1 = self.waypoints[i-1]
            wp2 = self.waypoints[i]
            dx = wp2.location.x - wp1.location.x
            dy = wp2.location.y - wp1.location.y
            distances.append(np.hypot(dx, dy))
        
        return np.std(distances) > 2.0 # High variance = intersection
    
    def calculate_obstacle_speed_factor(self, obstacles, curr_speed):
        """
        Calculate speed reduction factor based on obstacles with progressive reduction.
        
        Args:
            obstacles: List of detected obstacles
            curr_speed: Current ego vehicle speed
            
        Returns:
            tuple: (speed_factor, closest_distance, emergency_stop_required)
        """
        if not obstacles:
            return 1.0, float('inf'), False
        
        # Find closest obstacle with class-specific considerations
        closest_distance = float('inf')
        closest_class = None
        
        for obstacle in obstacles:
            if hasattr(obstacle, 'distance'):
                if obstacle.distance < closest_distance:
                    closest_distance = obstacle.distance
                    closest_class = getattr(obstacle, 'class_id', 0)
        
        if closest_distance == float('inf'):
            return 1.0, float('inf'), False
        
        # ========================
        # CLASS-SPECIFIC DISTANCE THRESHOLDS
        # ========================
        if closest_class in [2, 5, 7]:  # Cars, buses, trucks
            critical_distance = 5.0    # Emergency stop
            brake_distance = 12.0      # Start heavy braking
            slow_distance = 25.0       # Start speed reduction
            comfort_distance = 40.0    # Comfortable following
        elif closest_class in [0]:  # Pedestrians
            critical_distance = 3.0
            brake_distance = 8.0
            slow_distance = 15.0
            comfort_distance = 25.0
        else:  # Motorcycles, bikes, unknown
            critical_distance = 4.0
            brake_distance = 10.0
            slow_distance = 20.0
            comfort_distance = 30.0
        
        # ========================
        # PROGRESSIVE SPEED REDUCTION
        # ========================
        emergency_stop = False
        
        if closest_distance < critical_distance:
            # CRITICAL: Emergency stop required
            speed_factor = 0.0
            emergency_stop = True
            self.get_logger().error(f"CRITICAL DISTANCE: {closest_distance:.1f}m - EMERGENCY STOP!")
            
        elif closest_distance < brake_distance:
            # BRAKING ZONE: Rapid speed reduction
            # Linear reduction from 0.3 to 0.0 as we get closer
            normalized_distance = (closest_distance - critical_distance) / (brake_distance - critical_distance)
            speed_factor = 0.3 * normalized_distance  # Max 30% speed in braking zone
            self.get_logger().warning(f"BRAKING ZONE: {closest_distance:.1f}m - Speed factor: {speed_factor:.2f}")
            
        elif closest_distance < slow_distance:
            # SLOW ZONE: Moderate speed reduction  
            # Linear reduction from 0.7 to 0.3 as we get closer
            normalized_distance = (closest_distance - brake_distance) / (slow_distance - brake_distance)
            speed_factor = 0.3 + 0.4 * normalized_distance  # 30% to 70% speed
            self.get_logger().info(f"SLOW ZONE: {closest_distance:.1f}m - Speed factor: {speed_factor:.2f}")
            
        elif closest_distance < comfort_distance:
            # CAUTION ZONE: Light speed reduction
            # Linear reduction from 1.0 to 0.7 as we get closer  
            normalized_distance = (closest_distance - slow_distance) / (comfort_distance - slow_distance)
            speed_factor = 0.7 + 0.3 * normalized_distance  # 70% to 100% speed
            self.get_logger().debug(f"CAUTION ZONE: {closest_distance:.1f}m - Speed factor: {speed_factor:.2f}")
            
        else:
            # SAFE ZONE: Full speed allowed
            speed_factor = 1.0
        
        # ========================
        # SPEED-DEPENDENT ADJUSTMENTS
        # ========================
        # At higher speeds, be more conservative
        if curr_speed > 10.0:  # Above 36 km/h
            speed_factor *= 0.9  # 10% more conservative
        if curr_speed > 15.0:  # Above 54 km/h  
            speed_factor *= 0.8  # Additional 20% reduction
        
        return speed_factor, closest_distance, emergency_stop

    def control_loop(self):
        """
        Main control loop executed at 20Hz. Integrates path following, speed control, and AEB.
        
        Args:
            None (uses internal state variables)
        
        Returns:
            None (publishes vehicle control commands)
        """

        # Ensure valid time information
        if not hasattr(self, 'current_time') or self.current_time is None:
            return
            
        # Calculate time-delta for contorl updates
        curr_time = self.current_time
        dt = 0.05
        if self.last_control_time is not None:
            dt = max(0.001, min(0.2, curr_time - self.last_control_time))
        self.last_control_time = curr_time

        # ========================
        # GOAL COMPLETION CHECK
        # ========================
        # Check if the ego vehicle is at the destination
        if self.has_active_goal and self.waypoints:
            last_wp = self.waypoints[-1].location
            curr_loc = self.curr_pose.location
            distance_to_goal = np.hypot(
                last_wp.x - curr_loc.x,
                last_wp.y - curr_loc.y
            )
            
            if distance_to_goal < 2.0:  # Stopping threshold (2 meters)
                # Goal reached - stop and reset
                control_msg = CarlaEgoVehicleControl()
                control_msg.brake = 1.0
                control_msg.steer = 0.0
                self.control_pub.publish(control_msg)
                self.get_logger().info("Goal reached! Stopping.")
                self.has_active_goal = False
                self.waypoints = []  # Clear path
                return
        
        # Stop if no active goal
        if not self.has_active_goal:
            control_msg = CarlaEgoVehicleControl()
            control_msg.brake = 1.0
            control_msg.steer = 0.0
            self.control_pub.publish(control_msg)
            return

        # ========================
        # STEERING CALCULATION
        # ========================
        # Update controllers with current state
        self.pure_pursuit.update_pose(self.curr_pose)

        # Calculate steering from both controllers
        pp_steering = self.pure_pursuit.calculate_steering(self.curr_speed)
        
        # ========================
        # SPEED ADAPTATION
        # ========================
        in_intersection = self.is_intersection()

        if in_intersection:
            # More aggressive speed reduction in intersections
            curve_factor = min(0.7, self.calculate_curve_speed_factor())
        else:
            # Apply rate limiting & smoothing
            curve_factor = self.calculate_curve_speed_factor()

        # ========================
        # OBSTACLE PROCESSING & AEB
        # ========================
        # Handle obstacle data freshness
        if self.obstacles_stamp:
            obs_time = self.obstacles_stamp.sec + self.obstacles_stamp.nanosec / 1e9
        else:
            obs_time = curr_time

        # Filter stale obstacle detections
        if curr_time - obs_time > 0.5:  # 500ms staleness threshold
            obstacles = []
        else:
            obstacles = self.obstacles  

        # Calculate progressive speed reduction based on obstacles
        obstacle_speed_factor, closest_distance, emergency_stop = self.calculate_obstacle_speed_factor(obstacles, self.curr_speed)

        # Log obstacle information for debugging
        # if obstacles:
        #     self.get_logger().info(f"Processing {len(obstacles)} obstacles")
        #     for i, obs in enumerate(obstacles[:3]):  # Log first 3 obstacles
        #         self.get_logger().info(
        #             f"Obstacle {i}: Dist={obs.distance:.1f}m, "
        #             f"Class={obs.class_id}, TrackID={getattr(obs, 'track_id', 'None')}"
        #         )

        # ========================
        # COMPREHENSIVE SPEED ADAPTATION
        # ========================
        # Start with base target speed
        adapted_target_speed = self.target_speed
        
        # Apply curve speed factor
        curve_factor = self.calculate_curve_speed_factor()
        adapted_target_speed *= curve_factor
        
        # Apply intersection speed factor
        if self.is_intersection():
            adapted_target_speed *= 0.7  # 30% reduction in intersections
        
        # Apply obstacle-based speed reduction (most important)
        adapted_target_speed *= obstacle_speed_factor
        
        # ========================
        # EMERGENCY OVERRIDE OR NORMAL CONTROL
        # ========================
        if emergency_stop:
            # EMERGENCY: Complete stop required
            throttle = 0.0
            brake = 1.0  # Maximum brake
            
            # Engage handbrake if still moving
            if self.curr_speed > 0.5:
                control_msg = CarlaEgoVehicleControl()
                control_msg.throttle = 0.0
                control_msg.brake = 1.0
                control_msg.steer = float(np.nan_to_num(pp_steering, nan=0.0))
                control_msg.reverse = False
                control_msg.hand_brake = True
                self.control_pub.publish(control_msg)
                
                self.get_logger().error(f"EMERGENCY HANDBRAKE: Distance={closest_distance:.1f}m, Speed={self.curr_speed:.1f}m/s")
                return
                
        else:
            # NORMAL OPERATION: Use adapted target speed
            throttle, brake = self.speed_pid.update(adapted_target_speed, self.curr_speed, curr_time)
            
            # Additional AEB check for extra safety
            ttc_msg = self.aeb_controller.calculate_ttc(obstacles, self.curr_speed, obs_time)
            aeb_brake = self.aeb_controller.decide_braking(ttc_msg, self.curr_speed)
            
            if aeb_brake > brake:
                brake = aeb_brake
                throttle = 0.0  # Cut throttle when AEB is active
        
        # ========================
        # STOPPED VEHICLE HOLD
        # ========================
        # Prevent creeping when stopped near obstacles
        if self.curr_speed < 0.8 and closest_distance < 15.0:
            throttle = 0.0
            brake = max(brake, 0.4)  # Maintain brake pressure
            self.get_logger().debug(f"HOLDING BRAKE: Speed={self.curr_speed:.1f}, Distance={closest_distance:.1f}")
        
        # ========================
        # SAFETY BOUNDS AND COMMAND PUBLICATION
        # ========================
        steer = np.nan_to_num(pp_steering, nan=0.0)
        throttle = np.nan_to_num(throttle, nan=0.0)
        brake = np.nan_to_num(brake, nan=0.0)
        
        # Clip values to valid ranges
        steer = np.clip(steer, -1.0, 1.0)
        throttle = np.clip(throttle, 0.0, 1.0)
        brake = np.clip(brake, 0.0, 1.0)
        
        # Create and publish control message
        control_msg = CarlaEgoVehicleControl()
        control_msg.steer = float(steer)
        control_msg.throttle = float(throttle)
        control_msg.brake = float(brake)
        control_msg.reverse = False
        control_msg.hand_brake = False
        
        self.control_pub.publish(control_msg)

        # ========================
        # ENHANCED LOGGING
        # ========================
        if obstacle_speed_factor < 0.9:  # Log when speed is being reduced
            self.get_logger().info(
                f'SPEED CONTROL: Target={self.target_speed:.1f} -> Adapted={adapted_target_speed:.1f}m/s, '
                f'ObsFactor={obstacle_speed_factor:.2f}, Distance={closest_distance:.1f}m, '
                f'Throttle={throttle:.2f}, Brake={brake:.2f}'
            )

# ========================
# MAIN FUNCTION
# ========================

def main(args=None):
    """
    Main entry point for the vehicle control node.
    
    Args:
        args: Command line arguments (optional)
    """
    rclpy.init(args=args)
    node = ControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()