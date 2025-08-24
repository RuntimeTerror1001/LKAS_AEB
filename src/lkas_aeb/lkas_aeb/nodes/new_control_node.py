#!/usr/bin/env python3
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node
import carla
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from carla_msgs.msg import CarlaEgoVehicleControl, CarlaEgoVehicleStatus #type:ignore
from carla_common.transforms import ros_pose_to_carla_transform #type:ignore

from lkas_aeb_msgs.msg import LaneInfo, ObstacleArray
from lkas_aeb.util.helpers import load_params
from lkas_aeb.modules.control.pure_pursuit import PurePursuit
from lkas_aeb.modules.control.speed_pid import SpeedPID
from lkas_aeb.modules.control.aeb_controller import AEBController
from lkas_aeb.modules.control.lc_hold import should_hold_for_lane_change, Side

"""
ENHANCED CONTROL NODE WITH REAR SENSOR FUSION
"""

class NewControlNode(Node):
    """
    Enhanced ROS2 Node for vehicle control that integrates:
    - LKAS (Lane Keeping Assist System) with rear sensor awareness
    - AEB (Automatic Emergency Braking) with front obstacle detection  
    - Lane Change Hold functionality using rear radar/camera fusion
    - Adaptive speed control based on multi-sensor obstacle detection
    
    Subscribes to:
        - /carla/hero/odometry: Vehicle pose and position
        - /perception/lane_info: Lane detection results
        - /perception/obstacles_info: Front obstacle detection results
        - /perception/obstacles_rear_fused: Rear sensor fusion results
        - /carla/hero/vehicle_status: Vehicle speed and status
        - /carla/hero/waypoints: Path to follow
        - /goal_pose: Target destination
    
    Publishes:
        - /carla/hero/vehicle_control_cmd: Vehicle control commands
        - /carla/hero/goal_pose: Goal pose for path planning
    """

    def __init__(self):
        super().__init__('new_control_node')

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

        # Combine all parameters for lane change hold
        self.control_params = {
            'control': {
                **lkas_control_params,
                **aeb_control_params
            },
            'lane_change_hold': {
                # Lane change hold specific parameters
                'D_front_min': 30.0,           # Minimum front gap (meters)
                'D_rear_min': 25.0,            # Minimum rear gap (meters)  
                'TTC_min': 5.0,                # Minimum time-to-collision (seconds)
                'near_side_window': 8.0,       # Distance for "alongside" detection
                'lc_max_distance': 80.0,       # Maximum relevant obstacle distance
                'lc_min_confidence': 0.3,      # Minimum obstacle confidence
                'lc_relevant_classes': [1, 2, 3],  # Vehicle class IDs
                'min_speed_for_lc_hold': 2.0,  # Minimum speed for lane change gating
                'critical_distance': 5.0,      # Critical threat distance
                'fusion_max_position_distance': 3.0,
                'fusion_max_bearing_diff': 0.3,
                'fusion_min_bbox_overlap': 0.1,
                'fusion_min_confidence': 0.2,
                'fusion_max_distance': 100.0,
                'fusion_max_distance_error': 0.5,
                # Lane polygons for left/right lane change detection
                'left_poly': {
                    'x_min': -12.0, 'x_max': 2.0,
                    'y_min': -3.8, 'y_max': -2.2
                },
                'right_poly': {
                    'x_min': -12.0, 'x_max': 2.0,
                    'y_min': 2.2, 'y_max': 3.8
                }
            }
        }

        # ========================
        # CONTROLLER INITIALIZATION
        # ========================
        self.pure_pursuit = PurePursuit({'control': self.control_params['control']})
        self.speed_pid = SpeedPID({'control': self.control_params['control']})
        self.aeb_controller = AEBController({'control': self.control_params['control']}, self.get_logger())

        # ========================
        # STATE VARIABLES
        # ========================
        # Vehicle State
        self.curr_speed = 0.0
        self.curr_pose = None
        self.current_time = None

        # Perception Data
        self.lane_info = None
        self.front_obstacles = []
        self.front_obstacles_stamp = None
        self.rear_obstacles = []
        self.rear_obstacles_stamp = None

        # Path Planning & Lane Change State
        self.waypoints = []
        self.target_speed = 15.0
        self.has_active_goal = False
        self.curr_goal = None
        
        # Lane change detection state
        self.current_lane_change_side = "none"  # "left", "right", or "none"
        self.lane_change_start_time = None
        self.lane_change_duration_threshold = 2.0  # seconds
        self.last_lateral_offset = 0.0
        self.lateral_offset_history = []
        self.lateral_offset_window = 20  # frames for trend analysis

        # Control History
        self.last_control_time = None
        self.throttle_hold_active = False
        self.hold_reason = ""

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
        self.front_obstacles_sub = self.create_subscription(
            ObstacleArray, '/perception/obstacles_front_fused', self.front_obstacles_cb, 10
        )
        # NEW: Rear sensor fusion subscriber
        self.rear_obstacles_sub = self.create_subscription(
            ObstacleArray, '/perception/obstacles_rear_fused', self.rear_obstacles_cb, 10
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

        # Publishers
        self.control_pub = self.create_publisher(
            CarlaEgoVehicleControl, '/carla/hero/vehicle_control_cmd', 10
        )
        self.goal_pub = self.create_publisher(
            PoseStamped, '/carla/hero/goal_pose', 10
        )

        # Control Timer (20Hz)
        self.control_timer = self.create_timer(0.05, self.control_loop)

        self.get_logger().info("Enhanced Control Node with Rear Sensor Fusion initialized")

    def odom_cb(self, msg):
        """Callback for vehicle odometry updates."""
        transform = ros_pose_to_carla_transform(msg.pose.pose)
        stamp = msg.header.stamp
        self.current_time = stamp.sec + stamp.nanosec/1e9
        self.curr_pose = transform
    
    def lane_info_cb(self, msg):
        """Callback for lane detection updates."""
        self.lane_info = msg
        
        # Track lateral offset for lane change detection
        if msg.detected and msg.lateral_offset != 0.0:
            self.lateral_offset_history.append(msg.lateral_offset)
            if len(self.lateral_offset_history) > self.lateral_offset_window:
                self.lateral_offset_history.pop(0)
            self.last_lateral_offset = msg.lateral_offset

    def front_obstacles_cb(self, msg):
        """Callback for front obstacle detection updates."""
        self.front_obstacles = msg.obstacles
        self.front_obstacles_stamp = msg.header.stamp

    def rear_obstacles_cb(self, msg):
        """NEW: Callback for rear sensor fusion obstacle updates."""
        self.rear_obstacles = msg.obstacles
        self.rear_obstacles_stamp = msg.header.stamp
        
        # Log rear obstacle count for debugging
        if self.rear_obstacles:
            sensor_counts = {}
            for obs in self.rear_obstacles:
                sensor_type = obs.sensor_type
                sensor_counts[sensor_type] = sensor_counts.get(sensor_type, 0) + 1
            
            self.get_logger().debug(
                f"Rear obstacles: {len(self.rear_obstacles)} total - {dict(sensor_counts)}"
            )

    def vehicle_status_cb(self, msg):
        """Callback for vehicle status updates."""
        self.curr_speed = msg.velocity

    def path_cb(self, msg):
        """Enhanced path processing with validation and smoothing."""
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
        """Callback for new goal pose."""
        self.has_active_goal = True
        self.curr_goal = msg
        self.goal_pub.publish(msg)
        self.get_logger().info('New Goal Received')

    def detect_lane_change_intent(self):
        """
        Detect if vehicle is performing a lane change based on lateral movement.
        
        Returns:
            str: "left", "right", or "none"
        """
        if not self.lateral_offset_history or len(self.lateral_offset_history) < 10:
            return "none"
        
        # Calculate trend in lateral offset
        recent_offsets = self.lateral_offset_history[-10:]
        offset_trend = np.polyfit(range(len(recent_offsets)), recent_offsets, 1)[0]
        
        # Thresholds for lane change detection
        trend_threshold = 0.05  # m/frame
        offset_magnitude_threshold = 0.5  # meters
        
        current_offset = self.last_lateral_offset
        
        # Determine lane change direction
        if abs(offset_trend) > trend_threshold or abs(current_offset) > offset_magnitude_threshold:
            if current_offset > 0 or offset_trend > 0:
                return "left"  # Moving left
            else:
                return "right"  # Moving right
        
        return "none"

    def update_lane_change_state(self):
        """Update the current lane change state and duration tracking."""
        detected_side = self.detect_lane_change_intent()
        
        if detected_side != "none":
            if self.current_lane_change_side != detected_side:
                # Lane change direction changed or started
                self.current_lane_change_side = detected_side
                self.lane_change_start_time = self.current_time
                self.get_logger().info(f"Lane change detected: {detected_side}")
            # Continue existing lane change
        else:
            # Check if we should end the lane change
            if self.current_lane_change_side != "none":
                if (self.current_time - self.lane_change_start_time) > self.lane_change_duration_threshold:
                    self.get_logger().info(f"Lane change completed: {self.current_lane_change_side}")
                    self.current_lane_change_side = "none"
                    self.lane_change_start_time = None

    def calculate_curve_speed_factor(self):
        """Calculate speed reduction factor for curves."""
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
        """Detect intersections using waypoint density analysis."""
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
            normalized_distance = (closest_distance - critical_distance) / (brake_distance - critical_distance)
            speed_factor = 0.3 * normalized_distance  # Max 30% speed in braking zone
            self.get_logger().warning(f"BRAKING ZONE: {closest_distance:.1f}m - Speed factor: {speed_factor:.2f}")
            
        elif closest_distance < slow_distance:
            # SLOW ZONE: Moderate speed reduction  
            normalized_distance = (closest_distance - brake_distance) / (slow_distance - brake_distance)
            speed_factor = 0.3 + 0.4 * normalized_distance  # 30% to 70% speed
            self.get_logger().info(f"SLOW ZONE: {closest_distance:.1f}m - Speed factor: {speed_factor:.2f}")
            
        elif closest_distance < comfort_distance:
            # CAUTION ZONE: Light speed reduction
            normalized_distance = (closest_distance - slow_distance) / (comfort_distance - slow_distance)
            speed_factor = 0.7 + 0.3 * normalized_distance  # 70% to 100% speed
            self.get_logger().debug(f"CAUTION ZONE: {closest_distance:.1f}m - Speed factor: {speed_factor:.2f}")
            
        else:
            # SAFE ZONE: Full speed allowed
            speed_factor = 1.0
        
        # ========================
        # REMOVE SPEED-DEPENDENT ADJUSTMENTS
        # ========================
        # These were causing excessive braking even at safe distances
        
        return speed_factor, closest_distance, emergency_stop

    def control_loop(self):
        """Main control loop executed at 20Hz with rear sensor fusion integration."""

        # Ensure valid time information
        if not hasattr(self, 'current_time') or self.current_time is None:
            return
            
        # Calculate time-delta for control updates
        curr_time = self.current_time
        dt = 0.05
        if self.last_control_time is not None:
            dt = max(0.001, min(0.2, curr_time - self.last_control_time))
        self.last_control_time = curr_time

        # Update lane change state
        self.update_lane_change_state()

        # ========================
        # GOAL COMPLETION CHECK
        # ========================
        if self.has_active_goal and self.waypoints:
            last_wp = self.waypoints[-1].location
            curr_loc = self.curr_pose.location
            distance_to_goal = np.hypot(
                last_wp.x - curr_loc.x,
                last_wp.y - curr_loc.y
            )
            
            if distance_to_goal < 2.0:  # Stopping threshold
                control_msg = CarlaEgoVehicleControl()
                control_msg.brake = 1.0
                control_msg.steer = 0.0
                self.control_pub.publish(control_msg)
                self.get_logger().info("Goal reached! Stopping.")
                self.has_active_goal = False
                self.waypoints = []
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
        self.pure_pursuit.update_pose(self.curr_pose)
        pp_steering = self.pure_pursuit.calculate_steering(self.curr_speed)
        
        # ========================
        # OBSTACLE DATA PROCESSING
        # ========================
        # Handle front obstacle data freshness
        if self.front_obstacles_stamp:
            front_obs_time = self.front_obstacles_stamp.sec + self.front_obstacles_stamp.nanosec / 1e9
        else:
            front_obs_time = curr_time

        # Handle rear obstacle data freshness
        if self.rear_obstacles_stamp:
            rear_obs_time = self.rear_obstacles_stamp.sec + self.rear_obstacles_stamp.nanosec / 1e9
        else:
            rear_obs_time = curr_time

        # Filter stale obstacle detections
        if curr_time - front_obs_time > 0.5:  # 500ms staleness threshold
            front_obstacles = []
        else:
            front_obstacles = self.front_obstacles

        if curr_time - rear_obs_time > 0.5:  # 500ms staleness threshold
            rear_obstacles = []
        else:
            rear_obstacles = self.rear_obstacles

        # ========================
        # LANE CHANGE HOLD DECISION
        # ========================
        # NEW: Check if we should hold throttle for lane change safety
        lane_width = self.lane_info.lane_width if self.lane_info and self.lane_info.detected else 3.5
        
        # Create ObstacleArray from rear obstacles for lane change hold module
        rear_obstacle_array = ObstacleArray()
        rear_obstacle_array.obstacles = rear_obstacles
        rear_obstacle_array.header.stamp = self.rear_obstacles_stamp or rclpy.time.Time().to_msg()
        
        hold_decision = should_hold_for_lane_change(
            fused_rear=rear_obstacle_array,
            target_side=self.current_lane_change_side,
            ego_speed=self.curr_speed,
            lane_width=lane_width,
            params=self.control_params['lane_change_hold']
        )
        
        # Update hold state
        self.throttle_hold_active = hold_decision['hold']
        self.hold_reason = hold_decision['reason']
        
        if self.throttle_hold_active:
            self.get_logger().warning(f"THROTTLE HOLD ACTIVE: {self.hold_reason}")

        # ========================
        # SPEED ADAPTATION CALCULATIONS
        # ========================
        # Calculate speed factors from front obstacles
        front_obstacle_speed_factor, front_closest_distance, front_emergency_stop = \
            self.calculate_obstacle_speed_factor(front_obstacles, self.curr_speed)

        # Calculate speed factors from rear obstacles (less influence)
        rear_obstacle_speed_factor, rear_closest_distance, rear_emergency_stop = \
            self.calculate_obstacle_speed_factor(rear_obstacles, self.curr_speed)
        
        # Rear obstacles have less influence on speed (mainly for situational awareness)
        rear_influence_weight = 0.3 if self.current_lane_change_side != "none" else 0.1
        combined_rear_factor = 1.0 - (1.0 - rear_obstacle_speed_factor) * rear_influence_weight

        # Calculate curve and intersection factors
        curve_factor = self.calculate_curve_speed_factor()
        intersection_factor = 0.7 if self.is_intersection() else 1.0

        # ========================
        # COMPREHENSIVE SPEED ADAPTATION
        # ========================
        # Start with base target speed
        adapted_target_speed = self.target_speed
        
        # Apply all speed factors
        adapted_target_speed *= curve_factor
        adapted_target_speed *= intersection_factor
        adapted_target_speed *= front_obstacle_speed_factor  # Primary factor
        adapted_target_speed *= combined_rear_factor         # Secondary factor
        
        # ========================
        # EMERGENCY OVERRIDE OR NORMAL CONTROL
        # ========================
        if front_emergency_stop or rear_emergency_stop:
            # EMERGENCY: Complete stop required
            throttle = 0.0
            brake = 1.0
            
            # Engage handbrake if still moving
            if self.curr_speed > 0.5:
                control_msg = CarlaEgoVehicleControl()
                control_msg.throttle = 0.0
                control_msg.brake = 1.0
                control_msg.steer = float(np.nan_to_num(pp_steering, nan=0.0))
                control_msg.reverse = False
                control_msg.hand_brake = True
                self.control_pub.publish(control_msg)
                
                emergency_source = "FRONT" if front_emergency_stop else "REAR"
                emergency_distance = front_closest_distance if front_emergency_stop else rear_closest_distance
                self.get_logger().error(f"EMERGENCY HANDBRAKE ({emergency_source}): Distance={emergency_distance:.1f}m")
                return
                
        else:
            # NORMAL OPERATION: Use adapted target speed
            throttle, brake = self.speed_pid.update(adapted_target_speed, self.curr_speed, curr_time)
            
            # Additional AEB check for extra front safety
            ttc_msg = self.aeb_controller.calculate_ttc(front_obstacles, self.curr_speed, front_obs_time)
            aeb_brake = self.aeb_controller.decide_braking(ttc_msg, self.curr_speed)
            
            if aeb_brake > brake:
                brake = aeb_brake
                throttle = 0.0  # Cut throttle when AEB is active
        
        # ========================
        # LANE CHANGE HOLD ENFORCEMENT
        # ========================
        # NEW: Override throttle if lane change hold is active
        if self.throttle_hold_active:
            throttle = 0.0  # Cut throttle completely
            # Don't override brake - let normal brake logic handle stopping
            self.get_logger().debug(f"Throttle held for lane change safety: {hold_decision['offender_id']}")

        # ========================
        # STOPPED VEHICLE HOLD
        # ========================
        # Prevent creeping when stopped near obstacles
        min_distance = min(front_closest_distance, rear_closest_distance)
        if self.curr_speed < 0.8 and min_distance < 15.0:
            throttle = 0.0
            brake = max(brake, 0.4)  # Maintain brake pressure
        
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
        should_log = (front_obstacle_speed_factor < 0.9 or 
                     self.throttle_hold_active or 
                     len(rear_obstacles) > 0)
        
        if should_log:
            log_parts = [
                f'Target={self.target_speed:.1f} -> Adapted={adapted_target_speed:.1f}m/s',
                f'Front: {len(front_obstacles)}obs, {front_closest_distance:.1f}m, factor={front_obstacle_speed_factor:.2f}',
                f'Rear: {len(rear_obstacles)}obs, {rear_closest_distance:.1f}m, factor={rear_obstacle_speed_factor:.2f}',
                f'LC: {self.current_lane_change_side}, Hold: {self.throttle_hold_active}',
                f'Throttle={throttle:.2f}, Brake={brake:.2f}'
            ]
            
            if self.throttle_hold_active:
                log_parts.append(f'Hold reason: {self.hold_reason}')
            
            self.get_logger().info(' | '.join(log_parts))

# ========================
# MAIN FUNCTION
# ========================

def main(args=None):
    """
    Main entry point for the enhanced vehicle control node.
    
    Args:
        args: Command line arguments (optional)
    """
    rclpy.init(args=args)
    node = NewControlNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Control node interrupted by user")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()