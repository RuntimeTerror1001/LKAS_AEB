#!/usr/bin/env python3
import numpy as np
import os
import math
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

UNKNOWN_CLASS_ID = 65535

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

        control_defaults = control_params['control']
        self.default_lane_width = control_defaults.get('default_lane_width', 3.5)
        self.obstacle_path_lateral_margin = control_defaults.get('obstacle_path_lateral_margin', 0.5)
        self.obstacle_projection_window = int(control_defaults.get('obstacle_projection_window', 75))

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

        # Debug
        n = len(msg.obstacles)
        self.get_logger().debug(
            f"[obstacles_cb] received {n} obstacles "
            f"frame={msg.header.frame_id} "
            f"stamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}"
        )

        for i, obs in enumerate(msg.obstacles[:3]):  # log first 3 for sanity
            track_id = getattr(obs, 'track_id', -1)
            rel_v = getattr(obs, 'relative_speed', float('nan'))
            self.get_logger().debug(
                f"  RawObs[{i}]: id={track_id}, class={obs.class_id}, "
                f"pos=({obs.position.x:.2f},{obs.position.y:.2f}), "
                f"rel_v={rel_v:.2f}"
            )

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
    
    @staticmethod
    def _point_to_segment_distance(px, py, x1, y1, x2, y2):
        """
        Calculate perpendicular distance from point to line segment.
        
        Args:
            px, py: Point coordinates
            x1, y1, x2, y2: Line segment endpoints
        
        Returns:
            float: Perpendicular distance from point to segment
        """
        seg_dx = x2 - x1
        seg_dy = y2 - y1
        seg_len_sq = seg_dx**2 + seg_dy**2
        
        if seg_len_sq == 0.0:
            return math.hypot(px - x1, py - y1)
        
        # Project point onto line segment
        t = ((px - x1) * seg_dx + (py - y1) * seg_dy) / seg_len_sq
        t = max(0.0, min(1.0, t))  # Clamp to segment
        
        # Find closest point on segment
        closest_x = x1 + t * seg_dx
        closest_y = y1 + t * seg_dy
        
        return math.hypot(px - closest_x, py - closest_y)
    
    def filter_obstacles_on_path(self, obstacles):
        """
        Filter obstacles to those that lie within a lateral corridor around a planned path

        Args:
            obstacles (Sequence[Obstacle]) : Obstacles expressed in the base_link frame.

        Returns:
            list: Obstacles whose projected position falls within the allowable lateral envelope.
        """
        if not obstacles:
            return []
        
        # ====================
        # SETUP
        # ====================
        lane_width = self.default_lane_width
        if self.lane_info and self.lane_info.lane_width > 0.0:
            lane_width = self.lane_info.lane_width
        
        base_lateral_limit = (lane_width / 2.0) + self.obstacle_path_lateral_margin

        #====================
        # VALIDATE PATH
        #====================
        pp = self.pure_pursuit

        if self.curr_pose is None or not pp.waypoints:
            self.get_logger().warn(
                "[filter] No path available, skipping path-based filtering"
            )
            return list(obstacles)
        
        if len(pp.waypoints) < 2:
            self.get_logger().warn(
                "[filter] Too few waypoints, skipping path-based filtering"
            )
            return list(obstacles)
        
        vehicle_loc = self.curr_pose.location
        yaw_rad = math.radians(self.curr_pose.rotation.yaw)
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)

        #====================
        # FIND CLOSEST WAYPOINT TO EGO VEHICLE
        #====================
        # Use Pure Pursuit's existing logic to find closest waypoint
        closest_idx = pp.find_closest_waypoint(
            max(0, pp.prev_target_idx - 5)
        )

        closest_wp = pp.waypoints[closest_idx]
        closest_wp_dist = math.sqrt(
            (closest_wp[0] - vehicle_loc.x)**2 +
            (closest_wp[1] - vehicle_loc.y)**2
        )

        # If closest waypoint is still far (>30m)
        if closest_wp_dist > 30.0:
            self.get_logger().warn(
                f"[filter] Closest waypoint {closest_wp_dist:.1f}m away, "
                f"skipping path-based filtering"
            )
            return list(obstacles)
        
        #====================
        # BUILD RELEVANT PATH SEGMENT
        #====================
        # Start from closest waypoint, collect wps w/i 30.0m
        max_path_dist = 30.0

        relevant_waypoints = []
        path_distance_accumulated = 0.0

        # Start from closest waypoint
        for idx in range(closest_idx, len(pp.waypoints)):
            wx, wy = pp.waypoints[idx]

            # Add this waypoint
            relevant_waypoints.append((wx, wy))

            # Calculate distance to next waypoint
            if idx + 1 < len(pp.waypoints):
                next_wx, next_wy = pp.waypoints[idx+1]
                segment_length = math.sqrt(
                    (next_wx - wx)**2 + (next_wy - wy)**2
                )
                path_distance_accumulated += segment_length

                if path_distance_accumulated > max_path_dist:
                    break
        
        if len(relevant_waypoints) < 2:
            self.get_logger().warn(
                "[filter] Not enough waypoints ahead, skipping path based filtering"
            )
            return list(obstacles)
        
        self.get_logger().debug(
            f"[PATH] Using {len(relevant_waypoints)} waypoints "
            f"covering {path_distance_accumulated:.1f}m of path "
            f"(starting from idx={closest_idx})"
        )

        #====================
        # DETECT IMMEDIATE PATH CURVATURE
        #====================
        # Only check next 15m of path for curvature
        path_lateral_positions = []
        for wx, wy in relevant_waypoints:
            # Calculate distance from vehicle
            dx = wx - vehicle_loc.x
            dy = wy - vehicle_loc.y
            dist_to_wp = math.sqrt(dx*dx + dy*dy)

            # Only check waypoints w/i 15m
            if dist_to_wp > 15.0:
                break

            # Calculate lateral offset in vehicle frame
            lateral_offset = -dx * sin_yaw + dy * cos_yaw
            path_lateral_positions.append(lateral_offset)
        
        is_curved_path = False
        if len(path_lateral_positions) >= 3:
            path_lateral_range = max(path_lateral_positions) - min(path_lateral_positions)
            # Real Curve : 1-4m lateral range in next 15m
            is_curved_path = (1.0 < path_lateral_range < 4.0)

            if path_lateral_range > 4.0:
                # Distant turn, not immediate curve
                is_curved_path = False
                self.get_logger().debug(
                    f"[PATH] Ignoring distant turn: lateral_range={path_lateral_range:.2f}m"
                )
            elif is_curved_path:
                self.get_logger().debug(
                    f"[PATH] Curved path detected: lateral_range={path_lateral_range:.2f}m"
                )
        
        #====================
        # FILTER OBSTACLES
        #====================
        filtered = []

        for obstacle in obstacles:
            base_link_position = getattr(obstacle, 'position', None)
            if base_link_position is None:
                continue

            distance = getattr(base_link_position, 'x', None)
            lateral = getattr(base_link_position, 'y', None)
            if distance is None or lateral is None:
                continue

            # Skip obstacles behind vehicle
            if distance < 0.0:
                continue

            obstacle_id = getattr(obstacle, 'track_id', 0)
            class_id = getattr(obstacle, 'class_id', UNKNOWN_CLASS_ID)

            #====================
            # LAYER 1: EMERGENCY ZONE (0-8m)
            #====================
            if distance < 8.0:
                # For very close objects, use simple base link check
                if abs(lateral) < 2.0:
                    filtered.append(obstacle)
                    self.get_logger().warn(
                        f"[EMERGENCY] dist={distance:.2f}m, lat={lateral:.2f}m "
                        f"class={class_id}, id={obstacle_id}"
                    )
                    continue
                else:
                    self.get_logger().debug(
                        f"[EMERGENCY REJECT] dist={distance:.2f}m, lat={lateral:.2f}m "
                        f"(outside +-2m corridor)"
                    )
                    continue

            # Transform obstacle to world frame
            world_x = vehicle_loc.x + distance * cos_yaw - lateral * sin_yaw
            world_y = vehicle_loc.y + distance * sin_yaw + lateral * cos_yaw

            min_lateral_offset = float('inf')

            for i in range(len(relevant_waypoints) - 1):
                x1, y1 = relevant_waypoints[i]
                x2, y2 = relevant_waypoints[i+1]

                lateral_offset = self._point_to_segment_distance(
                    world_x, world_y,  x1, y1, x2, y2
                )

                if lateral_offset < min_lateral_offset:
                    min_lateral_offset = lateral_offset 
            
            #====================
            # LAYER 2: CRITICAL ZONE (8-15m)
            #====================
            if distance < 15.0:
                critical_lateral = base_lateral_limit * 1.2

                if min_lateral_offset <= critical_lateral:
                    filtered.append(obstacle)
                    self.get_logger().warn(
                        f"[CRITICAL] dist={distance:.2f}m, "
                        f"lat_to_path={min_lateral_offset:.2f}m "
                        f"class={class_id}, id={obstacle_id}"
                    )
                    continue
                else:
                    self.get_logger().debug(
                        f"[CRITICAL REJECT] dist={distance:.2f}m "
                        f"lat_to_path={min_lateral_offset:.2f}m"
                    )
                    continue
            
            #====================
            # LAYER 3: WARNING ZONE (15-25m)
            #====================
            elif distance < 25.0:
                # Matched camera detections - more lenient
                if class_id in [0,2,3,5,7,12] and obstacle_id < 65535:
                    warning_lateral = base_lateral_limit
                    if is_curved_path:
                        warning_lateral = (lane_width / 2.0) + 0.5

                    if min_lateral_offset <= warning_lateral:
                        filtered.append(obstacle)
                        self.get_logger().warn(
                            f"[WARNING] Matched: dist={distance:.1f}m, "
                            f"lat_to_path={min_lateral_offset:.2f}m, class={class_id}"
                        )
                        continue
                
                # Unmatched LiDAR - strict filtering
                elif obstacle_id == 0 or obstacle_id >= 65535:
                    unmatched_lateral = 1.2
                    if is_curved_path:
                        unmatched_lateral = 1.0

                    if min_lateral_offset <= unmatched_lateral:
                        filtered.append(obstacle)
                        self.get_logger().warn(
                            f"[WARNING] Unmatched: dist={distance:.1f}m, "
                            f"lat_to_path={min_lateral_offset:.2f}m"
                        )
                        continue
                    else:
                        self.get_logger().debug(
                            f"[WARNING REJECT] Unmatched roadside: "
                            f"dist={distance:.1f}m, lat_to_path={min_lateral_offset:.2f}m"
                        )
                        continue
            
            #====================
            # LAYER 4: FAR ZONE (>25m)
            #====================
            else:
                # Only keep well-classified obstacles on centerline
                if class_id in [0, 2, 3, 5, 7, 12] and obstacle_id < 65535:
                    far_lateral = (lane_width / 2.0) + 0.3
                    
                    if min_lateral_offset <= far_lateral:
                        filtered.append(obstacle)
                        self.get_logger().debug(
                            f"[FAR] dist={distance:.1f}m, "
                            f"lat_to_path={min_lateral_offset:.2f}m"
                        )
                        continue
                
                # Reject all unmatched distant obstacles
                self.get_logger().debug(
                    f"[FAR REJECT] dist={distance:.1f}m, "
                    f"lat_to_path={min_lateral_offset:.2f}m, "
                    f"class={class_id}, id={obstacle_id}"
                )
                continue
        
        self.get_logger().debug(
            f"[filter_obstacles_on_path] kept {len(filtered)}/{len(obstacles)} obstacles"
        )
        
        return filtered

    def calculate_obstacle_speed_factor(self, obstacles, curr_speed):
        """
        Calculate speed reduction factor based on obstacles with progressive reduction.
        
        Args:
            obstacles: List of detected obstacles
            curr_speed: Current ego vehicle speed
            
        Returns:
            tuple: (speed_factor, closest_distance, emergency_stop_required)
        """
        self.get_logger().debug(
            f"[AEB] input: {len(obstacles)} obstacles, curr_speed={curr_speed:.2f}m/s"
        )

        if not obstacles:
            return 1.0, float('inf'), False
        
        # Find closest obstacle with class-specific considerations
        closest_distance = float('inf')
        closest_class = None
        
        for obstacle in obstacles:
            base_link_position = getattr(obstacle, 'position', None)
            if base_link_position is None:
                continue
            
            distance = getattr(base_link_position, 'x', None)
            if distance is None:
                continue

            if distance < closest_distance:
                closest_distance = distance
                closest_class = getattr(obstacle, 'class_id', UNKNOWN_CLASS_ID)
        
        if closest_distance == float('inf'):
            return 1.0, float('inf'), False
        
        self.get_logger().debug(
            f"[AEB] closest_obstacle: dist={closest_distance:.2f}m, "
            f"class={closest_class}"
        )
        
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
            self.get_logger().info(f"CAUTION ZONE: {closest_distance:.1f}m - Speed factor: {speed_factor:.2f}")
            
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

        obs_age = curr_time - obs_time
        raw_obs_count = len(self.obstacles) if self.obstacles is not None else 0
        self.get_logger().debug(
            f"[control_loop] obstacle_age={obs_age:.3f}s, raw_count={raw_obs_count}"
        )

        # Filter stale obstacle detections
        if curr_time - obs_time > 0.5:  # 500ms staleness threshold
            obstacles = []
        else:
            obstacles = list(self.obstacles)
        
        self.get_logger().debug(
            f"[control_loop] after freshness filter: {len(obstacles)} obstacles"
        )

        obstacles = self.filter_obstacles_on_path(obstacles)
        self.get_logger().debug(
            f"[control_loop] after path filter: {len(obstacles)} obstacles"
        )  

        # Calculate progressive speed reduction based on obstacles
        obstacle_speed_factor, closest_distance, emergency_stop = self.calculate_obstacle_speed_factor(obstacles, self.curr_speed)

        self.get_logger().debug(
            f"[control_loop] AEB summary: closest={closest_distance:.2f}m, "
            f"speed_factor={obstacle_speed_factor:.2f}, emergency={emergency_stop}"
        )

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
            self.get_logger().info(f"HOLDING BRAKE: Speed={self.curr_speed:.1f}, Distance={closest_distance:.1f}")
        
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