import math
import numpy as np
from geometry_msgs.msg import Point

"""
PURE PURSUIT PATH FOLLOWING CONTROLLER
"""
class PurePursuit:
    def __init__(self, params):
        """
        Initialize Pure Pursuit controller with vehicle and algorithm parameters.
        
        Args:
            params (dict): Control parameters containing vehicle specs and tuning values
        """
        self.params = params['control']
        self.lookahead = self.params['lookahead_distance']
        self.wheelbase = self.params['wheelbase']
        self.max_steer = self.params['max_steering_angle']
        
        # ========================
        # STATE VARIABLES
        # ========================
        self.waypoints = []      # List of (x, y) waypoint coordinates
        self.curr_pose = None    # Current vehicle pose (x, y, yaw)

        # ========================
        # ADAPTIVE LOOKAHEAD PARAMETERS
        # ========================
        self.min_lookahead = 5.0    # Minimum lookahead distance
        self.max_lookahead = 20.0   # Maximum lookahead distance
        self.speed_gain = 0.2       # Lookahead increase per m/s of speed

        # ========================
        # PERFORMANCE OPTIMIZATION
        # ========================
        self.prev_target_idx = 0    # Remember last target index for efficiency
        
        # Steering smoothing
        self.steering_history = []
        self.max_steering_history = 3
    
    def update_path(self, waypoints_list):
        """
        Update the path to follow from CARLA Transform waypoints.
        
        Args:
            waypoints_list (list): List of CARLA Transform objects containing waypoint locations
        """
        self.waypoints = []
        for transform in waypoints_list:
            location = transform.location
            self.waypoints.append((location.x, location.y))

        # Restart target index when path changes
        self.prev_target_idx = 0
    
    def update_pose(self, pose_transform):
        """
        Update current vehicle pose from CARLA Transform.
        
        Args:
            pose_transform: CARLA Transform object with vehicle location and rotation
        """
        loc = pose_transform.location
        rot = pose_transform.rotation

        x = loc.x
        y = loc.y

        yaw = math.radians(rot.yaw) # to radians

        self.curr_pose = (x, y, yaw)
    
    def find_closest_waypoint(self, start_idx=0):
        """
        Find the closest waypoint to current position with optimized search.
        
        Args:
            start_idx (int): Starting index for search optimization
            
        Returns:
            int: Index of closest waypoint
        """
        if not self.waypoints or self.curr_pose is None:
            return 0
            
        curr_x, curr_y, _ = self.curr_pose
        closest_idx = start_idx
        min_dist = float('inf')

        # Forward search from start_idx (more efficient for sequential waypoints)
        search_range = min(len(self.waypoints), start_idx + 50)  # Limit search range
        
        for i in range(start_idx, search_range):
            wx, wy = self.waypoints[i]
            dist = np.hypot(wx - curr_x, wy - curr_y)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # Backward search if no close point found (handles path discontinuities)
        if min_dist > 20.0 and start_idx > 0:
            for i in range(start_idx - 1, max(0, start_idx - 20), -1):
                wx, wy = self.waypoints[i]
                dist = np.hypot(wx - curr_x, wy - curr_y)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
        
        return closest_idx

    def calculate_lookahead_point(self, current_speed):
        """
        Find target point at adaptive lookahead distance along the path.
        
        Args:
            current_speed (float): Current vehicle speed in m/s for adaptive lookahead
            
        Returns:
            tuple: ((target_x, target_y), lookahead_distance) or (None, 0) if no path
        """
        if not self.waypoints or self.curr_pose is None:
            return None, 0
        
        # ========================
        # ADAPTIVE LOOKAHEAD CALCULATION
        # ========================
        adaptive_lookahead = self.min_lookahead + self.speed_gain * current_speed
        lookahead = np.clip(adaptive_lookahead, self.min_lookahead, self.max_lookahead)

        # Find closest waypoint (start from previous target for efficiency)
        curr_x, curr_y, curr_yaw = self.curr_pose
        closest_idx = self.find_closest_waypoint(max(0, self.prev_target_idx - 5))
        
        # ========================
        # ENSURE FORWARD-LOOKING BEHAVIOR
        # ========================
        if closest_idx < len(self.waypoints) - 1:
            # Check if closest waypoint is behind vehicle
            waypoint_vec = [self.waypoints[closest_idx][0] - curr_x,
                           self.waypoints[closest_idx][1] - curr_y]
            forward_vec = [np.cos(curr_yaw), np.sin(curr_yaw)]
            
            # Skip waypoints that are behind the vehicle
            while (closest_idx < len(self.waypoints) - 1 and 
                   np.dot(waypoint_vec, forward_vec) < 0):
                closest_idx += 1
                if closest_idx < len(self.waypoints):
                    waypoint_vec = [self.waypoints[closest_idx][0] - curr_x,
                                   self.waypoints[closest_idx][1] - curr_y]
        
        # ========================
        # FIND LOOKAHEAD POINT ALONG PATH
        # ========================
        total_dist = 0.0
        
        for i in range(closest_idx, len(self.waypoints) - 1):
            x1, y1 = self.waypoints[i]
            x2, y2 = self.waypoints[i + 1]
            
            # Distance from current position to start of segment
            dist_to_segment_start = np.hypot(x1 - curr_x, y1 - curr_y)
            
            # Account for distance to segment start on first segment 
            if i == closest_idx:
                total_dist = dist_to_segment_start
            
            segment_length = np.hypot(x2 - x1, y2 - y1)
            
            # Check if lookahead point is within this segment
            if total_dist + segment_length >= lookahead:
                # Interpolate within this segment
                remaining_dist = lookahead - total_dist
                if segment_length > 0:
                    ratio = remaining_dist / segment_length
                    target_x = x1 + ratio * (x2 - x1)
                    target_y = y1 + ratio * (y2 - y1)
                else:
                    target_x, target_y = x1, y1
                
                self.prev_target_idx = i  # Remember for next iteration
                return (target_x, target_y), lookahead
                
            total_dist += segment_length
        
        # Return last point if lookahead extends beyond path
        if self.waypoints:
            self.prev_target_idx = len(self.waypoints) - 1
            return self.waypoints[-1], lookahead
        
        return None, 0

    def calculate_steering(self, current_speed):
        """
        Calculate steering angle using Pure Pursuit algorithm with speed-dependent smoothing.
        
        Args:
            current_speed (float): Current vehicle speed in m/s
            
        Returns:
            float: Steering angle in radians (positive = left, negative = right)
        """
        # Get target lookahead point
        target_result = self.calculate_lookahead_point(current_speed)
        if target_result[0] is None or self.curr_pose is None:
            return 0.0
            
        target_point, lookahead = target_result
        tx, ty = target_point
        cx, cy, cyaw = self.curr_pose
        
        # ========================
        # TRANSFORM TARGET TO VEHICLE COORDINATES
        # ========================
        dx = tx - cx
        dy = ty - cy
        target_x = dx * np.cos(cyaw) + dy * np.sin(cyaw)
        target_y = -dx * np.sin(cyaw) + dy * np.cos(cyaw)
        
        # Handle edge case where target is at vehicle position
        if abs(target_x) < 0.1 and abs(target_y) < 0.1:
            return 0.0
        
        # ========================
        # PURE PURSUIT STEERING CALCULATION
        # ========================
        alpha = np.arctan2(target_y, target_x)
        
        # Avoid division by zero in steering calculation
        if lookahead < 0.1:
            lookahead = 0.1
            
        steering = np.arctan(2 * self.wheelbase * np.sin(alpha) / lookahead)
        
        # ========================
        # SPEED DEPENDENT STEERING ADJUSTMENTS
        # ========================
        if current_speed > 10.0:  # (m/s) High Speed Smoothing
            curvature = abs(2 * np.sin(alpha) / lookahead)  
            speed_factor = min(1.0, 15.0 / current_speed)
            smoothing_factor = 1.0 - 0.6 * curvature * speed_factor
            steering *= smoothing_factor
    
        elif current_speed < 3.0: # Low Speed oscillation prevention
            steering *= 0.7

        # ========================
        # STEERING HISTORY SMOOTHING
        # ========================
        self.steering_history.append(steering)
        if len(self.steering_history) > self.max_steering_history:
            self.steering_history.pop(0)
        
        # Average recent steering commands for smoother control
        if len(self.steering_history) > 1:
            steering = np.mean(self.steering_history)


        # ========================
        # FINAL SAFETY CHECKS
        # ========================
        # Clip to maximum steering angle
        steering = np.clip(steering, -self.max_steer, self.max_steer)
        
        # Safety check for NaN values
        steering = np.nan_to_num(steering, nan=0.0)
            
        return steering
    
    def get_lookahead_marker(self, target_point):
        """
        Get ROS Point marker for visualization of current lookahead point.
        
        Args:
            target_point (tuple): Target point coordinates (x, y) or None
            
        Returns:
            Point: ROS Point message for visualization
        """
        if target_point is None:
            marker = Point()
            marker.x = 0.0
            marker.y = 0.0
            marker.z = 0.0
            return marker
            
        marker = Point()
        marker.x = target_point[0]
        marker.y = target_point[1]
        marker.z = 0.0
        return marker
    
    def get_debug_info(self):
        """
        Return debug information for system monitoring and parameter tuning.
        
        Returns:
            dict: Debug information containing controller state and performance metrics
        """
        return {
            'prev_target_idx': self.prev_target_idx,
            'num_waypoints': len(self.waypoints),
            'steering_history': self.steering_history.copy() if self.steering_history else [],
            'current_pose': self.curr_pose
        }