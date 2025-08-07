import numpy as np
from lkas_aeb_msgs.msg import TTC  

"""
AUTOMATIC EMERGENCY BRAKING CONTROLLER
"""

class AEBController:
    def __init__(self, params, logger):
        """
        Initialize the AEB Controller with parameters and logger.
        
        Args:
            params (dict): Control parameters containing AEB configuration
            logger: ROS logger instance for debugging and status messages
        """
        self.params = params['control']
        self.critical_ttc = self.params['critical_ttc']
        self.max_brake = self.params['max_brake_force']
        self.min_obstacle_size = self.params['min_obstacle_size']
        self.logger = logger
        
        # ========================
        # OBSTACLE TRACKING SYSTEM
        # ========================
        self.obstacle_history = {}  # track_id: (distance, timestamp, speed, confidence)
        self.last_update_time = None

        # ========================
        # SAFETY PARAMETERS
        # ========================
        self.max_reasonable_speed = 50.0    # m/s (180 kmph) - Maximum believable relative speed
        self.min_reasonable_distance = 0.5  # m - Minimum valid obstacle distance
        self.max_reasonable_distance = 200.0 # m - Maximum detection range
        self.history_timeout = 5.0           # seconds - Clean up stale obstacle tracks

        # ========================
        # MULTI-STAGE BRAKING THRESHOLDS
        # ========================
        self.warning_ttc = self.critical_ttc * 1.5   # Early warning stage
        self.emergency_ttc = self.critical_ttc * 0.5 # Full emergency braking

        # Distance-based emergency braking for stationary objects
        self.emergency_distance = 8.0   # Emergency brake if closer than 8m
        self.warning_distance = 15.0    # Warning brake if closer than 15m

    def validate_obstacle_data(self, obstacle):
        """
        Validate obstacle data for reasonableness and consistency.
        
        Args:
            obstacle: Obstacle message with distance, bbox, class_id fields
            
        Returns:
            bool: True if obstacle data is valid, False otherwise
        """
        # Check distance bounds
        if not (self.min_reasonable_distance <= obstacle.distance <= self.max_reasonable_distance):
            return False
            
        # Check bbox size
        if len(obstacle.bbox) != 4:
            return False
        
        # Check minimum obstacle size to filter noise
        bbox_area = (obstacle.bbox[2] - obstacle.bbox[0]) * (obstacle.bbox[3] - obstacle.bbox[1])
        if bbox_area < self.min_obstacle_size:
            return False
            
        return True

    def calculate_relative_speed(self, obstacle, track_id, timestamp, dt, ego_speed):
        """
        Calculate relative speed between ego vehicle and obstacle, with special handling for stationary objects.
        
        Args:
            obstacle: Obstacle message containing distance and other data
            track_id (int): Unique identifier for tracking this obstacle
            timestamp (float): Current timestamp in seconds
            dt (float): Time delta since last update
            ego_speed (float): Current ego vehicle speed in m/s
            
        Returns:
            float: Relative approach speed (positive = approaching, 0 = no threat)
        """
        track_id = int(track_id)

        # Initialize new obstacle track        
        if track_id not in self.obstacle_history:
            self.obstacle_history[track_id] = {
                'distance': obstacle.distance,
                'timestamp': timestamp,
                'speed': 0.0,  # Start with 0 relative speed
                'confidence': 1.0
            }
            return ego_speed  # For new obstacles, assume relative speed = ego speed

        # Get previous tracking data
        prev_data = self.obstacle_history[track_id]
        prev_dist = prev_data['distance']
        prev_speed = prev_data['speed']
        prev_confidence = prev_data['confidence']
        
        # Calculate distance change (negative = approaching)
        distance_change = obstacle.distance - prev_dist
        
        # Validate distance change
        if abs(distance_change) > 20 * dt:  # Sanity check: max 20 m/s relative speed change
            filtered_speed = prev_speed * 0.9  # Gradually reduce if no valid update
            self.logger.debug(f'Impossible Speed Change: {abs(distance_change/dt):.1f} m/s')
        else:
            # Calculate relative speed based on distance change
            if abs(distance_change) < 0.5 * dt:  # Nearly stationary relative motion
                # Object is stationary relative to ego - use ego speed as relative speed
                raw_relative_speed = ego_speed
            else:
                # Object has significant relative motion
                range_rate = -distance_change / dt  # Negative because approaching objects have decreasing distance
                raw_relative_speed = range_rate
            
            # Apply Kalman-like filtering for smoothness
            alpha = 0.4 if abs(raw_relative_speed - prev_speed) < 5 else 0.2
            filtered_speed = alpha * raw_relative_speed + (1 - alpha) * prev_speed
            
            # Bound speed to reasonable values 
            filtered_speed = np.clip(filtered_speed, -10.0, self.max_reasonable_speed)

        # Update confidence based on measurement consistency
        speed_consistency = 1.0 - min(1.0, abs(filtered_speed - prev_speed) / 10.0)
        new_confidence = 0.8 * prev_confidence + 0.2 * speed_consistency
        
        # Update history with dictionary
        self.obstacle_history[track_id] = {
            'distance': obstacle.distance,
            'timestamp': timestamp,
            'speed': filtered_speed,
            'confidence': new_confidence
        }
        
        return max(0.0, filtered_speed)  # Only consider approaching objects (positive relative speed)

    def calculate_ttc(self, obstacles, curr_speed, timestamp):
        """
        Calculate Time-To-Collision for the most critical obstacle with distance-based fallback.
        
        Args:
            obstacles (list): List of detected obstacle messages
            curr_speed (float): Current ego vehicle speed in m/s
            timestamp (float): Current timestamp in seconds
            
        Returns:
            TTC: Message containing time to collision, distance, and criticality information
        """
        ttc_msg = TTC()
        ttc_msg.ttc = float('inf')
        ttc_msg.critical = False
        ttc_msg.distance = float('inf')
        ttc_msg.relative_speed = 0.0
        ttc_msg.obstacle_id = 65535

        if not obstacles:
            return ttc_msg

        # Calculate time delta for relative speed estimation
        dt = 0.1  # Default dt
        if self.last_update_time is not None:
            dt = max(0.01, min(0.5, timestamp - self.last_update_time))
        self.last_update_time = timestamp

        # ========================
        # MOST CRITICAL OBSTACLE
        # ========================
        min_ttc = float('inf')
        critical_obstacle = None
        closest_distance = float('inf')
        
        for obstacle in obstacles:
            # Validate obstacle data
            if not self.validate_obstacle_data(obstacle):
                continue
                
            # Extract track ID - must be integer
            track_id = getattr(obstacle, 'track_id', None)
            if track_id is None:
                continue  # Skip if no track ID
                
            try:
                track_id = int(track_id)  # Ensure integer type
            except (TypeError, ValueError):
                continue  # Skip if invalid track ID
            
            # Calculate relative speed with ego speed consideration
            relative_speed = self.calculate_relative_speed(obstacle, track_id, timestamp, dt, curr_speed)
            
            # Track closest obstacle for distance-based braking
            if obstacle.distance < closest_distance:
                closest_distance = obstacle.distance
            
            # Calculate TTC if there's significant relative speed
            ttc = float('inf')
            if relative_speed > 0.5:  # Reduced threshold for better detection
                ttc = obstacle.distance / relative_speed
                
                # Apply safety margins based on object class
                safety_factor = 1.0
                if obstacle.class_id in [0]:  # Person - more conservative
                    safety_factor = 0.8
                elif obstacle.class_id in [5, 7]:  # Bus, truck - larger objects
                    safety_factor = 1.2
                    
                adjusted_ttc = ttc * safety_factor
                
                # Consider obstacle confidence in decision
                confidence = self.obstacle_history.get(track_id, {}).get('confidence', 0.5)
                if confidence > 0.5 and adjusted_ttc < min_ttc:  # Reduced confidence threshold
                    min_ttc = adjusted_ttc
                    critical_obstacle = obstacle
                    ttc_msg.distance = obstacle.distance
                    ttc_msg.relative_speed = relative_speed
                    ttc_msg.obstacle_id = min(obstacle.class_id, 65535)

        # ========================
        # DISTANCE-BASED EMERGENCY DETECTION
        # ========================
        # Force critical TTC for very close objects where TTC calculation might be unreliable
        if closest_distance < self.emergency_distance:
            if closest_distance < min_ttc * curr_speed or min_ttc == float('inf'):
                min_ttc = max(0.1, closest_distance / max(curr_speed, 1.0))  # Minimum TTC of 0.1s
                ttc_msg.distance = closest_distance
                ttc_msg.relative_speed = curr_speed  # Assume approaching at ego speed
                ttc_msg.critical = True
                self.logger.warning(f"EMERGENCY: Very close obstacle at {closest_distance:.1f}m!")

        # ========================
        # OBSTACLE HISTORY CLEANUP
        # ========================
        # Remove expired obstacle tracks to prevent memory leaks
        expired_tracks = [
            track_id for track_id, data in self.obstacle_history.items()
            if (timestamp - data['timestamp']) > self.history_timeout
        ]
        for track_id in expired_tracks:
            del self.obstacle_history[track_id]

        # ========================
        # UPDATE TTC MESSAGE
        # ========================
        if critical_obstacle or closest_distance < self.warning_distance:
            ttc_msg.ttc = min_ttc
            ttc_msg.critical = bool(min_ttc < self.critical_ttc)
            
            # if min_ttc != float('inf'):
            #     self.logger.info(
            #         f"Critical obstacle: TTC={min_ttc:.1f}s, "
            #         f"Dist={ttc_msg.distance:.1f}m, "
            #         f"RelSpeed={ttc_msg.relative_speed:.1f}m/s, "
            #         f"EgoSpeed={curr_speed:.1f}m/s"
            #     )

        return ttc_msg
    
    def decide_braking(self, ttc_msg, curr_speed):
        """
        Determine appropriate braking force based on TTC and distance with multi-stage response.
        
        Args:
            ttc_msg (TTC): Time-to-collision message with threat information
            curr_speed (float): Current ego vehicle speed in m/s
            
        Returns:
            float: Brake force (0.0 to max_brake_force)
        """
        distance = ttc_msg.distance
        ttc = ttc_msg.ttc

        # ========================
        # IMMEDIATE SAFETY OVERRIDE
        # ========================
        # Very close obstacles - immediate max braking
        if distance < 5.0:  # Critical distance
            brake_force = self.max_brake
            self.logger.warning(f"CRITICAL DISTANCE: {distance:.1f}m - MAXIMUM BRAKE!")
            return np.clip(brake_force, 0.0, self.max_brake)
        
        # ========================
        # DISTANCE-BASED EMERGENCY BRAKING
        # ========================
        if distance < self.emergency_distance:
            # Progressive braking based on distance
            distance_severity = (self.emergency_distance - distance) / self.emergency_distance
            brake_force = 0.6 + 0.4 * distance_severity  # 60-100% brake force
            brake_force *= self.max_brake
            self.logger.warning(f"DISTANCE EMERGENCY: {distance:.1f}m - Brake: {brake_force:.2f}")
            
        elif distance < self.warning_distance:
            # Warning zone - moderate braking
            distance_factor = (self.warning_distance - distance) / (self.warning_distance - self.emergency_distance)
            brake_force = 0.3 * self.max_brake * distance_factor
            self.logger.info(f"DISTANCE WARNING: {distance:.1f}m - Brake: {brake_force:.2f}")

        # ========================
        # TTC BASED BRAKING STAGES
        # ========================    
        elif ttc != float('inf'):
            if ttc < self.emergency_ttc:
                # Emergency: Full braking
                brake_force = self.max_brake
            elif ttc < self.critical_ttc:
                # Critical: Strong braking with progressive increase
                severity = 1.0 - (ttc / self.critical_ttc)
                brake_force = 0.6 * self.max_brake + 0.4 * self.max_brake * severity
            elif ttc < self.warning_ttc:
                # Warning: Light braking to alert driver
                severity = 1.0 - (ttc / self.warning_ttc)
                brake_force = 0.3 * self.max_brake * severity
            else:
                return 0.0
        else:
            return 0.0
        
        # ========================
        # SPEED DEPENDENT BRAKING ADJUSTMENTS
        # ========================
        if brake_force > 0:
            # Increase braking force at higher speeds
            speed_factor = min(1.5, 1.0 + curr_speed / 20.0)
            brake_force *= speed_factor
            
            # Ensure maximum brake force for very close obstacles
            if distance < 10.0:
                brake_force = max(brake_force, 0.4 * self.max_brake)
        
        # Final bounds checking
        brake_force = np.clip(brake_force, 0.0, self.max_brake)
        
        # ========================
        # ANTI-OSCILLATION SMOOTHING
        # ========================
        # Add hysteresis to prevent oscillation
        if hasattr(self, 'prev_brake_force'):
            # Smooth transitions
            if brake_force > self.prev_brake_force:
                # Increasing brake - allow immediate response
                pass
            else:
                # Decreasing brake - smooth transition
                brake_force = 0.8 * brake_force + 0.2 * self.prev_brake_force
        
        self.prev_brake_force = brake_force
        
        # Log braking decisions
        # if brake_force > 0.1:
        #     self.logger.info(f"AEB ACTIVE: Brake={brake_force:.2f}, TTC={ttc:.1f}s, Dist={distance:.1f}m")
        
        return brake_force
    
    def get_debug_info(self):
        """
        Return comprehensive debug information for system monitoring and parameter tuning.
        
        Returns:
            dict: Debug information containing tracking status and threshold values
        """
        return {
            'tracked_obstacles': len(self.obstacle_history),
            'obstacle_history': self.obstacle_history,
            'thresholds': {
                'warning_ttc': self.warning_ttc,
                'critical_ttc': self.critical_ttc,
                'emergency_ttc': self.emergency_ttc,
                'emergency_distance': self.emergency_distance,
                'warning_distance': self.warning_distance
            }
        }