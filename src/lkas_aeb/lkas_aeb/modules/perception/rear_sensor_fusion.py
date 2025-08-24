#!/usr/bin/env python3
"""
FIXED: Refactored Rear Sensor Fusion Module

Fixed array boolean comparison issues that were causing the ambiguous truth value error.
Key fixes:
1. Proper array element checking instead of direct boolean operations
2. Use of .any()/.all() where appropriate  
3. Fixed array comparisons in validation functions
"""

from typing import Dict, List, Optional, Tuple
import time
import math
import numpy as np
from lkas_aeb_msgs.msg import Obstacle, ObstacleArray

# Import base classes and utilities
from .base_classes import BaseFusion, BaseKalmanTracker, ParameterValidator, TrackState
from lkas_aeb.util.perception_utils import (
    validate_bbox, validate_position_3d, validate_numeric,
    bbox_overlap, angle_difference, weighted_average,
    ProcessingError, time_function
)


class RearKalmanTracker(BaseKalmanTracker):
    """Kalman filter tracker for rear sensor fusion"""
    
    def __init__(self, params: Dict):
        super().__init__(params, "RearKalmanTracker")
    
    def _initialize(self) -> None:
        """Initialize rear-specific Kalman filter parameters"""
        super()._initialize()
        
        # Measurement models for different sensors
        self.R_radar = np.diag([
            self.params.get('sigma_radar_x', 1.0)**2,
            self.params.get('sigma_radar_y', 1.0)**2
        ]).astype(float)
        
        self.R_camera = np.diag([
            self.params.get('sigma_camera_x', 2.0)**2,
            self.params.get('sigma_camera_y', 2.0)**2
        ]).astype(float)
        
        # Velocity measurement model for radar
        self.H_velocity = np.array([[0, 0, 1, 0],
                                   [0, 0, 0, 1]], dtype=float)
        self.R_velocity = np.diag([
            self.params.get('sigma_vel_x', 2.0)**2,
            self.params.get('sigma_vel_y', 2.0)**2
        ]).astype(float)
    
    def update_radar(self, track_id: int, position: np.ndarray, 
                    velocity: Optional[np.ndarray] = None) -> bool:
        """Update track with radar measurement"""
        if track_id not in self.tracks:
            return False
        
        track = self.tracks[track_id]
        
        # Update position with radar noise model
        original_R = self.R
        self.R = self.R_radar
        
        try:
            self.kalman_update(track, position)
            
            # Update velocity if available
            if velocity is not None:
                self._velocity_update(track, velocity)
            
            track.sensor_sources = list(set(track.sensor_sources + ['radar_rear']))
            return True
        finally:
            self.R = original_R
    
    def update_camera(self, track_id: int, position: np.ndarray) -> bool:
        """Update track with camera measurement"""
        if track_id not in self.tracks:
            return False
        
        track = self.tracks[track_id]
        
        # Update with camera noise model
        original_R = self.R
        self.R = self.R_camera
        
        try:
            self.kalman_update(track, position)
            track.sensor_sources = list(set(track.sensor_sources + ['camera_rear']))
            return True
        finally:
            self.R = original_R
    
    def _velocity_update(self, track: 'TrackState', velocity: np.ndarray) -> None:
        """Update track with velocity measurement"""
        try:
            z = np.array(velocity, dtype=float).reshape(-1, 1)
            x = track.position.reshape(-1, 1)
            P = track.covariance
            
            # Innovation
            y = z - self.H_velocity @ x
            
            # Innovation covariance
            S = self.H_velocity @ P @ self.H_velocity.T + self.R_velocity
            
            # Kalman gain
            K = P @ self.H_velocity.T @ np.linalg.inv(S)
            
            # Update state
            track.position = (x + K @ y).flatten()
            
            # Update covariance
            I = np.eye(len(x))
            track.covariance = (I - K @ self.H_velocity) @ P
            
        except np.linalg.LinAlgError:
            # Skip velocity update if matrix inversion fails
            pass
    
    def create_track_from_radar(self, position: np.ndarray, 
                              velocity: Optional[np.ndarray] = None) -> int:
        """Create new track from radar detection"""
        if velocity is not None:
            initial_state = np.array([position[0], position[1], velocity[0], velocity[1]])
            initial_cov = np.diag([2.0, 2.0, 5.0, 5.0])  # Lower pos uncertainty for radar
        else:
            initial_state = np.array([position[0], position[1], 0.0, 0.0])
            initial_cov = np.diag([2.0, 2.0, 10.0, 10.0])
        
        track = self._create_new_track(None, initial_state, initial_cov)
        track.sensor_sources = ['radar_rear']
        return track.id
    
    def create_track_from_camera(self, position: np.ndarray) -> int:
        """Create new track from camera detection"""
        initial_state = np.array([position[0], position[1], 0.0, 0.0])
        initial_cov = np.diag([5.0, 5.0, 15.0, 15.0])  # Higher uncertainty for camera-only
        
        track = self._create_new_track(None, initial_state, initial_cov)
        track.sensor_sources = ['camera_rear']
        return track.id

    def process(self, detections: List[np.ndarray] = None, dt: float = 0.1) -> List:
        """Generic processing path for tracker-only use."""
        start = time.time()
        try:
            # Predict step
            try:
                self.predict(float(dt))
            except Exception:
                self.predict(0.1)
            # Update with provided detections if any (assume [x, y] points)
            if detections:
                self.update(detections)
            # Prune stale tracks
            self.prune_tracks()
            # Return confirmed tracks
            return self.get_confirmed_tracks()
        finally:
            if hasattr(self, 'stats'):
                self._update_stats(start_time=start,
                                   input_count=len(detections) if detections else 0,
                                   output_count=len(getattr(self, 'tracks', {})))

    def update(self, detections: List[np.ndarray]) -> None:
        """Generic update: associate 2D measurements to tracks and apply a Kalman update."""
        if not detections:
            return

        # Normalize and filter measurements
        meas_list = []
        for d in detections:
            arr = np.asarray(d, dtype=float).reshape(-1)
            if arr.size >= 2 and np.isfinite(arr[:2]).all():
                meas_list.append(arr[:2])

        if not meas_list:
            return

        associations = self.associate(meas_list)

        # Temporarily switch to radar noise
        original_R = getattr(self, 'R', None)
        self.R = getattr(self, 'R_radar', original_R)

        try:
            for track_id, meas_idx, _ in associations:
                m = np.array(meas_list[meas_idx], dtype=float)
                self.kalman_update(self.tracks[track_id], m)
                # Tag source and timestamp
                self.tracks[track_id].sensor_sources = list(set((self.tracks[track_id].sensor_sources or []) + ['radar_rear']))
                self.tracks[track_id].last_update = time.time()
        finally:
            if original_R is not None:
                self.R = original_R


class RearSensorFusion(BaseFusion):
    """Rear sensor fusion module combining radar and camera"""
    
    def __init__(self, params: Dict):
        """
        Initialize rear sensor fusion
        
        Args:
            params: Fusion parameters
        """
        # Validate parameters
        self.fusion_params = ParameterValidator.validate_fusion_params(params)
        self.tracking_params = ParameterValidator.validate_kalman_params(
            params.get('tracking', {})
        )
        
        super().__init__(self.fusion_params, "RearSensorFusion")
    
    def _initialize(self) -> None:
        """Initialize fusion components"""
        # Create Kalman tracker
        self.tracker = RearKalmanTracker(self.tracking_params)
        
        # Cache frequently used parameters
        self.max_position_distance = self.fusion_params.get('max_position_distance', 3.0)
        self.max_bearing_diff = self.fusion_params.get('max_bearing_diff', 0.3)
        self.min_bbox_overlap = self.fusion_params.get('min_bbox_overlap', 0.1)
        
        # Fusion weights
        fusion_weights = self.fusion_params.get('fusion_weights', {})
        self.camera_weight = fusion_weights.get('camera', 0.7)
        self.radar_weight = fusion_weights.get('radar', 0.3)
    
    @time_function
    def process(self, radar_obs: ObstacleArray, camera_obs: ObstacleArray) -> ObstacleArray:
        """Main fusion processing function"""
        start_time = time.time()
        
        try:
            # Calculate time step
            current_time = time.time()
            dt = min(0.2, max(1e-3, current_time - getattr(self, '_last_time', current_time)))
            self._last_time = current_time
            
            # Predict all tracks
            self.tracker.predict(dt)
            
            # Process input data
            radar_detections = self._process_radar_obstacles(radar_obs)
            camera_detections = self._process_camera_obstacles(camera_obs)
            
            # Associate detections between sensors
            associations = self.associate_detections(radar_detections, camera_detections)
            
            # Update tracks with associated detections
            self._update_tracks_with_associations(associations, radar_detections, camera_detections)
            
            # Handle unmatched detections
            self._handle_unmatched_detections(associations, radar_detections, camera_detections)
            
            # Prune old tracks
            self.tracker.prune_tracks()
            
            # Generate output
            result = self._create_output_obstacles(radar_obs, camera_obs)
            
            # Update statistics
            self._update_stats(
                start_time,
                len(radar_detections) + len(camera_detections),
                len(result.obstacles)
            )
            
            return result
            
        except Exception as e:
            self._update_stats(start_time, 0, 0, had_error=True)
            raise ProcessingError(f"Rear fusion failed: {str(e)}")
    
    def _process_radar_obstacles(self, radar_obs: ObstacleArray) -> List[Obstacle]:
        """Process and validate radar obstacles"""
        if radar_obs is None or not hasattr(radar_obs, 'obstacles') or len(radar_obs.obstacles) == 0:
            return []
        
        processed = []
        for obs in radar_obs.obstacles:
            # FIXED: Validate position - proper array element checking
            position = validate_position_3d(getattr(obs, 'position_3d', None))
            
            # FIXED: Check if position is effectively zero using proper array element access
            if len(position) >= 2:
                # Check if both x and y are zero (invalid position)
                if abs(position[0]) < 1e-6 and abs(position[1]) < 1e-6:
                    continue
            else:
                continue
            
            # Update obstacle with validated data
            obs.position_3d = position
            obs.distance = validate_numeric(
                getattr(obs, 'distance', np.hypot(position[0], position[1])),
                np.hypot(position[0], position[1]),
                min_val=0.0
            )
            obs.relative_speed = validate_numeric(getattr(obs, 'relative_speed', 0.0), 0.0)
            obs.confidence = validate_numeric(getattr(obs, 'confidence', 0.6), 0.6, 0.0, 1.0)
            obs.point_count = max(0, int(getattr(obs, 'point_count', 0)))
            
            processed.append(obs)
        
        return processed
    
    def _process_camera_obstacles(self, camera_obs: ObstacleArray) -> List[Obstacle]:
        """Process and validate camera obstacles"""
        if camera_obs is None or not hasattr(camera_obs, 'obstacles') or len(camera_obs.obstacles) == 0:
            return []
        
        processed = []
        for obs in camera_obs.obstacles:
            # Validate bounding box
            bbox = validate_bbox(getattr(obs, 'bbox', None))
            if bbox[0] < 0:  # Invalid bbox
                continue
            
            # Update obstacle with validated data
            obs.bbox = bbox
            obs.confidence = validate_numeric(getattr(obs, 'confidence', 0.7), 0.7, 0.0, 1.0)
            obs.distance = validate_numeric(getattr(obs, 'distance', -1.0), -1.0, min_val=0.0)
            
            # Validate position if available
            position = validate_position_3d(getattr(obs, 'position_3d', None))
            obs.position_3d = position
            
            processed.append(obs)
        
        return processed
    
    def associate_detections(self, radar_detections: List[Obstacle], 
                           camera_detections: List[Obstacle]) -> List[Tuple[int, int, float]]:
        """Associate radar and camera detections"""
        if not radar_detections or not camera_detections:
            return []
        
        associations = []
        
        # Calculate association costs
        for radar_idx, radar_obs in enumerate(radar_detections):
            for camera_idx, camera_obs in enumerate(camera_detections):
                cost = self._calculate_association_cost(radar_obs, camera_obs)
                
                if cost < float('inf'):  # Valid association
                    associations.append((radar_idx, camera_idx, cost))
        
        # Sort by cost and resolve conflicts greedily
        associations.sort(key=lambda x: x[2])
        
        used_radar = set()
        used_camera = set()
        final_associations = []
        
        for radar_idx, camera_idx, cost in associations:
            if radar_idx not in used_radar and camera_idx not in used_camera:
                final_associations.append((radar_idx, camera_idx, cost))
                used_radar.add(radar_idx)
                used_camera.add(camera_idx)
        
        return final_associations
    
    def _calculate_association_cost(self, radar_obs: Obstacle, camera_obs: Obstacle) -> float:
        """Calculate association cost between radar and camera detections"""
        try:
            # Position distance cost
            pos_distance = self._calculate_position_distance(radar_obs, camera_obs)
            if pos_distance > self.max_position_distance:
                return float('inf')
            
            # Bearing difference cost
            bearing_diff = self._calculate_bearing_difference(radar_obs, camera_obs)
            if bearing_diff > self.max_bearing_diff:
                return float('inf')
            
            # Bounding box overlap cost (if available)
            bbox_cost = 0.5  # Neutral cost if no bbox overlap available
            if (hasattr(camera_obs, 'bbox') and len(camera_obs.bbox) >= 4 and camera_obs.bbox[0] >= 0 and
                hasattr(radar_obs, 'bbox') and len(radar_obs.bbox) >= 4 and radar_obs.bbox[0] >= 0):
                overlap = bbox_overlap(radar_obs.bbox, camera_obs.bbox)
                if overlap < self.min_bbox_overlap:
                    return float('inf')
                bbox_cost = 1.0 - overlap
            
            # Combine costs with weights
            pos_cost = pos_distance / self.max_position_distance
            bearing_cost = bearing_diff / self.max_bearing_diff
            
            total_cost = (0.4 * pos_cost + 0.4 * bearing_cost + 0.2 * bbox_cost)
            return total_cost
            
        except Exception:
            return float('inf')
    
    def _calculate_position_distance(self, obs1: Obstacle, obs2: Obstacle) -> float:
        """FIXED: Calculate position distance between two obstacles"""
        # Use 3D position if available
        p1 = getattr(obs1, 'position_3d', None)
        p2 = getattr(obs2, 'position_3d', None)
        
        if p1 is not None and p2 is not None:
            p1 = np.asarray(p1, dtype=float).ravel()
            p2 = np.asarray(p2, dtype=float).ravel()
            
            # FIXED: Check array elements individually instead of boolean operations on arrays
            if (p1.size >= 2 and p2.size >= 2):
                # Check if positions are valid (not at origin)
                p1_valid = not (abs(p1[0]) < 1e-6 and abs(p1[1]) < 1e-6)
                p2_valid = not (abs(p2[0]) < 1e-6 and abs(p2[1]) < 1e-6)
                
                if p1_valid and p2_valid:
                    pos1 = p1[:2]
                    pos2 = p2[:2]
                    return np.linalg.norm(pos1 - pos2)
        
        # Fallback: use distance and bearing if available
        bearing1 = self._calculate_bearing(obs1)
        bearing2 = self._calculate_bearing(obs2)
        
        dist1 = obs1.distance if obs1.distance > 0 else getattr(obs1, 'fusion_distance', 0)
        dist2 = obs2.distance if obs2.distance > 0 else getattr(obs2, 'fusion_distance', 0)
        
        if dist1 > 0 and dist2 > 0:
            # Convert to Cartesian and calculate distance
            x1 = dist1 * math.cos(bearing1)
            y1 = dist1 * math.sin(bearing1)
            x2 = dist2 * math.cos(bearing2)
            y2 = dist2 * math.sin(bearing2)
            
            return math.hypot(x2 - x1, y2 - y1)
        
        return float('inf')
    
    def _calculate_bearing_difference(self, obs1: Obstacle, obs2: Obstacle) -> float:
        """Calculate bearing difference between two obstacles"""
        bearing1 = self._calculate_bearing(obs1)
        bearing2 = self._calculate_bearing(obs2)
        return angle_difference(bearing1, bearing2)
    
    def _calculate_bearing(self, obs: Obstacle) -> float:
        """FIXED: Calculate bearing angle of obstacle from ego vehicle"""
        p = getattr(obs, 'position_3d', None)
        if p is not None:
            p = np.asarray(p, dtype=float).ravel()
            # FIXED: Check array elements individually
            if p.size >= 2:
                # Check if position is valid (not at origin)
                if not (abs(p[0]) < 1e-6 and abs(p[1]) < 1e-6):
                    x, y = p[0], p[1]
                    return math.atan2(y, -abs(x))  # Bearing relative to rear direction
        
        # Fallback: estimate from bounding box if available
        if hasattr(obs, 'bbox') and obs.bbox is not None and len(obs.bbox) >= 4 and obs.bbox[0] >= 0:
            bbox_center_x = (obs.bbox[0] + obs.bbox[2]) / 2
            # Rough bearing estimation (would need camera parameters for accuracy)
            image_width = 800  # Assumed image width
            focal_length = 800  # Assumed focal length
            return math.atan((bbox_center_x - image_width/2) / focal_length)
        
        return 0.0
    
    def _update_tracks_with_associations(self, associations: List[Tuple[int, int, float]],
                                       radar_detections: List[Obstacle],
                                       camera_detections: List[Obstacle]) -> None:
        """Update tracks with associated detections"""
        for radar_idx, camera_idx, _ in associations:
            radar_obs = radar_detections[radar_idx]
            camera_obs = camera_detections[camera_idx]
            
            # Create measurements
            radar_pos = np.array(radar_obs.position_3d[:2])
            radar_vel = None
            if abs(radar_obs.relative_speed) > 0.1:  # Has velocity measurement
                # Convert relative speed to velocity components (simplified)
                bearing = self._calculate_bearing(radar_obs)
                radar_vel = np.array([
                    radar_obs.relative_speed * math.cos(bearing),
                    radar_obs.relative_speed * math.sin(bearing)
                ])
            
            # Find best matching track
            track_associations = self.tracker.associate([radar_pos])
            
            if track_associations:
                # Update existing track
                track_id, _, _ = track_associations[0]
                self.tracker.update_radar(track_id, radar_pos, radar_vel)
                
                # Update with camera if position available
                # FIXED: Proper array checking for camera position
                cam_pos_3d = getattr(camera_obs, 'position_3d', None)
                if cam_pos_3d is not None and len(cam_pos_3d) >= 2:
                    # Check if camera position is valid
                    if not (abs(cam_pos_3d[0]) < 1e-6 and abs(cam_pos_3d[1]) < 1e-6):
                        camera_pos = np.array(cam_pos_3d[:2])
                        self.tracker.update_camera(track_id, camera_pos)
                
                # Store detection references
                track = self.tracker.tracks[track_id]
                track.last_detection = self._merge_detections(radar_obs, camera_obs)
                
            else:
                # Create new track from radar
                track_id = self.tracker.create_track_from_radar(radar_pos, radar_vel)
                track = self.tracker.tracks[track_id]
                track.last_detection = self._merge_detections(radar_obs, camera_obs)
    
    def _handle_unmatched_detections(self, associations: List[Tuple[int, int, float]],
                                   radar_detections: List[Obstacle],
                                   camera_detections: List[Obstacle]) -> None:
        """Handle unmatched detections"""
        # Get matched indices
        matched_radar = {assoc[0] for assoc in associations}
        matched_camera = {assoc[1] for assoc in associations}
        
        # Handle unmatched radar detections
        for radar_idx, radar_obs in enumerate(radar_detections):
            if radar_idx in matched_radar:
                continue
            
            radar_pos = np.array(radar_obs.position_3d[:2])
            radar_vel = None
            if abs(radar_obs.relative_speed) > 0.1:
                bearing = self._calculate_bearing(radar_obs)
                radar_vel = np.array([
                    radar_obs.relative_speed * math.cos(bearing),
                    radar_obs.relative_speed * math.sin(bearing)
                ])
            
            # Try to associate with existing tracks
            track_associations = self.tracker.associate([radar_pos])
            
            if track_associations:
                track_id, _, _ = track_associations[0]
                self.tracker.update_radar(track_id, radar_pos, radar_vel)
            else:
                # Create new radar-only track
                track_id = self.tracker.create_track_from_radar(radar_pos, radar_vel)
                track = self.tracker.tracks[track_id]
                track.last_detection = radar_obs
        
        # Handle unmatched camera detections
        for camera_idx, camera_obs in enumerate(camera_detections):
            if camera_idx in matched_camera:
                continue
            
            # FIXED: Only create tracks for camera detections with valid positions
            cam_pos_3d = getattr(camera_obs, 'position_3d', None)
            if cam_pos_3d is None or len(cam_pos_3d) < 2:
                continue
                
            # Check if camera position is valid (not at origin)
            if abs(cam_pos_3d[0]) < 1e-6 and abs(cam_pos_3d[1]) < 1e-6:
                continue
            
            camera_pos = np.array(cam_pos_3d[:2])
            
            # Try to associate with existing tracks
            track_associations = self.tracker.associate([camera_pos])
            
            if track_associations:
                track_id, _, _ = track_associations[0]
                self.tracker.update_camera(track_id, camera_pos)
            else:
                # Create new camera-only track
                track_id = self.tracker.create_track_from_camera(camera_pos)
                track = self.tracker.tracks[track_id]
                track.last_detection = camera_obs
    
    def _merge_detections(self, radar_obs: Obstacle, camera_obs: Obstacle) -> Obstacle:
        """FIXED: Merge radar and camera detections into single obstacle"""
        merged = Obstacle()
        
        # FIXED: Position: prefer radar (more accurate ranging) - proper array checking
        radar_pos_valid = False
        radar_pos_3d = getattr(radar_obs, 'position_3d', None)
        if radar_pos_3d is not None and len(radar_pos_3d) >= 2:
            # Check if radar position is valid (not at origin)
            radar_pos_valid = not (abs(radar_pos_3d[0]) < 1e-6 and abs(radar_pos_3d[1]) < 1e-6)
        
        if radar_pos_valid:
            merged.position_3d = list(radar_obs.position_3d[:])
            merged.distance = radar_obs.distance
        else:
            merged.position_3d = list(camera_obs.position_3d[:])
            merged.distance = camera_obs.distance
        
        # Velocity: prefer radar (Doppler measurement)
        merged.speed = radar_obs.speed if abs(radar_obs.speed) > 0.1 else camera_obs.speed
        merged.relative_speed = radar_obs.relative_speed if abs(radar_obs.relative_speed) > 0.1 else camera_obs.relative_speed
        
        # Classification: prefer camera
        merged.class_id = camera_obs.class_id if camera_obs.class_id > 0 else radar_obs.class_id
        
        # Bounding box: use camera
        merged.bbox = camera_obs.bbox if hasattr(camera_obs, 'bbox') else [-1, -1, -1, -1]
        
        # Confidence: weighted combination
        merged.confidence = weighted_average(
            [radar_obs.confidence, camera_obs.confidence],
            [self.radar_weight, self.camera_weight]
        )
        
        # Sensor information
        merged.sensor_type = "fused_rear"
        merged.sensor_sources = list(set(
            getattr(radar_obs, 'sensor_sources', ['radar_rear']) + 
            getattr(camera_obs, 'sensor_sources', ['camera_rear'])
        ))
        
        # Tracking info: use best available
        merged.track_id = camera_obs.track_id if camera_obs.track_id > 0 else radar_obs.track_id
        merged.track_age = max(
            getattr(radar_obs, 'track_age', 0),
            getattr(camera_obs, 'track_age', 0)
        )
        
        # Point count from radar
        merged.point_count = getattr(radar_obs, 'point_count', 0)
        
        # Size information
        if hasattr(camera_obs, 'size_3d') and (camera_obs.size_3d is not None) and (len(camera_obs.size_3d) >= 3):
            merged.size_3d = camera_obs.size_3d[:]
        elif hasattr(radar_obs, 'size_3d') and (radar_obs.size_3d is not None) and (len(radar_obs.size_3d) >= 3):
            merged.size_3d = radar_obs.size_3d[:]
        else:
            merged.size_3d = [0.0, 0.0, 0.0]
        
        merged.fusion_distance = merged.distance
        
        return merged
    
    def _create_output_obstacles(self, radar_obs: ObstacleArray, 
                               camera_obs: ObstacleArray) -> ObstacleArray:
        """Create output obstacle array from confirmed tracks"""
        result = ObstacleArray()
        
        # Set header from most recent input
        if (radar_obs is not None) and hasattr(radar_obs, 'obstacles') and len(radar_obs.obstacles) > 0:
            result.header = radar_obs.header
        elif (camera_obs is not None) and hasattr(camera_obs, 'obstacles') and len(camera_obs.obstacles) > 0:
            result.header = camera_obs.header
        else:
            result.header.frame_id = "base_link"
        
        result.header.frame_id = "base_link"
        
        # Convert confirmed tracks to obstacles
        confirmed_tracks = self.tracker.get_confirmed_tracks()
        
        for track in confirmed_tracks:
            obstacle = Obstacle()
            
            # Position and motion from track state
            px, py = track.position[0], track.position[1]
            if len(track.position) >= 4:
                vx, vy = track.position[2], track.position[3]
                speed = np.hypot(vx, vy)
            else:
                speed = 0.0
            
            obstacle.position_3d = [float(px), float(py), 0.0]
            obstacle.distance = float(np.hypot(px, py))
            obstacle.speed = float(speed)
            obstacle.relative_speed = 0.0  # To be set by caller
            
            # Track information
            obstacle.track_id = track.id
            obstacle.track_age = max(0, track.hits - 1)
            obstacle.sensor_sources = track.sensor_sources
            
            # Detection properties from last detection
            if track.last_detection:
                last_det = track.last_detection
                obstacle.class_id = getattr(last_det, 'class_id', -1)
                obstacle.confidence = getattr(last_det, 'confidence', 0.6)
                obstacle.bbox = validate_bbox(getattr(last_det, 'bbox', None))
                obstacle.point_count = getattr(last_det, 'point_count', 0)
                
                size_attr = getattr(last_det, 'size_3d', None)
                if size_attr is not None:
                    size_arr = np.asarray(size_attr, dtype=float).reshape(-1)
                    if size_arr.size >= 3 and np.isfinite(size_arr[:3]).all():
                        obstacle.size_3d = size_arr[:3].tolist()
                    else:
                        obstacle.size_3d = [0.0, 0.0, 0.0]
                else:
                    obstacle.size_3d = [0.0, 0.0, 0.0]
            else:
                obstacle.class_id = -1
                obstacle.confidence = 0.6
                obstacle.bbox = [-1, -1, -1, -1]
                obstacle.point_count = 0
                obstacle.size_3d = [0.0, 0.0, 0.0]
            
            # Determine sensor type based on sources
            if 'radar_rear' in obstacle.sensor_sources and 'camera_rear' in obstacle.sensor_sources:
                obstacle.sensor_type = "fused_rear"
            elif 'radar_rear' in obstacle.sensor_sources:
                obstacle.sensor_type = "radar_rear"
            else:
                obstacle.sensor_type = "camera_rear"
            
            obstacle.fusion_distance = obstacle.distance
            
            result.obstacles.append(obstacle)
        
        return result
    
    def fuse_sensors(self, sensor_data: Dict) -> ObstacleArray:
        """Implement BaseFusion interface"""
        radar_data = sensor_data.get('radar', ObstacleArray())
        camera_data = sensor_data.get('camera', ObstacleArray())
        return self.process(radar_data, camera_data)


# ============================================================================
# GLOBAL FUSION INSTANCE (for backward compatibility)
# ============================================================================

_global_rear_fusion: Optional[RearSensorFusion] = None

def fuse_rear(radar_rear: ObstacleArray,
              cam_rear: ObstacleArray,
              params: Dict = None) -> ObstacleArray:
    """
    Convenience function for rear sensor fusion (maintains global state)
    
    Args:
        radar_rear: Radar obstacle array
        cam_rear: Camera obstacle array  
        params: Fusion parameters
        
    Returns:
        Fused obstacle array
    """
    global _global_rear_fusion
    
    if params is None:
        params = {}
    
    # Create or update fusion instance
    if _global_rear_fusion is None:
        _global_rear_fusion = RearSensorFusion(params)
    else:
        # Update parameters if they changed
        if params != _global_rear_fusion.fusion_params:
            _global_rear_fusion.update_params(params)
    
    return _global_rear_fusion.process(radar_rear, cam_rear)


# ============================================================================
# PARAMETER DEFAULTS
# ============================================================================

DEFAULT_REAR_FUSION_PARAMS = {
    # Association parameters
    'max_position_distance': 3.0,
    'max_bearing_diff': 0.3,
    'min_bbox_overlap': 0.1,
    
    # Fusion weights
    'fusion_weights': {
        'camera': 0.7,
        'radar': 0.3
    },
    
    # Validation parameters
    'min_confidence': 0.2,
    'max_distance': 100.0,
    'max_distance_error': 0.5,
    
    # Tracking parameters
    'tracking': {
        # Kalman filter parameters
        'q_c': 1.5,
        'sigma_radar_x': 1.0,
        'sigma_radar_y': 1.0, 
        'sigma_camera_x': 2.0,
        'sigma_camera_y': 2.0,
        'sigma_vel_x': 2.0,
        'sigma_vel_y': 2.0,
        
        # Track management
        'min_hits': 2,
        'max_age': 1.0,
        'max_misses': 3,
        'gate_threshold': 9.21
    }
}