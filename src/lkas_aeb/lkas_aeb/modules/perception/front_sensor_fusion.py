#!/usr/bin/env python3
"""
FIXED: Refactored Front Sensor Fusion Module

Fixed potential array boolean comparison issues similar to the rear fusion module.
Consistent implementation using base classes and common utilities.
Fuses camera and LiDAR detections using Kalman filter tracking.
"""

from typing import Dict, List, Optional, Tuple
import time
import numpy as np
from lkas_aeb_msgs.msg import Obstacle, ObstacleArray

# Import base classes and utilities
from .base_classes import BaseFusion, BaseKalmanTracker, ParameterValidator
from lkas_aeb.util.perception_utils import (
    validate_bbox, validate_position_3d, validate_numeric, validate_integer,
    bbox_contains_point, bbox_center, bbox_area,
    project_to_image, create_intrinsic_matrix,
    homogeneous_transform, ProcessingError
)


class FrontKalmanTracker(BaseKalmanTracker):
    """Kalman filter tracker for front sensor fusion"""
    
    def __init__(self, params: Dict):
        super().__init__(params, "FrontKalmanTracker")
    
    def _initialize(self) -> None:
        """Initialize front-specific Kalman filter parameters"""
        super()._initialize()
        
        # Separate measurement models for LiDAR and camera
        self.R_lidar = np.diag([
            self.params.get('sigma_lidar_x', 0.4)**2,
            self.params.get('sigma_lidar_y', 0.2)**2
        ]).astype(float)
        
        self.R_camera = np.diag([
            self.params.get('sigma_camera_x', 3.0)**2,
            self.params.get('sigma_camera_y', 1.5)**2
        ]).astype(float)
    
    def update_lidar(self, track_id: int, measurement: np.ndarray) -> bool:
        """Update track with LiDAR measurement"""
        if track_id not in self.tracks:
            return False
        
        track = self.tracks[track_id]
        
        # Temporarily use LiDAR measurement noise
        original_R = self.R
        self.R = self.R_lidar
        
        try:
            self.kalman_update(track, measurement)
            track.sensor_sources = list(set(track.sensor_sources + ['lidar_front']))
            return True
        finally:
            self.R = original_R
    
    def update_camera(self, track_id: int, measurement: np.ndarray) -> bool:
        """Update track with camera measurement"""
        if track_id not in self.tracks:
            return False
        
        track = self.tracks[track_id]
        
        # Temporarily use camera measurement noise
        original_R = self.R
        self.R = self.R_camera
        
        try:
            self.kalman_update(track, measurement)
            track.sensor_sources = list(set(track.sensor_sources + ['camera_front']))
            return True
        finally:
            self.R = original_R
    
    def create_track_from_lidar(self, measurement: np.ndarray) -> int:
        """Create new track from LiDAR detection"""
        initial_state = np.array([measurement[0], measurement[1], 0.0, 0.0])
        initial_cov = np.diag([3.0, 3.0, 10.0, 10.0])
        
        track = self._create_new_track(None, initial_state, initial_cov)
        track.sensor_sources = ['lidar_front']
        return track.id
    
    def create_track_from_camera(self, measurement: np.ndarray) -> int:
        """Create new track from camera detection"""
        initial_state = np.array([measurement[0], measurement[1], 0.0, 0.0])
        initial_cov = np.diag([15.0, 15.0, 20.0, 20.0])  # Higher uncertainty for camera-only
        
        track = self._create_new_track(None, initial_state, initial_cov)
        track.sensor_sources = ['camera_front']
        return track.id

    def process(self, detections: List[np.ndarray] = None, dt: float = 0.1) -> List:
        """Generic processing path for tracker-only use."""
        start = time.time()
        try:
            # Predict forward
            try:
                self.predict(float(dt))
            except Exception:
                self.predict(0.1)
            # Update with provided detections if any (assume lidar-like [x,y])
            if detections:
                self.update(detections)
            # Prune old tracks
            self.prune_tracks()
            # Return confirmed track states for convenience
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
        # Guard dtype/shape
        meas_list = []
        for d in detections:
            arr = np.asarray(d, dtype=float).reshape(-1)
            if arr.size >= 2 and np.isfinite(arr[:2]).all():
                meas_list.append(arr[:2])
        if not meas_list:
            return
        associations = self.associate(meas_list)
        # Temporarily switch to lidar noise
        original_R = self.R
        self.R = getattr(self, 'R_lidar', self.R)
        try:
            for track_id, meas_idx, _ in associations:
                m = np.array(meas_list[meas_idx], dtype=float)
                self.kalman_update(self.tracks[track_id], m)
                self.tracks[track_id].sensor_sources = list(set((self.tracks[track_id].sensor_sources or []) + ['lidar_front']))
                self.tracks[track_id].last_update = time.time()
        finally:
            self.R = original_R


class FrontSensorFusion(BaseFusion):
    """Front sensor fusion module combining camera and LiDAR"""
    
    def __init__(self, params: Dict, intrinsics: Dict, transform: np.ndarray):
        """
        Initialize front sensor fusion
        
        Args:
            params: Fusion parameters
            intrinsics: Camera intrinsics dict with fx, fy, cx, cy
            transform: 4x4 transformation matrix from base_link to camera
        """
        # Validate and store parameters
        self.fusion_params = ParameterValidator.validate_fusion_params(params)
        self.tracking_params = ParameterValidator.validate_kalman_params(
            params.get('tracking', {})
        )
        
        super().__init__(self.fusion_params, "FrontSensorFusion")
        
        # Store camera parameters
        self.intrinsics = intrinsics
        self.T_cam_base = np.array(transform).reshape(4, 4)
        self.T_base_cam = np.linalg.inv(self.T_cam_base)
        
        # Create camera matrix - using proper import
        from lkas_aeb.util.perception_utils import create_intrinsic_matrix
        self.camera_matrix = create_intrinsic_matrix(
            self.intrinsics.get('fx', 1100.0),
            self.intrinsics.get('fy', 1100.0),
            self.intrinsics.get('cx', 640.0),
            self.intrinsics.get('cy', 360.0)
        )
    
    def _initialize(self) -> None:
        """Initialize fusion components"""
        # Create Kalman tracker
        self.tracker = FrontKalmanTracker(self.tracking_params)
        
        # Cache frequently used parameters
        self.bbox_padding = self.fusion_params.get('bbox_padding', 6)
        self.min_bbox_area = self.fusion_params.get('min_bbox_area_px', 6000)
        self.distance_gate = self.fusion_params.get('dist_consistency_m', 8.0)
        self.lidar_min_points = self.fusion_params.get('lidar_confirm_min_pts', 10)
    
    def process(self, camera_obs: ObstacleArray, lidar_obs: ObstacleArray) -> ObstacleArray:
        """Main fusion processing function"""
        start_time = time.time()
        
        try:
            # Calculate time step
            current_time = time.time()
            dt = min(0.2, max(1e-3, current_time - getattr(self, '_last_time', current_time)))
            self._last_time = current_time
            
            # Predict all tracks
            self.tracker.predict(dt)
            
            # Clean and validate input data
            camera_detections = self._process_camera_obstacles(camera_obs)
            lidar_detections = self._process_lidar_obstacles(lidar_obs)
            
            # Associate detections between sensors
            associations = self.associate_detections(camera_detections, lidar_detections)
            
            # Update tracks with associated detections
            self._update_tracks_with_associations(associations, camera_detections, lidar_detections)
            
            # Handle unmatched detections
            self._handle_unmatched_detections(associations, camera_detections, lidar_detections)
            
            # Prune old tracks
            self.tracker.prune_tracks()
            
            # Generate output
            result = self._create_output_obstacles(camera_obs, lidar_obs)
            
            # Update statistics
            self._update_stats(
                start_time,
                len(camera_detections) + len(lidar_detections),
                len(result.obstacles)
            )
            
            return result
            
        except Exception as e:
            self._update_stats(start_time, 0, 0, had_error=True)
            raise ProcessingError(f"Front fusion failed: {str(e)}")
    
    def _process_camera_obstacles(self, camera_obs: ObstacleArray) -> List[Obstacle]:
        """Process and validate camera obstacles"""
        if camera_obs is None or not hasattr(camera_obs, 'obstacles') or len(camera_obs.obstacles) == 0:
            return []
        
        processed = []
        for obs in camera_obs.obstacles:
            # Validate bounding box
            bbox = validate_bbox(getattr(obs, 'bbox', None))
            # Use proper utility function for bbox area calculation
            from lkas_aeb.util.perception_utils import bbox_area
            if bbox[0] < 0 or bbox_area(bbox) < self.min_bbox_area:
                continue
            
            # Update obstacle with validated data
            obs.bbox = bbox
            obs.distance = validate_numeric(getattr(obs, 'distance', -1.0), -1.0, min_val=0.0)
            obs.confidence = validate_numeric(getattr(obs, 'confidence', 0.5), 0.5, 0.0, 1.0)
            
            processed.append(obs)
        
        return processed
    
    def _process_lidar_obstacles(self, lidar_obs: ObstacleArray) -> List[Obstacle]:
        """FIXED: Process and validate LiDAR obstacles"""
        if lidar_obs is None or not hasattr(lidar_obs, 'obstacles') or len(lidar_obs.obstacles) == 0:
            return []
        
        processed = []
        for obs in lidar_obs.obstacles:
            # Validate position
            position = validate_position_3d(getattr(obs, 'position_3d', None))
            
            # FIXED: Check array elements individually instead of boolean operations on arrays
            if len(position) >= 2:
                # Check if position is valid (not at origin)
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
            from lkas_aeb.util.perception_utils import validate_integer
            obs.point_count = validate_integer(getattr(obs, 'point_count', 0), 0, min_val=0)
            
            processed.append(obs)
        
        return processed
    
    def associate_detections(self, camera_detections: List[Obstacle], 
                           lidar_detections: List[Obstacle]) -> List[Tuple[int, int, float]]:
        """Associate camera and LiDAR detections"""
        # FIXED: Avoid None checks on potentially empty lists
        if not camera_detections or not lidar_detections:
            return []
        
        # Pre-compute camera bounding box centers and grown boxes
        camera_info = []
        from lkas_aeb.util.perception_utils import bbox_center
        for cam_obs in camera_detections:
            bbox = cam_obs.bbox
            center = bbox_center(bbox)
            grown_bbox = [bbox[0] - self.bbox_padding, bbox[1] - self.bbox_padding,
                         bbox[2] + self.bbox_padding, bbox[3] + self.bbox_padding]
            cam_distance = cam_obs.distance
            camera_info.append((grown_bbox, center, cam_distance))
        
        # Find potential associations
        candidates = []
        for lidar_idx, lidar_obs in enumerate(lidar_detections):
            # Project LiDAR point to image
            lidar_pos = np.array(lidar_obs.position_3d[:3])
            from lkas_aeb.util.perception_utils import project_to_image
            projected = project_to_image(lidar_pos.reshape(1, -1), 
                                       self.camera_matrix, 
                                       self.T_cam_base)
            
            if np.isnan(projected[0]).any():
                continue
            
            u, v = projected[0]
            
            # Check association with each camera detection
            from lkas_aeb.util.perception_utils import bbox_contains_point
            for cam_idx, (grown_bbox, cam_center, cam_distance) in enumerate(camera_info):
                # Check if LiDAR projection falls within grown bounding box
                if not bbox_contains_point(grown_bbox, u, v):
                    continue
                
                # Check distance consistency if camera has distance estimate
                if cam_distance > 0:
                    distance_diff = abs(lidar_obs.distance - cam_distance)
                    if distance_diff > self.distance_gate:
                        continue
                
                # Calculate association cost
                pixel_distance = np.hypot(u - cam_center[0], v - cam_center[1])
                distance_cost = distance_diff / self.distance_gate if cam_distance > 0 else 0.0
                total_cost = pixel_distance + distance_cost * 100  # Weight distance consistency higher
                
                candidates.append((cam_idx, lidar_idx, total_cost))
        
        # Greedy association (lowest cost first)
        candidates.sort(key=lambda x: x[2])
        
        used_camera = set()
        used_lidar = set()
        final_associations = []
        
        for cam_idx, lidar_idx, cost in candidates:
            if cam_idx not in used_camera and lidar_idx not in used_lidar:
                final_associations.append((cam_idx, lidar_idx, cost))
                used_camera.add(cam_idx)
                used_lidar.add(lidar_idx)
        
        return final_associations
    
    def _update_tracks_with_associations(self, associations: List[Tuple[int, int, float]],
                                       camera_detections: List[Obstacle],
                                       lidar_detections: List[Obstacle]) -> None:
        """Update tracks with associated detections"""
        for cam_idx, lidar_idx, _ in associations:
            camera_obs = camera_detections[cam_idx]
            lidar_obs = lidar_detections[lidar_idx]
            
            # Create measurement from LiDAR position
            lidar_measurement = np.array([lidar_obs.position_3d[0], lidar_obs.position_3d[1]])
            
            # Find best matching track for LiDAR measurement
            track_associations = self.tracker.associate([lidar_measurement])
            
            if track_associations:
                # Update existing track
                track_id, _, _ = track_associations[0]
                self.tracker.update_lidar(track_id, lidar_measurement)
                
                # Update track's detection references
                track = self.tracker.tracks[track_id]
                track.last_detection = lidar_obs
                
                # Optional camera update if camera has position estimate
                camera_pos = self._camera_position_from_bbox(camera_obs)
                if camera_pos is not None:
                    self.tracker.update_camera(track_id, camera_pos)
                
            else:
                # Create new track from LiDAR
                track_id = self.tracker.create_track_from_lidar(lidar_measurement)
                track = self.tracker.tracks[track_id]
                track.last_detection = lidar_obs
    
    def _handle_unmatched_detections(self, associations: List[Tuple[int, int, float]],
                                   camera_detections: List[Obstacle],
                                   lidar_detections: List[Obstacle]) -> None:
        """Handle detections that weren't associated"""
        # Get indices of matched detections
        matched_camera = {assoc[0] for assoc in associations}
        matched_lidar = {assoc[1] for assoc in associations}
        
        # Handle unmatched camera detections
        for cam_idx, camera_obs in enumerate(camera_detections):
            if cam_idx in matched_camera:
                continue
            
            camera_pos = self._camera_position_from_bbox(camera_obs)
            if camera_pos is not None:
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
        
        # Handle unmatched LiDAR detections (if enabled)
        if self.fusion_params.get('publish_lidar_only', False):
            lidar_min_points = self.fusion_params.get('lidar_only_min_pts', 30)
            
            for lidar_idx, lidar_obs in enumerate(lidar_detections):
                if lidar_idx in matched_lidar:
                    continue
                
                if lidar_obs.point_count < lidar_min_points:
                    continue
                
                lidar_measurement = np.array([lidar_obs.position_3d[0], lidar_obs.position_3d[1]])
                track_associations = self.tracker.associate([lidar_measurement])
                
                if track_associations:
                    track_id, _, _ = track_associations[0]
                    self.tracker.update_lidar(track_id, lidar_measurement)
                else:
                    track_id = self.tracker.create_track_from_lidar(lidar_measurement)
                    track = self.tracker.tracks[track_id]
                    track.last_detection = lidar_obs
    
    def _camera_position_from_bbox(self, camera_obs: Obstacle) -> Optional[np.ndarray]:
        """Estimate 3D position from camera bounding box"""
        if camera_obs.distance <= 0:
            return None
        
        # Get bounding box center
        from lkas_aeb.util.perception_utils import bbox_center
        center = bbox_center(camera_obs.bbox)
        
        # Simple pinhole projection model
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
        
        # Small angle approximation
        beta_x = (center[0] - cx) / fx
        beta_y = (center[1] - cy) / fy
        
        # 3D position in camera frame
        camera_pos = np.array([
            camera_obs.distance * beta_x,
            camera_obs.distance * beta_y,
            camera_obs.distance,
            1.0
        ])
        
        # Transform to base_link
        try:
            from lkas_aeb.util.perception_utils import homogeneous_transform
            base_pos = homogeneous_transform(camera_pos, self.T_base_cam)
            return base_pos[:2]  # Return only x,y for tracking
        except Exception:
            return None
    
    def _create_output_obstacles(self, camera_obs: ObstacleArray, 
                               lidar_obs: ObstacleArray) -> ObstacleArray:
        """Create output obstacle array from confirmed tracks"""
        result = ObstacleArray()
        
        # Set header from most recent input
        if (camera_obs is not None) and hasattr(camera_obs, 'obstacles') and len(camera_obs.obstacles) > 0:
            result.header = camera_obs.header
        elif (lidar_obs is not None) and hasattr(lidar_obs, 'obstacles') and len(lidar_obs.obstacles) > 0:
            result.header = lidar_obs.header
        else:
            result.header.frame_id = "base_link"
        
        # Convert confirmed tracks to obstacles
        confirmed_tracks = self.tracker.get_confirmed_tracks()
        
        for track in confirmed_tracks:
            obstacle = Obstacle()
            
            # Position and motion
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
                
                if hasattr(last_det, 'bbox') and (last_det.bbox is not None) and (len(last_det.bbox) >= 4):
                    obstacle.bbox = validate_bbox(last_det.bbox)
                else:
                    obstacle.bbox = [-1, -1, -1, -1]
                
                obstacle.point_count = getattr(last_det, 'point_count', 0)
            else:
                obstacle.class_id = -1
                obstacle.confidence = 0.6
                obstacle.bbox = [-1, -1, -1, -1]
                obstacle.point_count = 0
            
            # Boost confidence if LiDAR confirmed
            if ('lidar_front' in obstacle.sensor_sources and 
                obstacle.point_count >= self.lidar_min_points):
                obstacle.confidence = min(1.0, obstacle.confidence + 0.2)
            
            # Fusion properties
            obstacle.sensor_type = "fused_front"
            obstacle.fusion_distance = obstacle.distance
            
            result.obstacles.append(obstacle)
        
        return result
    
    def fuse_sensors(self, sensor_data: Dict) -> ObstacleArray:
        """Implement BaseFusion interface"""
        camera_data = sensor_data.get('camera', ObstacleArray())
        lidar_data = sensor_data.get('lidar', ObstacleArray())
        return self.process(camera_data, lidar_data)


# ============================================================================
# GLOBAL FUSION INSTANCE (for backward compatibility)
# ============================================================================

_global_fusion_instance: Optional[FrontSensorFusion] = None

def fuse_front(camera: ObstacleArray,
               lidar: ObstacleArray,
               transform: np.ndarray,
               intrinsics: Dict,
               params: Dict = None) -> ObstacleArray:
    """
    Convenience function for front sensor fusion (maintains global state)
    
    Args:
        camera: Camera obstacle array
        lidar: LiDAR obstacle array
        transform: 4x4 transformation matrix (base_link to camera)
        intrinsics: Camera intrinsics dictionary
        params: Fusion parameters
        
    Returns:
        Fused obstacle array
    """
    global _global_fusion_instance
    
    if params is None:
        params = {}
    
    # Check if we need to recreate the fusion instance
    need_new_instance = _global_fusion_instance is None
    
    if not need_new_instance:
        # Check if critical parameters changed
        current_transform = _global_fusion_instance.T_cam_base
        current_intrinsics = _global_fusion_instance.intrinsics
        
        if not np.allclose(current_transform, transform, atol=1e-6):
            need_new_instance = True
        
        for key in ['fx', 'fy', 'cx', 'cy']:
            if abs(current_intrinsics.get(key, 0) - intrinsics.get(key, 0)) > 1e-6:
                need_new_instance = True
                break
    
    if need_new_instance:
        _global_fusion_instance = FrontSensorFusion(params, intrinsics, transform)
    else:
        # Update parameters if they changed
        if params != _global_fusion_instance.fusion_params:
            _global_fusion_instance.update_params(params)
    
    return _global_fusion_instance.process(camera, lidar)