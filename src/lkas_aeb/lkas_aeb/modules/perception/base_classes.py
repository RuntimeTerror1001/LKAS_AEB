#!/usr/bin/env python3
"""
Base Classes for Perception Modules

Provides consistent interfaces and common functionality for all perception modules.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time
import numpy as np
from lkas_aeb_msgs.msg import ObstacleArray, Obstacle


# ============================================================================
# COMMON DATA STRUCTURES
# ============================================================================

@dataclass
class TrackState:
    """Common track state representation"""
    id: int
    position: np.ndarray  # [x, y] or [x, y, vx, vy]
    covariance: np.ndarray
    last_update: float
    hits: int = 0
    misses: int = 0
    age: float = 0.0
    last_detection: Optional[Obstacle] = None
    sensor_sources: List[str] = None
    
    def __post_init__(self):
        if self.sensor_sources is None:
            self.sensor_sources = []

@dataclass 
class ProcessingStats:
    """Processing statistics"""
    processing_time: float = 0.0
    input_count: int = 0
    output_count: int = 0
    errors: int = 0
    last_update: float = 0.0


# ============================================================================
# BASE INTERFACES
# ============================================================================

class BasePerceptionModule(ABC):
    """Base class for all perception modules"""
    
    def __init__(self, params: Dict, name: str = "Unknown"):
        self.params = params.copy()
        self.name = name
        self.stats = ProcessingStats()
        self._last_update_time = time.time()
        self._is_initialized = False
        
        # Initialize the module
        self._initialize()
        self._is_initialized = True
    
    @abstractmethod
    def _initialize(self) -> None:
        """Initialize module-specific components"""
        pass
    
    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """Main processing function - must be implemented by subclasses"""
        pass
    
    def update_params(self, new_params: Dict) -> None:
        """Update parameters and reinitialize if needed"""
        old_params = self.params.copy()
        self.params.update(new_params)
        
        # Check if critical parameters changed
        if self._params_changed(old_params, self.params):
            self._initialize()
    
    def get_stats(self) -> Dict:
        """Get processing statistics"""
        return {
            'name': self.name,
            'processing_time': self.stats.processing_time,
            'input_count': self.stats.input_count,
            'output_count': self.stats.output_count,
            'errors': self.stats.errors,
            'last_update': self.stats.last_update,
            'is_initialized': self._is_initialized
        }
    
    def _params_changed(self, old_params: Dict, new_params: Dict) -> bool:
        """Check if critical parameters changed - override in subclasses"""
        return False
    
    def _update_stats(self, start_time: float, input_count: int = 0, 
                     output_count: int = 0, had_error: bool = False) -> None:
        """Update processing statistics"""
        self.stats.processing_time = time.time() - start_time
        self.stats.input_count = input_count
        self.stats.output_count = output_count
        if had_error:
            self.stats.errors += 1
        self.stats.last_update = time.time()


class BaseTracker(BasePerceptionModule):
    """Base class for tracking modules"""
    
    def __init__(self, params: Dict, name: str = "BaseTracker"):
        self.tracks: Dict[int, TrackState] = {}
        self.next_track_id = 0
        super().__init__(params, name)
    
    def _initialize(self) -> None:
        """Initialize tracker parameters"""
        self.max_age = self.params.get('max_age', 1.0)
        self.min_hits = self.params.get('min_hits', 2)
        self.association_threshold = self.params.get('association_threshold', 5.0)
    
    @abstractmethod
    def predict(self, dt: float) -> None:
        """Predict all tracks forward in time"""
        pass
    
    @abstractmethod
    def update(self, detections: List[Any]) -> None:
        """Update tracks with new detections"""
        pass
    
    @abstractmethod
    def associate(self, detections: List[Any]) -> List[Tuple[int, int, float]]:
        """Associate detections to tracks"""
        pass
    
    def prune_tracks(self) -> None:
        """Remove stale tracks"""
        current_time = time.time()
        to_remove = []
        
        for track_id, track in self.tracks.items():
            age = current_time - track.last_update
            if age > self.max_age or track.misses > self.params.get('max_misses', 5):
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracks[track_id]
    
    def get_confirmed_tracks(self) -> List[TrackState]:
        """Get tracks that meet confirmation criteria"""
        return [track for track in self.tracks.values() 
                if track.hits >= self.min_hits]
    
    def _create_new_track(self, detection: Any, initial_state: np.ndarray,
                         initial_covariance: np.ndarray) -> TrackState:
        """Create a new track"""
        track = TrackState(
            id=self.next_track_id,
            position=initial_state,
            covariance=initial_covariance,
            last_update=time.time(),
            hits=1,
            last_detection=detection
        )
        self.tracks[self.next_track_id] = track
        self.next_track_id += 1
        return track


class BaseKalmanTracker(BaseTracker):
    """Base Kalman filter tracker"""
    
    def __init__(self, params: Dict, name: str = "BaseKalmanTracker"):
        super().__init__(params, name)
        self.state_dim = params.get('state_dim', 4)  # [x, y, vx, vy]
        self.measurement_dim = params.get('measurement_dim', 2)  # [x, y]
    
    def _initialize(self) -> None:
        """Initialize Kalman filter parameters"""
        super()._initialize()
        
        # Process noise
        self.q_c = self.params.get('q_c', 2.0)
        
        # Measurement noise
        self.R = np.diag([
            self.params.get('sigma_x', 1.0)**2,
            self.params.get('sigma_y', 1.0)**2
        ]).astype(float)
        
        # Measurement model (position only)
        self.H = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0]], dtype=float)
        
        # Gate threshold for association
        self.gate_threshold = self.params.get('gate_threshold', 9.21)  # Chi-square 95% for 2DOF
    
    def predict(self, dt: float) -> None:
        """Predict all tracks using Kalman filter"""
        dt = max(1e-3, float(dt))
        
        # State transition matrix
        F = np.array([[1, 0, dt, 0],
                     [0, 1, 0, dt],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]], dtype=float)
        
        # Process noise covariance
        Q = self.q_c * np.array([
            [dt**4/4, 0, dt**3/2, 0],
            [0, dt**4/4, 0, dt**3/2],
            [dt**3/2, 0, dt**2, 0],
            [0, dt**3/2, 0, dt**2]
        ], dtype=float)
        
        current_time = time.time()
        for track in self.tracks.values():
            # Predict state
            track.position = F @ track.position
            track.covariance = F @ track.covariance @ F.T + Q
            track.age += dt
            track.misses += 1
    
    def kalman_update(self, track: TrackState, measurement: np.ndarray) -> None:
        """Update track with Kalman filter"""
        z = np.array(measurement, dtype=float).reshape(-1, 1)
        x = track.position.reshape(-1, 1)
        P = track.covariance
        
        # Innovation
        y = z - self.H @ x
        
        # Innovation covariance
        S = self.H @ P @ self.H.T + self.R
        
        try:
            # Kalman gain
            K = P @ self.H.T @ np.linalg.inv(S)
            
            # Update state
            track.position = (x + K @ y).flatten()
            
            # Update covariance
            I = np.eye(len(x))
            track.covariance = (I - K @ self.H) @ P
            
            # Update track info
            track.hits += 1
            track.misses = 0
            track.last_update = time.time()
            
        except np.linalg.LinAlgError:
            # Skip update if matrix inversion fails
            pass
    
    def mahalanobis_distance(self, track: TrackState, measurement: np.ndarray) -> float:
        """Calculate Mahalanobis distance for gating"""
        try:
            z = np.array(measurement, dtype=float).reshape(-1, 1)
            x_pred = self.H @ track.position.reshape(-1, 1)
            
            y = z - x_pred
            S = self.H @ track.covariance @ self.H.T + self.R
            
            S_inv = np.linalg.inv(S)
            distance_sq = float((y.T @ S_inv @ y).item())
            
            return distance_sq
            
        except (np.linalg.LinAlgError, ValueError):
            return float('inf')
    
    def associate(self, measurements: List[np.ndarray]) -> List[Tuple[int, int, float]]:
        """Associate measurements to tracks using Mahalanobis distance"""
        associations = []
        
        for meas_idx, measurement in enumerate(measurements):
            for track_id, track in self.tracks.items():
                distance = self.mahalanobis_distance(track, measurement)
                if distance <= self.gate_threshold:
                    associations.append((track_id, meas_idx, distance))
        
        # Sort by distance and resolve conflicts greedily
        associations.sort(key=lambda x: x[2])
        
        used_tracks = set()
        used_measurements = set()
        final_associations = []
        
        for track_id, meas_idx, distance in associations:
            if track_id not in used_tracks and meas_idx not in used_measurements:
                final_associations.append((track_id, meas_idx, distance))
                used_tracks.add(track_id)
                used_measurements.add(meas_idx)
        
        return final_associations


class BaseFusion(BasePerceptionModule):
    """Base class for sensor fusion modules"""
    
    def __init__(self, params: Dict, name: str = "BaseFusion"):
        super().__init__(params, name)
    
    @abstractmethod
    def fuse_sensors(self, sensor_data: Dict) -> ObstacleArray:
        """Fuse data from multiple sensors"""
        pass
    
    @abstractmethod
    def associate_detections(self, sensor1_data: List, sensor2_data: List) -> List[Tuple]:
        """Associate detections between sensors"""
        pass
    
    def create_fused_obstacle(self, detections: List[Obstacle], 
                            fusion_weights: Dict[str, float] = None) -> Obstacle:
        """Create fused obstacle from multiple detections"""
        if not detections:
            return Obstacle()
        
        if fusion_weights is None:
            fusion_weights = {'camera': 0.5, 'lidar': 0.5, 'radar': 0.3}
        
        fused = Obstacle()
        
        # Initialize with first detection
        primary = detections[0]
        fused.class_id = primary.class_id
        fused.track_id = primary.track_id
        fused.sensor_sources = []
        
        # Weighted fusion of numeric properties
        total_weight = 0.0
        weighted_distance = 0.0
        weighted_confidence = 0.0
        weighted_position = np.zeros(3)
        
        for det in detections:
            # Determine sensor weight
            sensor_type = getattr(det, 'sensor_type', 'unknown')
            weight = fusion_weights.get(sensor_type.split('_')[0], 0.3)
            
            # Accumulate weighted values
            if det.distance > 0:
                weighted_distance += det.distance * weight
                total_weight += weight
            
            weighted_confidence += det.confidence * weight
            
            if det.position_3d and len(det.position_3d) >= 3:
                pos_array = np.array(det.position_3d[:3])
                weighted_position += pos_array * weight
            
            # Collect sensor sources
            if hasattr(det, 'sensor_sources') and det.sensor_sources:
                fused.sensor_sources.extend(det.sensor_sources)
            elif sensor_type:
                fused.sensor_sources.append(sensor_type)
            
            # Use best available values for discrete properties
            if det.class_id > 0 and primary.class_id <= 0:
                fused.class_id = det.class_id
            
            if hasattr(det, 'bbox') and det.bbox and det.bbox[0] >= 0:
                fused.bbox = det.bbox
            
            if det.track_id > 0:
                fused.track_id = det.track_id
        
        # Finalize weighted averages
        if total_weight > 0:
            fused.distance = float(weighted_distance / total_weight)
            fused.position_3d = (weighted_position / total_weight).tolist()
        else:
            fused.distance = primary.distance
            fused.position_3d = primary.position_3d
        
        fused.confidence = min(1.0, weighted_confidence / len(detections))
        fused.sensor_sources = list(set(fused.sensor_sources))
        fused.sensor_type = "fused"
        fused.fusion_distance = fused.distance
        
        # Calculate speed from position if tracking is available
        fused.speed = primary.speed
        fused.relative_speed = primary.relative_speed
        
        return fused


class BasePreprocessor(BasePerceptionModule):
    """Base class for sensor preprocessing modules"""
    
    @abstractmethod
    def preprocess(self, raw_data: Any) -> Any:
        """Preprocess raw sensor data"""
        pass
    
    def apply_roi_filter(self, data: Any, roi_params: Dict) -> Any:
        """Apply region of interest filter - to be implemented by subclasses"""
        return data
    
    def apply_noise_filter(self, data: Any) -> Any:
        """Apply noise filtering - to be implemented by subclasses"""
        return data


# ============================================================================
# SPECIALIZED BASE CLASSES
# ============================================================================

class BasePointCloudProcessor(BasePreprocessor):
    """Base class for point cloud processing"""
    
    def __init__(self, params: Dict, name: str = "PointCloudProcessor"):
        super().__init__(params, name)
    
    def _initialize(self) -> None:
        """Initialize point cloud processing parameters"""
        self.voxel_size = self.params.get('voxel_size', 0.1)
        self.min_points_per_voxel = self.params.get('min_points_per_voxel', 1)
        self.cluster_eps = self.params.get('cluster_eps', 0.5)
        self.cluster_min_samples = self.params.get('cluster_min_samples', 3)
    
    @abstractmethod
    def extract_points(self, msg: Any) -> np.ndarray:
        """Extract point array from message"""
        pass
    
    @abstractmethod
    def cluster_points(self, points: np.ndarray) -> List[np.ndarray]:
        """Cluster points into objects"""
        pass
    
    def points_to_obstacles(self, clustered_points: List[np.ndarray], 
                          header: Any) -> ObstacleArray:
        """Convert clustered points to obstacles"""
        obstacle_array = ObstacleArray()
        obstacle_array.header = header
        
        for i, cluster in enumerate(clustered_points):
            if len(cluster) == 0:
                continue
                
            obstacle = Obstacle()
            
            # Calculate centroid and bounds
            centroid = cluster.mean(axis=0)
            bounds_min = cluster.min(axis=0)
            bounds_max = cluster.max(axis=0)
            size = bounds_max - bounds_min
            
            # Fill obstacle properties
            obstacle.class_id = -1  # Unknown class
            obstacle.distance = float(np.linalg.norm(centroid[:2]))
            obstacle.position_3d = centroid.tolist()
            obstacle.size_3d = size.tolist()
            obstacle.point_count = len(cluster)
            obstacle.confidence = min(1.0, len(cluster) / 50.0)  # More points = higher confidence
            obstacle.track_id = i  # Temporary ID
            obstacle.bbox = [-1, -1, -1, -1]  # No image bbox
            
            obstacle_array.obstacles.append(obstacle)
        
        return obstacle_array


class BaseCameraProcessor(BasePreprocessor):
    """Base class for camera processing"""
    
    def __init__(self, params: Dict, name: str = "CameraProcessor"):
        super().__init__(params, name)
    
    def _initialize(self) -> None:
        """Initialize camera processing parameters"""
        self.min_bbox_area = self.params.get('min_bbox_area', 100)
        self.min_confidence = self.params.get('min_confidence', 0.3)
        
        # Camera intrinsics
        self.fx = self.params.get('fx', 800.0)
        self.fy = self.params.get('fy', 800.0) 
        self.cx = self.params.get('cx', 640.0)
        self.cy = self.params.get('cy', 360.0)
    
    @abstractmethod
    def detect_objects(self, image: np.ndarray) -> List[Tuple]:
        """Detect objects in image"""
        pass
    
    def image_to_obstacles(self, detections: List[Tuple], header: Any) -> ObstacleArray:
        """Convert image detections to obstacles"""
        obstacle_array = ObstacleArray()
        obstacle_array.header = header
        
        for detection in detections:
            # Expected format: (x1, y1, x2, y2, distance, speed, class_id, track_id)
            if len(detection) < 4:
                continue
                
            obstacle = Obstacle()
            
            # Bounding box
            obstacle.bbox = [int(x) for x in detection[:4]]
            
            # Optional fields
            if len(detection) > 4:
                obstacle.distance = float(detection[4]) if detection[4] > 0 else -1.0
            if len(detection) > 5:
                obstacle.speed = float(detection[5])
            if len(detection) > 6:
                obstacle.class_id = int(detection[6])
            if len(detection) > 7:
                obstacle.track_id = int(detection[7])
            
            # Calculate position estimate if distance available
            if obstacle.distance > 0:
                cx, cy = (obstacle.bbox[0] + obstacle.bbox[2]) / 2, (obstacle.bbox[1] + obstacle.bbox[3]) / 2
                # Simple pinhole projection
                x = obstacle.distance * (cx - self.cx) / self.fx
                y = obstacle.distance * (cy - self.cy) / self.fy
                obstacle.position_3d = [obstacle.distance, -x, -y]  # Camera frame
            
            obstacle.sensor_type = "camera"
            obstacle.confidence = 0.7  # Default camera confidence
            
            obstacle_array.obstacles.append(obstacle)
        
        return obstacle_array


# ============================================================================
# PARAMETER VALIDATION
# ============================================================================

class ParameterValidator:
    """Validates and sanitizes parameters for perception modules"""
    
    @staticmethod
    def validate_base_params(params: Dict) -> Dict:
        """Validate common base parameters"""
        validated = {}
        
        # Timing parameters
        validated['max_age'] = max(0.1, params.get('max_age', 1.0))
        validated['min_hits'] = max(1, params.get('min_hits', 2))
        validated['max_misses'] = max(1, params.get('max_misses', 5))
        
        # Association parameters
        validated['association_threshold'] = max(0.1, params.get('association_threshold', 5.0))
        validated['gate_threshold'] = max(0.1, params.get('gate_threshold', 9.21))
        
        # Confidence parameters
        validated['min_confidence'] = max(0.0, min(1.0, params.get('min_confidence', 0.3)))
        
        return validated
    
    @staticmethod
    def validate_kalman_params(params: Dict) -> Dict:
        """Validate Kalman filter parameters"""
        validated = ParameterValidator.validate_base_params(params)
        
        # Process noise
        validated['q_c'] = max(0.01, params.get('q_c', 2.0))
        
        # Measurement noise
        validated['sigma_x'] = max(0.01, params.get('sigma_x', 1.0))
        validated['sigma_y'] = max(0.01, params.get('sigma_y', 1.0))
        
        # Dimensions
        validated['state_dim'] = max(2, params.get('state_dim', 4))
        validated['measurement_dim'] = max(1, params.get('measurement_dim', 2))
        
        return validated
    
    @staticmethod
    def validate_fusion_params(params: Dict) -> Dict:
        """Validate sensor fusion parameters"""
        validated = ParameterValidator.validate_base_params(params)
        
        # Distance consistency
        validated['max_distance_diff'] = max(0.1, params.get('max_distance_diff', 5.0))
        validated['max_bearing_diff'] = max(0.01, params.get('max_bearing_diff', 0.5))
        
        # Bounding box parameters
        validated['min_bbox_overlap'] = max(0.0, min(1.0, params.get('min_bbox_overlap', 0.1)))
        validated['bbox_padding'] = max(0, params.get('bbox_padding', 5))
        
        return validated