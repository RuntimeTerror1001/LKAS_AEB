#!/usr/bin/env python3
"""
Refactored LiDAR Preprocessing Module

Consistent implementation using base classes and common utilities.
Processes PointCloud2 messages into ObstacleArray format.
"""

from typing import Dict, List, Optional
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from lkas_aeb_msgs.msg import Obstacle, ObstacleArray

# Import base classes and utilities
from .base_classes import BasePointCloudProcessor, ParameterValidator
from lkas_aeb.util.perception_utils import (
    roi_filter_3d, voxel_downsample, connected_components_2d,
    validate_numeric, validate_position_3d, ProcessingError, time_function
)


class FrontLidarProcessor(BasePointCloudProcessor):
    """LiDAR preprocessing for front sensor"""
    
    def __init__(self, params: Dict):
        """
        Initialize LiDAR processor
        
        Args:
            params: Processing parameters
        """
        # Validate parameters
        self.processing_params = self._validate_lidar_params(params)
        
        super().__init__(self.processing_params, "FrontLidarProcessor")
    
    def _validate_lidar_params(self, params: Dict) -> Dict:
        """Validate LiDAR-specific parameters"""
        validated = ParameterValidator.validate_base_params(params)
        
        # ROI parameters
        roi_params = params.get('roi', {})
        validated['roi'] = {
            'x_min': validate_numeric(roi_params.get('x_min', 1.0), 1.0),
            'x_max': validate_numeric(roi_params.get('x_max', 120.0), 120.0, min_val=1.0),
            'y_min': validate_numeric(roi_params.get('y_min', -8.0), -8.0),
            'y_max': validate_numeric(roi_params.get('y_max', 8.0), 8.0, min_val=0.1),
            'z_min': validate_numeric(roi_params.get('z_min', -1.5), -1.5),
            'z_max': validate_numeric(roi_params.get('z_max', 3.0), 3.0, min_val=-1.0)
        }
        
        # Ground removal parameters
        ground_params = params.get('ground_removal', {})
        validated['ground_removal'] = {
            'z_max': validate_numeric(ground_params.get('z_max', 0.2), 0.2),
            'ransac_distance': validate_numeric(ground_params.get('ransac_distance', 0.08), 0.08, min_val=0.01),
            'normal_min_z': validate_numeric(ground_params.get('normal_min_z', 0.85), 0.85, min_val=0.1, max_val=1.0)
        }
        
        # Voxel parameters
        voxel_params = params.get('voxel', {})
        validated['voxel'] = {
            'size': validate_numeric(voxel_params.get('size', 0.15), 0.15, min_val=0.01),
            'min_points_per_voxel': max(1, int(voxel_params.get('min_points_per_voxel', 1)))
        }
        
        # Clustering parameters
        clustering_params = params.get('clustering', {})
        validated['clustering'] = {
            'method': clustering_params.get('method', 'connected_components'),
            'min_voxels': max(1, int(clustering_params.get('min_voxels', 3))),
            'max_voxels': max(10, int(clustering_params.get('max_voxels', 50000))),
            'eps': validate_numeric(clustering_params.get('eps', 0.6), 0.6, min_val=0.1),
            'min_samples': max(1, int(clustering_params.get('min_samples', 6)))
        }
        
        return validated
    
    def _initialize(self) -> None:
        """Initialize processor components"""
        super()._initialize()
        
        # Cache commonly used parameters
        self.roi_params = self.processing_params['roi']
        self.ground_params = self.processing_params['ground_removal']
        self.voxel_params = self.processing_params['voxel']
        self.clustering_params = self.processing_params['clustering']
    
    @time_function
    def preprocess(self, msg: PointCloud2) -> ObstacleArray:
        """
        Main preprocessing function
        
        Args:
            msg: Input PointCloud2 message
            
        Returns:
            ObstacleArray with detected obstacles
        """
        try:
            return self._process_pointcloud(msg)
        except Exception as e:
            raise ProcessingError(f"LiDAR preprocessing failed: {str(e)}")
    
    def process(self, msg: PointCloud2) -> ObstacleArray:
        """Implement BasePerceptionModule interface"""
        return self.preprocess(msg)
    
    def _process_pointcloud(self, msg: PointCloud2) -> ObstacleArray:
        """Internal point cloud processing pipeline"""
        start_time = self._update_processing_start()
        
        try:
            # Extract points from message
            points = self.extract_points(msg)
            
            if points.size == 0:
                return self._create_empty_result(msg)
            
            # Apply ROI filter
            points = self.apply_roi_filter(points, self.roi_params)
            if points.size == 0:
                return self._create_empty_result(msg)
            
            # Remove ground points
            points = self._remove_ground_points(points)
            if points.size == 0:
                return self._create_empty_result(msg)
            
            # Voxel downsampling
            voxel_centroids, voxel_indices = voxel_downsample(
                points,
                self.voxel_params['size'],
                self.voxel_params['min_points_per_voxel']
            )
            
            if voxel_centroids.size == 0:
                return self._create_empty_result(msg)
            
            # Cluster voxels
            clusters = self.cluster_points(voxel_centroids, voxel_indices)
            
            # Convert clusters to obstacles
            result = self.points_to_obstacles(
                [voxel_centroids[cluster_indices] for cluster_indices in clusters],
                msg.header
            )
            
            # Update statistics
            self._update_stats(
                start_time,
                input_count=len(points),
                output_count=len(result.obstacles)
            )
            
            return result
            
        except Exception as e:
            self._update_stats(start_time, 0, 0, had_error=True)
            raise e
    
    def extract_points(self, msg: PointCloud2) -> np.ndarray:
        """Extract XYZ points from PointCloud2 message"""
        if msg.width == 0 or msg.height == 0:
            return np.empty((0, 3), dtype=np.float32)
        
        try:
            # Read points using sensor_msgs_py
            points_iter = pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
            
            # Convert to numpy array
            points_list = []
            for point in points_iter:
                if len(point) >= 3:
                    points_list.append([float(point[0]), float(point[1]), float(point[2])])
            
            if not points_list:
                return np.empty((0, 3), dtype=np.float32)
            
            return np.array(points_list, dtype=np.float32)
            
        except Exception as e:
            raise ProcessingError(f"Failed to extract points: {str(e)}")
    
    def apply_roi_filter(self, points: np.ndarray, roi_params: Dict) -> np.ndarray:
        """Apply region of interest filter"""
        if points.size == 0:
            return points
        
        # Use utility function with standardized parameter names
        roi_filter_params = {
            'x_min': roi_params['x_min'],
            'x_max': roi_params['x_max'],
            'y_min': roi_params['y_min'],
            'y_max': roi_params['y_max'],
            'z_min': roi_params['z_min'],
            'z_max': roi_params['z_max']
        }
        
        return roi_filter_3d(points, roi_filter_params)
    
    def _remove_ground_points(self, points: np.ndarray) -> np.ndarray:
        """Remove ground points using simple height threshold"""
        if points.size == 0:
            return points
        
        # Simple ground removal based on Z coordinate
        z_threshold = self.ground_params['z_max']
        mask = points[:, 2] > z_threshold
        
        return points[mask]
    
    def cluster_points(self, voxel_centroids: np.ndarray, voxel_indices: np.ndarray) -> List[np.ndarray]:
        """Cluster voxel centroids into objects"""
        if voxel_centroids.size == 0:
            return []
        
        method = self.clustering_params['method']
        
        if method == 'connected_components':
            return self._cluster_connected_components(voxel_indices)
        elif method == 'dbscan':
            return self._cluster_dbscan(voxel_centroids)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
    
    def _cluster_connected_components(self, voxel_indices: np.ndarray) -> List[np.ndarray]:
        """Cluster using 2D connected components on voxel grid"""
        components = connected_components_2d(voxel_indices, connectivity=8)
        
        # Filter by size constraints
        min_voxels = self.clustering_params['min_voxels']
        max_voxels = self.clustering_params['max_voxels']
        
        filtered_components = []
        for component in components:
            if min_voxels <= len(component) <= max_voxels:
                filtered_components.append(component)
        
        return filtered_components
    
    def _cluster_dbscan(self, points: np.ndarray) -> List[np.ndarray]:
        """Cluster using DBSCAN algorithm (simplified implementation)"""
        if points.size == 0:
            return []
        
        eps = self.clustering_params['eps']
        min_samples = self.clustering_params['min_samples']
        
        # Simplified DBSCAN implementation using only XY coordinates
        points_xy = points[:, :2]
        
        visited = np.zeros(len(points_xy), dtype=bool)
        clusters = []
        
        for i in range(len(points_xy)):
            if visited[i]:
                continue
            
            # Find neighbors
            distances = np.linalg.norm(points_xy - points_xy[i], axis=1)
            neighbors = np.where(distances <= eps)[0]
            
            if len(neighbors) < min_samples:
                visited[i] = True
                continue
            
            # Start new cluster
            cluster = []
            queue = list(neighbors)
            
            while queue:
                current = queue.pop(0)
                if visited[current]:
                    continue
                
                visited[current] = True
                cluster.append(current)
                
                # Find neighbors of current point
                current_distances = np.linalg.norm(points_xy - points_xy[current], axis=1)
                current_neighbors = np.where(current_distances <= eps)[0]
                
                if len(current_neighbors) >= min_samples:
                    for neighbor in current_neighbors:
                        if not visited[neighbor]:
                            queue.append(neighbor)
            
            if len(cluster) >= min_samples:
                clusters.append(np.array(cluster, dtype=int))
        
        return clusters
    
    def points_to_obstacles(self, clustered_points: List[np.ndarray], header) -> ObstacleArray:
        """Convert clustered points to obstacle array"""
        obstacle_array = ObstacleArray()
        obstacle_array.header = header
        obstacle_array.header.frame_id = "base_link"
        
        track_id = 0
        for cluster_points in clustered_points:
            if len(cluster_points) == 0:
                continue
            
            # Calculate cluster statistics
            centroid = cluster_points.mean(axis=0)
            bounds_min = cluster_points.min(axis=0)
            bounds_max = cluster_points.max(axis=0)
            size = bounds_max - bounds_min
            
            # Create obstacle
            obstacle = Obstacle()
            
            # Position and geometry
            obstacle.position_3d = validate_position_3d(centroid.tolist())
            obstacle.size_3d = validate_position_3d(size.tolist())
            obstacle.distance = float(np.linalg.norm(centroid[:2]))
            
            # Classification (unknown for LiDAR-only)
            obstacle.class_id = -1
            obstacle.confidence = self._calculate_cluster_confidence(cluster_points)
            
            # Motion (not available from single scan)
            obstacle.speed = 0.0
            obstacle.relative_speed = 0.0
            
            # Tracking
            obstacle.track_id = track_id
            obstacle.track_age = 0
            track_id += 1
            
            # Sensor information
            obstacle.sensor_type = "lidar_front"
            obstacle.sensor_sources = ["lidar_front"]
            obstacle.point_count = len(cluster_points)
            
            # Image-related fields (not applicable)
            obstacle.bbox = [-1, -1, -1, -1]
            
            # Fusion fields
            obstacle.fusion_distance = obstacle.distance
            
            obstacle_array.obstacles.append(obstacle)
        
        return obstacle_array
    
    def _calculate_cluster_confidence(self, cluster_points: np.ndarray) -> float:
        """Calculate confidence score for a point cluster"""
        if len(cluster_points) == 0:
            return 0.0
        
        # Base confidence on point count (more points = higher confidence)
        point_confidence = min(1.0, len(cluster_points) / 50.0)
        
        # Consider cluster compactness
        if len(cluster_points) > 1:
            centroid = cluster_points.mean(axis=0)
            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            compactness = 1.0 / (1.0 + distances.std())
        else:
            compactness = 1.0
        
        # Combined confidence
        confidence = 0.7 * point_confidence + 0.3 * compactness
        return min(1.0, max(0.1, confidence))
    
    def _create_empty_result(self, msg: PointCloud2) -> ObstacleArray:
        """Create empty obstacle array with proper header"""
        result = ObstacleArray()
        result.header = msg.header
        result.header.frame_id = "base_link"
        return result
    
    def _update_processing_start(self) -> float:
        """Mark start of processing and return timestamp"""
        import time
        return time.time()


# ============================================================================
# CONVENIENCE FUNCTIONS (for backward compatibility)
# ============================================================================

_global_processor: Optional[FrontLidarProcessor] = None

def front_lidar_to_obstacles(msg: PointCloud2, params: Dict) -> ObstacleArray:
    """
    Convenience function for LiDAR preprocessing (maintains global state)
    
    Args:
        msg: Input PointCloud2 message
        params: Processing parameters
        
    Returns:
        ObstacleArray with detected obstacles
    """
    global _global_processor
    
    # Create processor if needed or parameters changed significantly
    if (_global_processor is None or 
        _global_processor._params_changed(_global_processor.processing_params, params)):
        _global_processor = FrontLidarProcessor(params)
    
    return _global_processor.preprocess(msg)


# ============================================================================
# PARAMETER DEFAULTS (for backward compatibility)
# ============================================================================

DEFAULT_LIDAR_PARAMS = {
    'roi': {
        'x_min': 1.0,
        'x_max': 120.0,
        'y_min': -2.2,
        'y_max': 2.2,
        'z_min': -1.5,
        'z_max': 3.0
    },
    'ground_removal': {
        'z_max': 0.20,
        'ransac_distance': 0.08,
        'normal_min_z': 0.85
    },
    'voxel': {
        'size': 0.15,
        'min_points_per_voxel': 1
    },
    'clustering': {
        'method': 'connected_components',
        'min_voxels': 3,
        'max_voxels': 50000,
        'eps': 0.6,
        'min_samples': 6
    }
}