#!/usr/bin/env python3
"""
Perception Utilities 

Common utilities shared across all perception modules including:
- Bounding box operations
- Coordinate transformations
- Point cloud utilities
- Validation functions
- Mathematical helpers
"""

from typing import Dict, List, Optional, Tuple, Union
import math
import numpy as np
from dataclasses import dataclass

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class BBox:
    """Standardized bounding box representation"""
    x1: int
    y1: int  
    x2: int
    y2: int
    
    @classmethod
    def from_list(cls, bbox_list: List[int]) -> Optional['BBox']:
        """Create BBox from list, return None if invalid"""
        if not bbox_list or len(bbox_list) != 4:
            return None
        try:
            return cls(int(bbox_list[0]), int(bbox_list[1]), 
                      int(bbox_list[2]), int(bbox_list[3]))
        except (ValueError, TypeError):
            return None
    
    def to_list(self) -> List[int]:
        """Convert to list format"""
        return [self.x1, self.y1, self.x2, self.y2]
    
    def is_valid(self) -> bool:
        """Check if bounding box is valid"""
        return (self.x1 >= 0 and self.y1 >= 0 and 
                self.x2 > self.x1 and self.y2 > self.y1)
    
    def area(self) -> int:
        """Calculate bounding box area"""
        if not self.is_valid():
            return 0
        return (self.x2 - self.x1) * (self.y2 - self.y1)
    
    def center(self) -> Tuple[float, float]:
        """Get center coordinates"""
        return ((self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5)
    
    def grow(self, padding: int) -> 'BBox':
        """Grow bounding box by padding"""
        return BBox(self.x1 - padding, self.y1 - padding,
                   self.x2 + padding, self.y2 + padding)

@dataclass
class Point3D:
    """3D point representation"""
    x: float
    y: float
    z: float
    
    def distance_to(self, other: 'Point3D') -> float:
        """Calculate distance to another point"""
        return math.sqrt((self.x - other.x)**2 + 
                        (self.y - other.y)**2 + 
                        (self.z - other.z)**2)
    
    def distance_2d(self) -> float:
        """Calculate 2D distance from origin"""
        return math.hypot(self.x, self.y)
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array"""
        return np.array([self.x, self.y, self.z], dtype=float)


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def validate_numeric(value, default: float = 0.0, 
                     min_val: float = -float('inf'),
                     max_val: float = float('inf')) -> float:
    """Safely convert to float with bounds checking"""
    try:
        if value is None:
            return float(default)
        val = float(value)
        if not math.isfinite(val):
            return float(default)
        return max(min_val, min(max_val, val))
    except (ValueError, TypeError):
        return float(default)

def validate_integer(value, default: int = 0,
                    min_val: int = -2**31, 
                    max_val: int = 2**31-1) -> int:
    """Safely convert to int with bounds checking"""
    try:
        if value is None:
            return default
        val = int(value)
        return max(min_val, min(max_val, val))
    except (ValueError, TypeError):
        return default

def validate_bbox(bbox: Union[List, Tuple, BBox, None]) -> List[int]:
    if isinstance(bbox, BBox):
        return bbox.to_list() if bbox.is_valid() else [-1, -1, -1, -1]

    if bbox is None:
        return [-1, -1, -1, -1]

    try:
        if len(bbox) != 4:
            return [-1, -1, -1, -1]
        sanitized = [validate_integer(x, -1) for x in bbox]
        if all(x >= 0 for x in sanitized) and sanitized[2] > sanitized[0] and sanitized[3] > sanitized[1]:
            return sanitized
    except Exception:
        pass

    return [-1, -1, -1, -1]

def validate_position_3d(position: Union[List, np.ndarray, None]) -> List[float]:
    """Validate and sanitize 3D position"""
    if position is None:
        return [0.0, 0.0, 0.0]
    
    try:
        if len(position) < 3:
            position = list(position) + [0.0] * (3 - len(position))
        return [validate_numeric(x) for x in position[:3]]
    except (TypeError, IndexError):
        return [0.0, 0.0, 0.0]


# ============================================================================
# BOUNDING BOX OPERATIONS
# ============================================================================

def bbox_overlap(bbox1: List[int], bbox2: List[int]) -> float:
    """Calculate IoU overlap between two bounding boxes"""
    b1 = validate_bbox(bbox1)
    b2 = validate_bbox(bbox2)
    
    if b1[0] < 0 or b2[0] < 0:
        return 0.0
    
    # Calculate intersection
    x_left = max(b1[0], b2[0])
    y_top = max(b1[1], b2[1])
    x_right = min(b1[2], b2[2])
    y_bottom = min(b1[3], b2[3])
    
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    
    # Areas
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - intersection
    
    return intersection / max(union, 1e-6)

def bbox_contains_point(bbox: List[int], u: float, v: float) -> bool:
    """Check if point (u,v) is inside bounding box"""
    b = validate_bbox(bbox)
    if b[0] < 0:
        return False
    return b[0] <= u <= b[2] and b[1] <= v <= b[3]

def bbox_center(bbox: List[int]) -> Tuple[float, float]:
    """Get center of bounding box"""
    b = validate_bbox(bbox)
    if b[0] < 0:
        return (0.0, 0.0)
    return ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5)

def bbox_area(bbox: List[int]) -> int:
    """Calculate bounding box area"""
    b = validate_bbox(bbox)
    if b[0] < 0:
        return 0
    return max(0, (b[2] - b[0]) * (b[3] - b[1]))


# ============================================================================
# COORDINATE TRANSFORMATIONS
# ============================================================================

def homogeneous_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply homogeneous transformation to points"""
    points = np.asarray(points, dtype=float)
    transform = np.asarray(transform, dtype=float).reshape(4, 4)
    
    # Handle both single points and arrays
    if points.ndim == 1:
        if len(points) == 3:
            points_h = np.array([points[0], points[1], points[2], 1.0])
        elif len(points) == 4:
            points_h = points
        else:
            raise ValueError("Points must be 3D or 4D")
        result = transform @ points_h
        return result[:3] / result[3]
    else:
        # Multiple points
        if points.shape[1] == 3:
            ones = np.ones((points.shape[0], 1))
            points_h = np.hstack([points, ones])
        elif points.shape[1] == 4:
            points_h = points
        else:
            raise ValueError("Points must be 3D or 4D")
        
        result = (transform @ points_h.T).T
        return result[:, :3] / result[:, 3:4]

def project_to_image(points_3d: np.ndarray, 
                    camera_matrix: np.ndarray,
                    transform_cam_base: Optional[np.ndarray] = None) -> np.ndarray:
    """Project 3D points to image coordinates"""
    points = np.asarray(points_3d, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    
    # Transform to camera frame if needed
    if transform_cam_base is not None:
        points = homogeneous_transform(points, transform_cam_base)
    
    # Filter points behind camera
    valid_mask = points[:, 2] > 0
    
    if not np.any(valid_mask):
        return np.empty((0, 2))
    
    valid_points = points[valid_mask]
    
    # Project to image
    K = np.asarray(camera_matrix, dtype=float)
    if K.shape != (3, 3):
        raise ValueError("Camera matrix must be 3x3")
    
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    u = fx * (valid_points[:, 0] / valid_points[:, 2]) + cx
    v = fy * (valid_points[:, 1] / valid_points[:, 2]) + cy
    
    # Create output array with invalid points marked as NaN
    result = np.full((len(points), 2), np.nan)
    result[valid_mask] = np.column_stack([u, v])
    
    return result

def camera_ray_to_ground(u: float, v: float, 
                        camera_matrix: np.ndarray,
                        camera_height: float = 1.5,
                        transform_base_cam: Optional[np.ndarray] = None) -> Optional[Point3D]:
    """Cast ray from image point to ground plane intersection"""
    try:
        K = np.asarray(camera_matrix, dtype=float)
        fx, fy = K[0, 0], K[1, 1] 
        cx, cy = K[0, 2], K[1, 2]
        
        # Ray direction in camera frame
        ray_dir = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
        ray_dir = ray_dir / np.linalg.norm(ray_dir)
        
        # Ray origin (camera position)
        ray_origin = np.array([0.0, 0.0, 0.0])
        
        # Transform to base_link if needed
        if transform_base_cam is not None:
            ray_origin = homogeneous_transform(ray_origin, transform_base_cam)
            ray_dir = (transform_base_cam[:3, :3] @ ray_dir.reshape(-1, 1)).flatten()
        
        # Ground plane intersection (z = -camera_height in base_link)
        ground_z = -camera_height
        if abs(ray_dir[2]) < 1e-6:  # Ray parallel to ground
            return None
        
        t = (ground_z - ray_origin[2]) / ray_dir[2]
        if t <= 0:  # Intersection behind camera
            return None
        
        intersection = ray_origin + t * ray_dir
        return Point3D(intersection[0], intersection[1], intersection[2])
        
    except (ValueError, np.linalg.LinAlgError):
        return None


# ============================================================================
# POINT CLOUD UTILITIES  
# ============================================================================

def roi_filter_3d(points: np.ndarray, roi_params: Dict) -> np.ndarray:
    """Apply 3D region-of-interest filter"""
    if points.size == 0:
        return points
    
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    mask = np.ones(len(points), dtype=bool)
    
    if 'x_min' in roi_params:
        mask &= (x >= roi_params['x_min'])
    if 'x_max' in roi_params:
        mask &= (x <= roi_params['x_max'])
    if 'y_min' in roi_params:
        mask &= (y >= roi_params['y_min'])
    if 'y_max' in roi_params:
        mask &= (y <= roi_params['y_max'])
    if 'z_min' in roi_params:
        mask &= (z >= roi_params['z_min'])
    if 'z_max' in roi_params:
        mask &= (z <= roi_params['z_max'])
    
    return points[mask]

def voxel_downsample(points: np.ndarray, voxel_size: float, 
                    min_points_per_voxel: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Downsample point cloud using voxel grid"""
    if points.size == 0:
        return np.empty((0, 3)), np.empty((0, 3), dtype=int)
    
    inv_voxel = 1.0 / max(voxel_size, 1e-6)
    voxel_indices = np.floor(points * inv_voxel).astype(np.int32)
    
    # Use lexicographic sorting for consistent voxel grouping
    sort_order = np.lexsort((voxel_indices[:, 2], voxel_indices[:, 1], voxel_indices[:, 0]))
    sorted_indices = voxel_indices[sort_order]
    sorted_points = points[sort_order]
    
    # Find unique voxels
    unique_mask = np.ones(len(sorted_indices), dtype=bool)
    unique_mask[1:] = np.any(sorted_indices[1:] != sorted_indices[:-1], axis=1)
    
    change_points = np.where(unique_mask)[0]
    segment_starts = change_points
    segment_ends = np.r_[change_points[1:], len(sorted_indices)]
    
    centroids = []
    voxel_coords = []
    
    for start, end in zip(segment_starts, segment_ends):
        segment_size = end - start
        if segment_size >= min_points_per_voxel:
            segment = sorted_points[start:end]
            centroids.append(segment.mean(axis=0))
            voxel_coords.append(sorted_indices[start])
    
    if not centroids:
        return np.empty((0, 3)), np.empty((0, 3), dtype=int)
    
    return np.vstack(centroids), np.vstack(voxel_coords)

def connected_components_2d(voxel_indices: np.ndarray,
                           connectivity: int = 8) -> List[np.ndarray]:
    """Find connected components in 2D voxel grid"""
    if voxel_indices.size == 0:
        return []
    
    from collections import defaultdict, deque
    
    # Map 2D coordinates to row indices
    coords_2d = voxel_indices[:, :2]  # Only x,y coordinates
    coord_to_rows = defaultdict(list)
    
    for row, (x, y) in enumerate(coords_2d):
        coord_to_rows[(int(x), int(y))].append(row)
    
    # Define neighborhood based on connectivity
    if connectivity == 4:
        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    else:  # 8-connectivity
        neighbors = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]
    
    visited = set()
    components = []
    
    for coord in coord_to_rows:
        if coord in visited:
            continue
        
        # BFS to find connected component
        queue = deque([coord])
        visited.add(coord)
        component_rows = []
        
        while queue:
            cx, cy = queue.popleft()
            component_rows.extend(coord_to_rows[(cx, cy)])
            
            for dx, dy in neighbors:
                neighbor = (cx + dx, cy + dy)
                if neighbor in coord_to_rows and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        if component_rows:
            components.append(np.array(component_rows, dtype=int))
    
    return components


# ============================================================================
# MATHEMATICAL UTILITIES
# ============================================================================

def mahalanobis_distance(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    """Calculate Mahalanobis distance"""
    try:
        diff = np.asarray(x) - np.asarray(mean)
        if diff.ndim == 1:
            diff = diff.reshape(-1, 1)
        
        cov_inv = np.linalg.inv(cov)
        distance_sq = float((diff.T @ cov_inv @ diff).item())
        return math.sqrt(max(0, distance_sq))
    except (np.linalg.LinAlgError, ValueError):
        return float('inf')

def angle_difference(angle1: float, angle2: float) -> float:
    """Calculate smallest angle difference (handling wraparound)"""
    diff = angle1 - angle2
    while diff > math.pi:
        diff -= 2 * math.pi
    while diff < -math.pi:
        diff += 2 * math.pi
    return abs(diff)

def weighted_average(values: List[float], weights: List[float]) -> float:
    """Calculate weighted average"""
    if not values or not weights or len(values) != len(weights):
        return 0.0
    
    total_weight = sum(weights)
    if total_weight == 0:
        return sum(values) / len(values)
    
    return sum(v * w for v, w in zip(values, weights)) / total_weight


# ============================================================================
# TIMING AND PERFORMANCE
# ============================================================================

def time_function(func):
    """Decorator to measure function execution time"""
    import time
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        # Store timing info in function attribute
        if not hasattr(wrapper, 'timing_history'):
            wrapper.timing_history = []
        wrapper.timing_history.append(end_time - start_time)
        
        # Keep only recent measurements
        if len(wrapper.timing_history) > 100:
            wrapper.timing_history = wrapper.timing_history[-100:]
        
        return result
    
    def get_avg_time():
        if hasattr(wrapper, 'timing_history') and wrapper.timing_history:
            return sum(wrapper.timing_history) / len(wrapper.timing_history)
        return 0.0
    
    wrapper.get_avg_time = get_avg_time
    return wrapper

def create_intrinsic_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Create camera intrinsic matrix from parameters"""
    return np.array([[fx,  0, cx],
                     [ 0, fy, cy],
                     [ 0,  0,  1]], dtype=float)


# ============================================================================
# ERROR HANDLING
# ============================================================================

class PerceptionError(Exception):
    """Base exception for perception module errors"""
    pass

class ValidationError(PerceptionError):
    """Raised when input validation fails"""
    pass

class TransformError(PerceptionError):
    """Raised when coordinate transformation fails"""
    pass

class ProcessingError(PerceptionError):
    """Raised when processing fails"""
    pass

def safe_execute(func, default_return=None, error_types=(Exception,)):
    """Safely execute function with error handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except error_types:
            return default_return
    return wrapper