#!/usr/bin/env python3
"""
Radar Preprocessing Module

Converts rear radar PointCloud2 measurements to obstacles for sensor fusion.
Uses Open3D for point cloud processing.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import math
import struct
from lkas_aeb_msgs.msg import Obstacle, ObstacleArray
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("Warning: Open3D not available. Using numpy-only processing.")

class RadarTracker:
    """Simple tracker for radar detections to maintain persistent track IDs."""
    
    def __init__(self):
        self.tracks = {}  # track_id: {'position': (x,y), 'age': int, 'last_seen': int}
        self.next_id = 0
        self.frame_count = 0
        self.max_age = 10
        self.max_association_distance = 5.0  # meters
    
    def update(self, detections):
        """
        Update tracks with new detections.
        
        Args:
            detections: List of (x, y, velocity, confidence) tuples
            
        Returns:
            List of (x, y, velocity, confidence, track_id) tuples
        """
        self.frame_count += 1
        
        # Clean up old tracks
        expired_ids = []
        for track_id, track_data in self.tracks.items():
            if self.frame_count - track_data['last_seen'] > self.max_age:
                expired_ids.append(track_id)
        for track_id in expired_ids:
            del self.tracks[track_id]
        
        # Associate detections with existing tracks
        tracked_detections = []
        used_track_ids = set()
        
        for detection in detections:
            x, y, velocity, confidence = detection
            best_match = None
            min_distance = float('inf')
            
            # Find closest existing track
            for track_id, track_data in self.tracks.items():
                if track_id in used_track_ids:
                    continue
                    
                track_x, track_y = track_data['position']
                distance = math.hypot(x - track_x, y - track_y)
                
                if distance < min_distance and distance < self.max_association_distance:
                    min_distance = distance
                    best_match = track_id
            
            if best_match is not None:
                # Update existing track
                self.tracks[best_match]['position'] = (x, y)
                self.tracks[best_match]['age'] += 1
                self.tracks[best_match]['last_seen'] = self.frame_count
                used_track_ids.add(best_match)
                tracked_detections.append((x, y, velocity, confidence, best_match))
            else:
                # Create new track
                new_id = self.next_id
                self.next_id += 1
                self.tracks[new_id] = {
                    'position': (x, y),
                    'age': 1,
                    'last_seen': self.frame_count
                }
                tracked_detections.append((x, y, velocity, confidence, new_id))
        
        return tracked_detections

# Global tracker instances for left and right radars
_left_tracker = RadarTracker()
_right_tracker = RadarTracker()

def pointcloud2_to_xyz_array(cloud_msg: PointCloud2) -> np.ndarray:
    """
    Convert PointCloud2 message to numpy array of XYZ coordinates.
    
    Args:
        cloud_msg: ROS PointCloud2 message
        
    Returns:
        numpy array of shape (N, 4) with columns [x, y, z, intensity/velocity]
    """
    try:
        # Read points from PointCloud2 message
        points_list = []
        for point in pc2.read_points(cloud_msg, skip_nans=True):
            # CARLA radar points typically have: x, y, z, velocity (Doppler)
            # Some might also have intensity or other fields
            if len(point) >= 3:
                x, y, z = point[0], point[1], point[2]
                # Try to get velocity (Doppler) if available
                velocity = point[3] if len(point) > 3 else 0.0
                points_list.append([x, y, z, velocity])
        
        if not points_list:
            return np.array([]).reshape(0, 4)
            
        return np.array(points_list)
    
    except Exception as e:
        print(f"Error converting PointCloud2: {e}")
        return np.array([]).reshape(0, 4)

def filter_radar_points(points: np.ndarray, params: Dict) -> np.ndarray:
    """
    Filter radar points based on range, elevation, and noise criteria.
    
    Args:
        points: numpy array of shape (N, 4) with [x, y, z, velocity]
        params: Configuration parameters
        
    Returns:
        Filtered numpy array
    """
    if points.size == 0:
        return points
    
    # Range filtering
    ranges = np.linalg.norm(points[:, :3], axis=1)
    min_range = params.get('radar_min_range', 1.0)
    max_range = params.get('radar_max_range', 100.0)
    range_mask = (ranges >= min_range) & (ranges <= max_range)
    
    # Elevation filtering (remove ground clutter and overhead objects)
    elevation_angles = np.arcsin(points[:, 2] / np.maximum(ranges, 0.001))
    min_elevation = np.radians(params.get('radar_min_elevation_deg', -20))
    max_elevation = np.radians(params.get('radar_max_elevation_deg', 20))
    elevation_mask = (elevation_angles >= min_elevation) & (elevation_angles <= max_elevation)
    
    # Velocity filtering (remove static ground clutter)
    velocity_threshold = params.get('radar_min_velocity', 0.5)
    velocity_mask = np.abs(points[:, 3]) >= velocity_threshold
    
    # Combine all filters
    combined_mask = range_mask & elevation_mask & velocity_mask
    
    return points[combined_mask]

def cluster_radar_points_numpy(points: np.ndarray, params: Dict) -> List[Tuple]:
    """
    Cluster radar points using numpy-based DBSCAN-like approach.
    
    Args:
        points: numpy array of shape (N, 4) with [x, y, z, velocity]
        params: Configuration parameters
        
    Returns:
        List of (center_x, center_y, avg_velocity, confidence, point_count) tuples
    """
    if points.size == 0:
        return []
    
    eps = params.get('radar_cluster_eps', 2.0)  # meters
    min_samples = params.get('radar_cluster_min_samples', 2)
    
    clusters = []
    used_points = set()
    
    for i, point in enumerate(points):
        if i in used_points:
            continue
        
        # Find neighbors within eps distance
        center = point[:3]
        distances = np.linalg.norm(points[:, :3] - center, axis=1)
        neighbors = np.where(distances <= eps)[0]
        
        if len(neighbors) >= min_samples:
            # Create cluster
            cluster_points = points[neighbors]
            
            # Calculate cluster properties
            center_x = np.mean(cluster_points[:, 0])
            center_y = np.mean(cluster_points[:, 1])
            avg_velocity = np.mean(cluster_points[:, 3])
            
            # Confidence based on point count and velocity consistency
            velocity_std = np.std(cluster_points[:, 3])
            confidence = min(1.0, len(neighbors) / 10.0) * max(0.1, 1.0 - velocity_std / 5.0)
            
            clusters.append((center_x, center_y, avg_velocity, confidence, len(neighbors)))
            
            # Mark points as used
            for neighbor_idx in neighbors:
                used_points.add(neighbor_idx)
    
    return clusters

def cluster_radar_points_open3d(points: np.ndarray, params: Dict) -> List[Tuple]:
    """
    Cluster radar points using Open3D DBSCAN clustering.
    
    Args:
        points: numpy array of shape (N, 4) with [x, y, z, velocity]
        params: Configuration parameters
        
    Returns:
        List of (center_x, center_y, avg_velocity, confidence, point_count) tuples
    """
    if points.size == 0 or not HAS_OPEN3D:
        return cluster_radar_points_numpy(points, params)
    
    try:
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        
        # DBSCAN clustering
        eps = params.get('radar_cluster_eps', 2.0)
        min_points = params.get('radar_cluster_min_samples', 2)
        
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
        
        # Process clusters
        clusters = []
        max_label = labels.max()
        
        for label in range(max_label + 1):
            cluster_mask = labels == label
            cluster_points = points[cluster_mask]
            
            if len(cluster_points) < min_points:
                continue
            
            # Calculate cluster properties
            center_x = np.mean(cluster_points[:, 0])
            center_y = np.mean(cluster_points[:, 1])
            avg_velocity = np.mean(cluster_points[:, 3])
            
            # Confidence based on cluster size and compactness
            cluster_size = len(cluster_points)
            positions = cluster_points[:, :3]
            compactness = 1.0 / (1.0 + np.std(np.linalg.norm(positions - np.mean(positions, axis=0), axis=1)))
            
            confidence = min(1.0, (cluster_size / 10.0) * compactness)
            
            clusters.append((center_x, center_y, avg_velocity, confidence, cluster_size))
        
        return clusters
    
    except Exception as e:
        print(f"Open3D clustering failed, falling back to numpy: {e}")
        return cluster_radar_points_numpy(points, params)

def apply_roi_filter(detections: List[Tuple], params: Dict) -> List[Tuple]:
    """
    Filter detections to rear Region of Interest.
    
    Args:
        detections: List of detection tuples
        params: Configuration with ROI bounds
        
    Returns:
        Filtered list of detections
    """
    # Default rear ROI: behind vehicle with reasonable lateral extent
    x_min = params.get('rear_roi_x_min', -60.0)  # 60m behind
    x_max = params.get('rear_roi_x_max', 10.0)   # 10m ahead (for lane changes)
    y_min = params.get('rear_roi_y_min', -8.0)   # 8m left
    y_max = params.get('rear_roi_y_max', 8.0)    # 8m right
    
    filtered = []
    for detection in detections:
        x, y = detection[0], detection[1]
        if x_min <= x <= x_max and y_min <= y <= y_max:
            filtered.append(detection)
    
    return filtered

def transform_radar_frame(points: np.ndarray, sensor_side: str) -> np.ndarray:
    """
    Transform points from radar sensor frame to base_link frame.
    
    Args:
        points: numpy array of shape (N, 4) with [x, y, z, velocity]
        sensor_side: 'left' or 'right' to indicate sensor position
        
    Returns:
        Transformed points in base_link frame
    """
    if points.size == 0:
        return points
    
    transformed = points.copy()
    
    if sensor_side == 'left':
        # Left rear radar: typically mounted at rear-left of vehicle
        # Assuming radar faces backward-left at ~135 degrees from vehicle forward
        # Transform: rotate by 135 degrees around Z-axis, then translate
        angle = np.radians(135)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        # Rotation matrix for 135 degrees
        x_new = transformed[:, 0] * cos_a - transformed[:, 1] * sin_a
        y_new = transformed[:, 0] * sin_a + transformed[:, 1] * cos_a
        
        transformed[:, 0] = x_new
        transformed[:, 1] = y_new
        
        # Translate to sensor position (approximate)
        transformed[:, 0] -= 2.0  # 2m behind base_link
        transformed[:, 1] -= 1.0  # 1m left of base_link
        
    elif sensor_side == 'right':
        # Right rear radar: typically mounted at rear-right of vehicle
        # Assuming radar faces backward-right at ~-135 degrees (225 degrees)
        angle = np.radians(-135)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        # Rotation matrix for -135 degrees
        x_new = transformed[:, 0] * cos_a - transformed[:, 1] * sin_a
        y_new = transformed[:, 0] * sin_a + transformed[:, 1] * cos_a
        
        transformed[:, 0] = x_new
        transformed[:, 1] = y_new
        
        # Translate to sensor position
        transformed[:, 0] -= 2.0  # 2m behind base_link
        transformed[:, 1] += 1.0  # 1m right of base_link
    
    return transformed

def rear_radars_to_obstacles(left: PointCloud2,
                             right: PointCloud2,
                             params: Dict) -> ObstacleArray:
    """
    Process rear radar PointCloud2 messages into obstacles.
    
    Args:
        left: Left rear radar PointCloud2 message
        right: Right rear radar PointCloud2 message  
        params: Configuration parameters
        
    Returns:
        ObstacleArray with detected obstacles
    """
    obstacle_array = ObstacleArray()
    
    # Use the most recent timestamp
    if left.header.stamp.sec > right.header.stamp.sec:
        obstacle_array.header = left.header
    elif right.header.stamp.sec > left.header.stamp.sec:
        obstacle_array.header = right.header
    else:
        # Same second, compare nanoseconds
        if left.header.stamp.nanosec >= right.header.stamp.nanosec:
            obstacle_array.header = left.header
        else:
            obstacle_array.header = right.header
    
    obstacle_array.header.frame_id = "base_link"
    
    all_points = []
    
    # Process left radar
    if left.width > 0 and left.height > 0:
        left_points = pointcloud2_to_xyz_array(left)
        if left_points.size > 0:
            # Filter and transform to base_link
            filtered_left = filter_radar_points(left_points, params)
            if filtered_left.size > 0:
                transformed_left = transform_radar_frame(filtered_left, 'left')
                all_points.extend(transformed_left)
    
    # Process right radar
    if right.width > 0 and right.height > 0:
        right_points = pointcloud2_to_xyz_array(right)
        if right_points.size > 0:
            # Filter and transform to base_link
            filtered_right = filter_radar_points(right_points, params)
            if filtered_right.size > 0:
                transformed_right = transform_radar_frame(filtered_right, 'right')
                all_points.extend(transformed_right)
    
    if not all_points:
        return obstacle_array
    
    # Convert to numpy array
    all_points_array = np.array(all_points)
    
    # Apply ROI filtering in base_link frame
    # Convert to format expected by ROI filter (x, y, velocity, confidence, source)
    roi_input = [(p[0], p[1], p[3], 1.0, 'radar') for p in all_points_array]
    filtered_detections = apply_roi_filter(roi_input, params)
    
    if not filtered_detections:
        return obstacle_array
    
    # Convert back to array format for clustering
    cluster_input = np.array([[d[0], d[1], 0.0, d[2]] for d in filtered_detections])
    
    # Cluster detections
    if HAS_OPEN3D:
        clustered = cluster_radar_points_open3d(cluster_input, params)
    else:
        clustered = cluster_radar_points_numpy(cluster_input, params)
    
    if not clustered:
        return obstacle_array
    
    # Apply tracking to maintain persistent IDs
    global _left_tracker  # Use single tracker for combined rear radar
    tracked_detections = _left_tracker.update(clustered)
    
    # Convert clustered detections to obstacles
    for detection in tracked_detections:
        x, y, velocity, confidence, track_id = detection[:5]
        point_count = detection[4] if len(detection) > 5 else 1
        
        obstacle = Obstacle()
        
        # Basic properties
        obstacle.class_id = 0  # Unknown class for radar
        obstacle.distance = math.hypot(x, y)
        obstacle.speed = 0.0  # Radar doesn't directly measure target speed
        obstacle.relative_speed = abs(velocity)  # Doppler velocity magnitude
        obstacle.track_id = track_id
        
        # Bounding box (not applicable for radar)
        obstacle.bbox = [-1, -1, -1, -1]
        
        # Sensor information
        obstacle.sensor_type = "radar_rear"
        obstacle.confidence = confidence
        
        # 3D position and size
        obstacle.position_3d = [x, y, 0.0]
        obstacle.size_3d = [0.0, 0.0, 0.0]  # Unknown size from radar
        
        # Tracking information
        obstacle.track_age = _left_tracker.tracks.get(track_id, {}).get('age', 1)
        
        # Sensor sources (determine which radars contributed)
        sources = ["radar_rear_left", "radar_rear_right"]  # Both by default
        obstacle.sensor_sources = sources
        
        # Fusion properties
        obstacle.fusion_distance = obstacle.distance
        obstacle.point_count = point_count
        
        obstacle_array.obstacles.append(obstacle)
    
    return obstacle_array