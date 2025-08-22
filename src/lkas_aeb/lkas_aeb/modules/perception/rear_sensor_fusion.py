#!/usr/bin/env python3
"""
Rear Sensor Fusion Module

Fuses radar and camera detections from rear sensors.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import math
from lkas_aeb_msgs.msg import ObstacleArray, Obstacle

def calculate_bearing(obstacle: Obstacle) -> float:
    """
    Calculate bearing angle of obstacle from ego vehicle.
    
    Args:
        obstacle: Obstacle message
        
    Returns:
        Bearing angle in radians (-π to π)
    """
    if obstacle.position_3d and (obstacle.position_3d[0] != 0.0 or obstacle.position_3d[1] != 0.0):
        # Use 3D position if available
        x, y = obstacle.position_3d[0], obstacle.position_3d[1]
        return math.atan2(y, -abs(x))  # Bearing relative to rear direction
    elif obstacle.bbox and obstacle.bbox[0] >= 0:
        # Fallback: use bounding box center for bearing estimation
        bbox_center_x = (obstacle.bbox[0] + obstacle.bbox[2]) / 2
        # Assume image width of 800 pixels and focal length for rough bearing
        image_width = 800  # This should be passed as parameter in real implementation
        focal_length = 800
        bearing_angle = math.atan((bbox_center_x - image_width/2) / focal_length)
        return bearing_angle
    else:
        return 0.0

def calculate_position_distance(obs1: Obstacle, obs2: Obstacle) -> float:
    """
    Calculate distance between two obstacles based on their positions.
    
    Args:
        obs1, obs2: Obstacle messages
        
    Returns:
        Distance in meters
    """
    # Prefer 3D position if available
    if (obs1.position_3d and obs2.position_3d and
        (obs1.position_3d[0] != 0.0 or obs1.position_3d[1] != 0.0) and
        (obs2.position_3d[0] != 0.0 or obs2.position_3d[1] != 0.0)):
        
        dx = obs1.position_3d[0] - obs2.position_3d[0]
        dy = obs1.position_3d[1] - obs2.position_3d[1]
        return math.hypot(dx, dy)
    
    # Fallback: use bearing and distance
    bearing1 = calculate_bearing(obs1)
    bearing2 = calculate_bearing(obs2)
    
    dist1 = obs1.distance if obs1.distance > 0 else obs1.fusion_distance
    dist2 = obs2.distance if obs2.distance > 0 else obs2.fusion_distance
    
    if dist1 > 0 and dist2 > 0:
        # Calculate Cartesian positions from polar coordinates
        x1 = dist1 * math.cos(bearing1)
        y1 = dist1 * math.sin(bearing1)
        x2 = dist2 * math.cos(bearing2)
        y2 = dist2 * math.sin(bearing2)
        
        return math.hypot(x2 - x1, y2 - y1)
    
    # If no distance info, use bearing difference as proxy
    bearing_diff = abs(bearing1 - bearing2)
    return bearing_diff * 10.0  # Rough conversion to meters

def calculate_bbox_overlap(bbox1: List[int], bbox2: List[int]) -> float:
    """
    Calculate overlap ratio between two bounding boxes.
    
    Args:
        bbox1, bbox2: Bounding boxes as [x1, y1, x2, y2]
        
    Returns:
        Overlap ratio (0.0 to 1.0)
    """
    if (not bbox1 or not bbox2 or 
        bbox1[0] < 0 or bbox2[0] < 0 or
        len(bbox1) != 4 or len(bbox2) != 4):
        return 0.0
    
    # Calculate intersection
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])
    
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    
    # Calculate areas
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    union_area = bbox1_area + bbox2_area - intersection_area
    
    if union_area <= 0:
        return 0.0
    
    return intersection_area / union_area

def find_best_associations(radar_obstacles: List[Obstacle], 
                          camera_obstacles: List[Obstacle],
                          params: Dict) -> List[Tuple[int, int, float]]:
    """
    Find best associations between radar and camera obstacles.
    
    Args:
        radar_obstacles: List of radar-detected obstacles
        camera_obstacles: List of camera-detected obstacles
        params: Configuration parameters
        
    Returns:
        List of (radar_idx, camera_idx, cost) tuples for associations
    """
    if not radar_obstacles or not camera_obstacles:
        return []
    
    # Association thresholds
    max_position_distance = params.get('fusion_max_position_distance', 3.0)  # meters
    max_bearing_difference = params.get('fusion_max_bearing_diff', 0.3)  # radians (~17 degrees)
    min_bbox_overlap = params.get('fusion_min_bbox_overlap', 0.1)
    
    associations = []
    
    for r_idx, radar_obs in enumerate(radar_obstacles):
        for c_idx, camera_obs in enumerate(camera_obstacles):
            # Calculate association cost components
            position_dist = calculate_position_distance(radar_obs, camera_obs)
            
            # Skip if position difference is too large
            if position_dist > max_position_distance:
                continue
            
            # Calculate bearing difference
            radar_bearing = calculate_bearing(radar_obs)
            camera_bearing = calculate_bearing(camera_obs)
            bearing_diff = abs(radar_bearing - camera_bearing)
            
            # Handle angle wraparound
            if bearing_diff > math.pi:
                bearing_diff = 2 * math.pi - bearing_diff
            
            # Skip if bearing difference is too large
            if bearing_diff > max_bearing_difference:
                continue
            
            # Calculate bounding box overlap (if both have valid bboxes)
            bbox_overlap = 0.0
            if (camera_obs.bbox and len(camera_obs.bbox) == 4 and camera_obs.bbox[0] >= 0 and
                radar_obs.bbox and len(radar_obs.bbox) == 4 and radar_obs.bbox[0] >= 0):
                bbox_overlap = calculate_bbox_overlap(radar_obs.bbox, camera_obs.bbox)
            
            # Calculate combined cost (lower is better)
            # Normalize each component
            position_cost = position_dist / max_position_distance
            bearing_cost = bearing_diff / max_bearing_difference
            bbox_cost = 1.0 - bbox_overlap if bbox_overlap > 0 else 0.5  # Neutral if no bbox
            
            # Weighted combination
            total_cost = (0.4 * position_cost + 
                         0.4 * bearing_cost + 
                         0.2 * bbox_cost)
            
            associations.append((r_idx, c_idx, total_cost))
    
    # Sort by cost and resolve conflicts (greedy assignment)
    associations.sort(key=lambda x: x[2])
    
    used_radar = set()
    used_camera = set()
    final_associations = []
    
    for r_idx, c_idx, cost in associations:
        if r_idx not in used_radar and c_idx not in used_camera:
            final_associations.append((r_idx, c_idx, cost))
            used_radar.add(r_idx)
            used_camera.add(c_idx)
    
    return final_associations

def merge_obstacles(radar_obs: Obstacle, camera_obs: Obstacle) -> Obstacle:
    """
    Merge radar and camera observations into a single fused obstacle.
    
    Args:
        radar_obs: Radar-detected obstacle
        camera_obs: Camera-detected obstacle
        
    Returns:
        Fused obstacle combining both sensor modalities
    """
    fused = Obstacle()
    
    # Header information (use camera timestamp as it's typically more recent)
    # Note: header will be set by the calling function
    
    # Class ID: prefer camera classification (radar doesn't classify)
    fused.class_id = camera_obs.class_id
    
    # Distance: prefer radar (more accurate for distance)
    if radar_obs.distance > 0:
        fused.distance = radar_obs.distance
    elif camera_obs.distance > 0:
        fused.distance = camera_obs.distance
    else:
        fused.distance = radar_obs.fusion_distance
    
    # Speed: prefer radar (Doppler measurement)
    fused.speed = radar_obs.speed if radar_obs.speed != 0 else camera_obs.speed
    fused.relative_speed = radar_obs.relative_speed if radar_obs.relative_speed != 0 else camera_obs.relative_speed
    
    # Track ID: prefer camera (more stable visual tracking)
    if camera_obs.track_id > 0:
        fused.track_id = camera_obs.track_id
    else:
        fused.track_id = radar_obs.track_id
    
    # Bounding box: use camera (radar doesn't provide bbox)
    fused.bbox = camera_obs.bbox
    
    # Sensor information
    fused.sensor_type = "fused"
    
    # Confidence: combine both sensors with weighted average
    radar_weight = 0.3  # Radar contributes to confidence through detection
    camera_weight = 0.7  # Camera provides visual confirmation
    fused.confidence = (radar_weight * radar_obs.confidence + 
                       camera_weight * camera_obs.confidence)
    
    # 3D position: prefer radar position (more accurate ranging)
    if (radar_obs.position_3d and 
        (radar_obs.position_3d[0] != 0.0 or radar_obs.position_3d[1] != 0.0)):
        fused.position_3d = radar_obs.position_3d[:]
    else:
        fused.position_3d = camera_obs.position_3d[:]
    
    # 3D size: use camera estimate (radar doesn't provide size)
    fused.size_3d = camera_obs.size_3d[:]
    
    # Tracking age: use maximum (longer track history)
    fused.track_age = max(radar_obs.track_age, camera_obs.track_age)
    
    # Sensor sources: combine both
    fused.sensor_sources = list(set(radar_obs.sensor_sources + camera_obs.sensor_sources))
    
    # Fusion properties
    fused.fusion_distance = fused.distance
    fused.point_count = radar_obs.point_count  # Radar point count
    
    return fused

def fuse_rear(radar_rear: ObstacleArray,
              cam_rear: ObstacleArray,
              params: Dict) -> ObstacleArray:
    """
    Associate radar and camera obstacles by bearing/position and fuse matched pairs.
    
    Args:
        radar_rear: Obstacles from rear radar sensors
        cam_rear: Obstacles from rear camera
        params: Configuration parameters
        
    Returns:
        Fused ObstacleArray with combined sensor data
    """
    fused_array = ObstacleArray()
    
    # Use the most recent timestamp
    if not radar_rear.obstacles and not cam_rear.obstacles:
        # No obstacles from either sensor
        fused_array.header = radar_rear.header if radar_rear.header.stamp.sec > 0 else cam_rear.header
        fused_array.header.frame_id = "base_link"
        return fused_array
    
    # Determine header timestamp (use the more recent)
    if radar_rear.header.stamp.sec > cam_rear.header.stamp.sec:
        fused_array.header = radar_rear.header
    elif cam_rear.header.stamp.sec > radar_rear.header.stamp.sec:
        fused_array.header = cam_rear.header
    else:
        # Same second, compare nanoseconds
        if radar_rear.header.stamp.nanosec >= cam_rear.header.stamp.nanosec:
            fused_array.header = radar_rear.header
        else:
            fused_array.header = cam_rear.header
    
    fused_array.header.frame_id = "base_link"
    
    # Find associations between radar and camera obstacles
    associations = find_best_associations(radar_rear.obstacles, cam_rear.obstacles, params)
    
    # Track which obstacles have been matched
    matched_radar = set()
    matched_camera = set()
    
    # Process matched pairs
    for radar_idx, camera_idx, cost in associations:
        radar_obs = radar_rear.obstacles[radar_idx]
        camera_obs = cam_rear.obstacles[camera_idx]
        
        # Create fused obstacle
        fused_obstacle = merge_obstacles(radar_obs, camera_obs)
        fused_array.obstacles.append(fused_obstacle)
        
        matched_radar.add(radar_idx)
        matched_camera.add(camera_idx)
    
    # Add unmatched radar obstacles
    for r_idx, radar_obs in enumerate(radar_rear.obstacles):
        if r_idx not in matched_radar:
            # Create copy with updated sensor type
            unmatched_radar = Obstacle()
            
            # Copy all fields
            unmatched_radar.class_id = radar_obs.class_id
            unmatched_radar.distance = radar_obs.distance
            unmatched_radar.speed = radar_obs.speed
            unmatched_radar.relative_speed = radar_obs.relative_speed
            unmatched_radar.track_id = radar_obs.track_id
            unmatched_radar.bbox = radar_obs.bbox[:]
            unmatched_radar.sensor_type = "radar_rear"  # Keep original type
            unmatched_radar.confidence = radar_obs.confidence
            unmatched_radar.position_3d = radar_obs.position_3d[:]
            unmatched_radar.size_3d = radar_obs.size_3d[:]
            unmatched_radar.track_age = radar_obs.track_age
            unmatched_radar.sensor_sources = radar_obs.sensor_sources[:]
            unmatched_radar.fusion_distance = radar_obs.fusion_distance
            unmatched_radar.point_count = radar_obs.point_count
            
            fused_array.obstacles.append(unmatched_radar)
    
    # Add unmatched camera obstacles
    for c_idx, camera_obs in enumerate(cam_rear.obstacles):
        if c_idx not in matched_camera:
            # Create copy with updated sensor type
            unmatched_camera = Obstacle()
            
            # Copy all fields
            unmatched_camera.class_id = camera_obs.class_id
            unmatched_camera.distance = camera_obs.distance
            unmatched_camera.speed = camera_obs.speed
            unmatched_camera.relative_speed = camera_obs.relative_speed
            unmatched_camera.track_id = camera_obs.track_id
            unmatched_camera.bbox = camera_obs.bbox[:]
            unmatched_camera.sensor_type = "camera_rear"  # Keep original type
            unmatched_camera.confidence = camera_obs.confidence
            unmatched_camera.position_3d = camera_obs.position_3d[:]
            unmatched_camera.size_3d = camera_obs.size_3d[:]
            unmatched_camera.track_age = camera_obs.track_age
            unmatched_camera.sensor_sources = camera_obs.sensor_sources[:]
            unmatched_camera.fusion_distance = camera_obs.fusion_distance
            unmatched_camera.point_count = camera_obs.point_count
            
            fused_array.obstacles.append(unmatched_camera)
    
    # Apply post-fusion filtering and validation
    validated_obstacles = []
    for obstacle in fused_array.obstacles:
        # Filter by confidence
        min_confidence = params.get('fusion_min_confidence', 0.2)
        if obstacle.confidence < min_confidence:
            continue
        
        # Filter by distance
        max_distance = params.get('fusion_max_distance', 100.0)
        if obstacle.distance > max_distance:
            continue
        
        # Validate position consistency
        if obstacle.position_3d and obstacle.distance > 0:
            calculated_distance = math.hypot(obstacle.position_3d[0], obstacle.position_3d[1])
            distance_error = abs(calculated_distance - obstacle.distance) / max(obstacle.distance, 1.0)
            
            # Allow some tolerance for position/distance mismatch
            if distance_error > params.get('fusion_max_distance_error', 0.5):
                # Fix position based on distance
                if calculated_distance > 0:
                    scale = obstacle.distance / calculated_distance
                    obstacle.position_3d[0] *= scale
                    obstacle.position_3d[1] *= scale
        
        validated_obstacles.append(obstacle)
    
    fused_array.obstacles = validated_obstacles
    return fused_array