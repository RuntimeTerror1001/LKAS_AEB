#!/usr/bin/env python3
"""
Rear Camera Adapter Module

Adapts the existing ObstacleDetector for rear camera processing.
"""

from typing import Dict
import numpy as np
import cv2
from cv_bridge import CvBridge
from lkas_aeb_msgs.msg import ObstacleArray, Obstacle
from sensor_msgs.msg import Image

# Global CV bridge instance
_bridge = CvBridge()

def bbox_to_world_position(bbox, image_shape, params: Dict):
    """
    Convert bounding box to estimated world position using flat ground assumption.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        image_shape: (height, width) of image
        params: Camera parameters
        
    Returns:
        Tuple of (x, y, z) in base_link coordinates, or None if invalid
    """
    try:
        x1, y1, x2, y2 = bbox
        
        # Get camera parameters
        focal_length = params.get('rear_focal_length', 800.0)
        camera_height = params.get('rear_camera_height', 1.5)  # meters above ground
        
        # Calculate bottom center of bounding box (ground contact point)
        bottom_center_x = (x1 + x2) / 2
        bottom_center_y = y2
        
        # Image center
        img_height, img_width = image_shape[:2]
        cx = img_width / 2
        cy = img_height / 2
        
        # Pixel coordinates relative to optical center
        u = bottom_center_x - cx
        v = bottom_center_y - cy
        
        # Flat ground assumption: object touches ground plane
        # Using similar triangles: distance = (focal_length * camera_height) / v_offset
        if v > 0:  # Object below optical center (as expected for ground objects)
            distance = (focal_length * camera_height) / v
            
            # Calculate lateral position
            lateral_offset = (u * distance) / focal_length
            
            # Transform to base_link coordinates (rear camera faces backward)
            # Camera x=right becomes base_link y=right
            # Camera z=forward becomes base_link x=backward
            x_base = -distance  # Negative because rear camera faces backward
            y_base = lateral_offset
            z_base = 0.0  # Assume ground level
            
            return x_base, y_base, z_base
        else:
            return None
            
    except (ValueError, ZeroDivisionError, IndexError):
        return None

def estimate_object_size(bbox, distance, class_id):
    """
    Estimate 3D object size based on bounding box and distance.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        distance: Estimated distance to object
        class_id: YOLO class ID
        
    Returns:
        Tuple of (length, width, height) in meters
    """
    # Default sizes by class (length, width, height in meters)
    default_sizes = {
        0: (0.6, 0.4, 1.7),   # person
        1: (1.8, 0.6, 1.0),   # bicycle  
        2: (4.5, 1.8, 1.5),   # car
        3: (2.2, 0.8, 1.2),   # motorcycle
        5: (12.0, 2.5, 3.0),  # bus
        7: (8.0, 2.5, 2.5),   # truck
    }
    
    default_size = default_sizes.get(class_id, (2.0, 1.0, 1.5))
    
    try:
        x1, y1, x2, y2 = bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        # Use bbox proportions to adjust default sizes
        aspect_ratio = bbox_width / max(bbox_height, 1)
        
        # Scale based on distance and bbox size (rough approximation)
        scale_factor = max(0.5, min(2.0, distance / 10.0))
        
        length = default_size[0] * scale_factor
        width = default_size[1] * scale_factor * aspect_ratio
        height = default_size[2] * scale_factor
        
        return length, width, height
        
    except (ValueError, ZeroDivisionError):
        return default_size

def rear_rgb_to_obstacles(img_msg: Image, detector, params: Dict) -> ObstacleArray:
    """
    Run the existing ObstacleDetector on the rear camera image.
    
    Args:
        img_msg: ROS Image message from rear camera
        detector: Instance of ObstacleDetector
        params: Configuration parameters
        
    Returns:
        ObstacleArray with detected obstacles adapted for rear sensing
    """
    global _bridge
    
    obstacle_array = ObstacleArray()
    obstacle_array.header = img_msg.header
    obstacle_array.header.frame_id = "base_link"
    
    try:
        # Convert ROS image to OpenCV format
        cv_image = _bridge.imgmsg_to_cv2(img_msg, 'bgr8')
        
        # Flip image horizontally for rear camera (mirror effect)
        if params.get('flip_rear_image', True):
            cv_image = cv2.flip(cv_image, 1)
        
        # Run obstacle detection
        _, obstacles = detector.detect(cv_image, img_msg.header.stamp)
        
        if not obstacles:
            return obstacle_array
        
        # Process each detected obstacle
        for obs in obstacles:
            # Obstacle format: (x1, y1, x2, y2, distance, speed, class_id, track_id)
            x1, y1, x2, y2, distance, speed, class_id, track_id = obs
            
            # Create obstacle message
            obstacle = Obstacle()
            
            # Map YOLO classes to our class system
            class_map = {
                0: 4,   # person -> pedestrian
                1: 3,   # bicycle -> bike  
                2: 1,   # car -> car
                3: 3,   # motorcycle -> bike
                5: 2,   # bus -> truck
                7: 2,   # truck -> truck
            }
            obstacle.class_id = class_map.get(class_id, 0)  # Default to unknown
            
            # Distance estimation (use detector's estimate or set to -1 if unreliable)
            if distance > 0 and distance < 200:
                obstacle.distance = float(distance)
            else:
                obstacle.distance = -1.0  # Invalid/unreliable distance
            
            # Speed estimation (rear camera typically doesn't have reliable speed)
            obstacle.speed = 0.0  # Set to 0 for rear camera
            obstacle.relative_speed = 0.0  # Set to 0 for rear camera
            
            # Tracking information
            obstacle.track_id = int(track_id)
            
            # Bounding box (flip coordinates if image was flipped)
            if params.get('flip_rear_image', True):
                img_width = cv_image.shape[1]
                flipped_x1 = img_width - x2
                flipped_x2 = img_width - x1
                obstacle.bbox = [int(flipped_x1), int(y1), int(flipped_x2), int(y2)]
            else:
                obstacle.bbox = [int(x1), int(y1), int(x2), int(y2)]
            
            # Sensor information
            obstacle.sensor_type = "camera_rear"
            
            # Confidence based on detection confidence and bounding box properties
            bbox_area = (x2 - x1) * (y2 - y1)
            img_area = cv_image.shape[0] * cv_image.shape[1]
            bbox_ratio = bbox_area / img_area
            
            # Higher confidence for larger, more central objects
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            img_center_x = cv_image.shape[1] / 2
            img_center_y = cv_image.shape[0] / 2
            
            center_distance = np.hypot(center_x - img_center_x, center_y - img_center_y)
            max_distance = np.hypot(img_center_x, img_center_y)
            centrality = 1.0 - (center_distance / max_distance)
            
            # Combine factors for confidence
            base_confidence = 0.7  # Base confidence for camera detections
            size_factor = min(1.0, bbox_ratio * 1000)  # Boost for larger objects
            central_factor = centrality * 0.3 + 0.7  # Boost for central objects
            
            obstacle.confidence = min(1.0, base_confidence * size_factor * central_factor)
            
            # 3D position estimation using flat ground assumption
            world_pos = bbox_to_world_position(obstacle.bbox, cv_image.shape, params)
            if world_pos:
                obstacle.position_3d = list(world_pos)
            else:
                # Fallback: use distance estimate and bearing
                if obstacle.distance > 0:
                    bearing = np.arctan2(center_x - img_center_x, 
                                       params.get('rear_focal_length', 800.0))
                    x_pos = -obstacle.distance * np.cos(bearing)  # Negative for rear
                    y_pos = obstacle.distance * np.sin(bearing)
                    obstacle.position_3d = [x_pos, y_pos, 0.0]
                else:
                    obstacle.position_3d = [0.0, 0.0, 0.0]
            
            # 3D size estimation
            if obstacle.distance > 0:
                size_3d = estimate_object_size(obstacle.bbox, obstacle.distance, class_id)
                obstacle.size_3d = list(size_3d)
            else:
                obstacle.size_3d = [0.0, 0.0, 0.0]
            
            # Tracking information
            obstacle.track_age = 1  # Camera doesn't maintain long-term tracks
            obstacle.sensor_sources = ["camera_rear"]
            
            # Fusion properties
            obstacle.fusion_distance = obstacle.distance
            obstacle.point_count = 0  # Not applicable for camera
            
            obstacle_array.obstacles.append(obstacle)
        
        # Apply rear-specific filtering
        filtered_obstacles = []
        for obstacle in obstacle_array.obstacles:
            # Filter by position if 3D position is available
            if obstacle.position_3d[0] != 0.0 or obstacle.position_3d[1] != 0.0:
                x, y = obstacle.position_3d[0], obstacle.position_3d[1]
                
                # Rear ROI filter (behind and to sides of vehicle)
                x_min = params.get('rear_camera_roi_x_min', -50.0)
                x_max = params.get('rear_camera_roi_x_max', 5.0)
                y_min = params.get('rear_camera_roi_y_min', -6.0)
                y_max = params.get('rear_camera_roi_y_max', 6.0)
                
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    filtered_obstacles.append(obstacle)
            else:
                # If no 3D position, include if confidence is high enough
                if obstacle.confidence > params.get('min_rear_camera_confidence', 0.5):
                    filtered_obstacles.append(obstacle)
        
        obstacle_array.obstacles = filtered_obstacles
        
    except Exception as e:
        # Log error but return empty array instead of crashing
        print(f"Error in rear camera processing: {str(e)}")
        obstacle_array.obstacles = []
    
    return obstacle_array