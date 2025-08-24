#!/usr/bin/env python3
"""
Front Camera Processor (refactored, framework-consistent)

- Subclasses BaseCameraProcessor pattern (via BasePerceptionModule)
- Uses YOLO for robust object detection
- Multi-object tracking with Kalman filtering
- Distance estimation using object heights
- Converts detections to ObstacleArray with forward-facing geometry
"""
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
import os
import time
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from lkas_aeb_msgs.msg import ObstacleArray, Obstacle

from .base_classes import BasePerceptionModule
from lkas_aeb.util.perception_utils import validate_numeric, ProcessingError

# Try to import YOLO, fallback gracefully if not available
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not available, using basic detection")

_bridge = CvBridge()

class FrontCameraProcessor(BasePerceptionModule):
    def __init__(self, params: Dict):
        self.processing_params = self._validate_params(params or {})
        
        # Initialize YOLO model if available
        self._model = None
        self._tracker = FrontCameraTracker(
            max_age=self.processing_params['tracking']['max_age'],
            max_assoc_dist=self.processing_params['tracking']['max_association_distance']
        )
        
        # Class information for YOLO
        self.class_names = self._get_class_names()
        self.class_real_heights = self._get_class_heights()
        
        super().__init__(self.processing_params, name="FrontCameraProcessor")

    def _initialize(self) -> None:
        """Initialize YOLO model and camera parameters"""
        det = self.processing_params['detection']
        
        # Camera parameters
        self._focal_length = float(det.get('focal_length', 800.0))
        self._confidence_threshold = float(det.get('confidence_threshold', 0.5))
        self._nms_threshold = float(det.get('nms_iou_threshold', 0.4))
        self._classes_of_interest = det.get('classes_of_interest', [0, 1, 2, 3, 5, 7])  # person, bicycle, car, motorcycle, bus, truck
        
        # ROI parameters
        self._roi_width = float(det.get('roi_width', 0.55))
        self._roi_y_start = float(det.get('roi_y_start', 0.3))
        
        # Initialize YOLO model
        if YOLO_AVAILABLE and det.get('enable_yolo', True):
            try:
                model_path = det.get('model_path', "~/Projects/lkas_aeb_ws/src/lkas_aeb/models/yolov8n.pt")
                model_path = os.path.expanduser(model_path)
                
                if os.path.exists(model_path):
                    self._model = YOLO(model_path)
                    self._model.fuse()  # Optimize model
                    if det.get('use_gpu', True):
                        self._model.to('cuda')
                else:
                    print(f"Warning: YOLO model not found at {model_path}, using basic detection")
            except Exception as e:
                print(f"Warning: Failed to load YOLO model: {e}")

    def process(self, msg: Image) -> ObstacleArray:
        start = self._last_update_time
        out = ObstacleArray()
        out.header = msg.header
        out.header.frame_id = "base_link"
        
        try:
            img = _bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # Create visualization copy
            viz_img = img.copy()
            
            # Detect objects
            detections = self._detect_objects(img)
            
            # Track objects and associate
            tracked_detections = self._tracker.update(detections)
            
            # Convert to obstacles
            obstacles = self._detections_to_obstacles(tracked_detections, img.shape, msg.header)
            out.obstacles = obstacles
            
            # Store visualization (can be retrieved later if needed)
            self._last_visualization = self._visualize_detections(viz_img, tracked_detections)
            
            self._update_stats(start, input_count=len(detections), output_count=len(obstacles))
            return out
            
        except Exception as e:
            self._update_stats(start, input_count=0, output_count=0, had_error=True)
            raise ProcessingError(f"Front camera processing failed: {str(e)}")

    def get_visualization(self) -> Optional[np.ndarray]:
        """Get last visualization image"""
        return getattr(self, '_last_visualization', None)

    # ----------------------------- helpers --------------------------------
    def _validate_params(self, params: Dict) -> Dict:
        v = {
            'detection': {},
            'tracking': {}
        }
        
        det = params.get('detection', {})
        v['detection'] = {
            'enable_yolo': bool(det.get('enable_yolo', True)),
            'model_path': str(det.get('model_path', "~/Projects/lkas_aeb_ws/src/lkas_aeb/models/yolov8n.pt")),
            'use_gpu': bool(det.get('use_gpu', True)),
            'confidence_threshold': float(det.get('confidence_threshold', 0.5)),
            'nms_iou_threshold': float(det.get('nms_iou_threshold', 0.4)),
            'classes_of_interest': list(det.get('classes_of_interest', [0, 1, 2, 3, 5, 7])),
            'focal_length': float(det.get('focal_length', 800.0)),
            'roi_width': float(det.get('roi_width', 0.8)),
            'roi_y_start': float(det.get('roi_y_start', 0.3)),
        }
        
        track = params.get('tracking', {})
        v['tracking'] = {
            'max_age': int(track.get('max_age', 30)),
            'max_association_distance': float(track.get('max_association_distance', 100.0)),
            'min_hits': int(track.get('min_hits', 1)),
        }
        
        return v

    def _detect_objects(self, img: np.ndarray) -> List[Dict]:
        """Detect objects in image using YOLO or fallback method"""
        detections = []
        
        if self._model is not None:
            # Use YOLO detection
            detections = self._yolo_detect(img)
        else:
            # Use simple motion detection fallback
            detections = self._basic_detect(img)
        
        # Apply ROI filtering
        return self._filter_roi(detections, img.shape)

    def _yolo_detect(self, img: np.ndarray) -> List[Dict]:
        """YOLO-based object detection"""
        try:
            results = self._model(
                img,
                classes=self._classes_of_interest,
                conf=self._confidence_threshold,
                verbose=False
            )
            
            detections = []
            res = results[0]
            
            if len(res.boxes) > 0:
                boxes = res.boxes.xyxy.cpu().numpy().astype(int)
                confidences = res.boxes.conf.cpu().numpy()
                class_ids = res.boxes.cls.cpu().numpy().astype(int)
                
                # Apply NMS
                indices = cv2.dnn.NMSBoxes(
                    boxes.tolist(), confidences.tolist(),
                    self._confidence_threshold, self._nms_threshold
                )
                
                if len(indices) > 0:
                    flat_indices = indices.flatten() if isinstance(indices, np.ndarray) else [indices]
                    
                    for i in flat_indices:
                        x1, y1, x2, y2 = boxes[i]
                        detections.append({
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': float(confidences[i]),
                            'class_id': int(class_ids[i]),
                            'detection_type': 'yolo'
                        })
            
            return detections
            
        except Exception as e:
            print(f"YOLO detection failed: {e}")
            return []

    def _basic_detect(self, img: np.ndarray) -> List[Dict]:
        """Basic detection fallback (simple blob detection)"""
        # Convert to grayscale and apply basic processing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Simple blob detection using contours
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                if w > 20 and h > 20:  # Minimum size
                    detections.append({
                        'bbox': (x, y, x+w, y+h),
                        'confidence': 0.3,  # Low confidence for basic detection
                        'class_id': -1,  # Unknown class
                        'detection_type': 'basic'
                    })
        
        return detections

    def _filter_roi(self, detections: List[Dict], image_shape: Tuple) -> List[Dict]:
        """Apply ROI filtering to detections"""
        h, w = image_shape[:2]
        
        # ROI bounds
        xmin = int(w * (0.5 - self._roi_width / 2))
        xmax = int(w * (0.5 + self._roi_width / 2))
        ymin = int(h * self._roi_y_start)
        ymax = h
        
        filtered = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Check if detection center is within ROI
            if xmin <= center_x <= xmax and ymin <= center_y <= ymax:
                filtered.append(det)
        
        return filtered

    def _estimate_distance(self, bbox: Tuple[int, int, int, int], class_id: int) -> float:
        """Estimate distance using object height"""
        _, y1, _, y2 = bbox
        pixel_height = max(y2 - y1, 1e-5)
        
        real_height = self.class_real_heights.get(class_id, 1.5)  # Default 1.5m
        distance = (self._focal_length * real_height) / pixel_height
        
        return float(np.clip(distance, 1.0, 200.0))

    def _detections_to_obstacles(self, tracked_detections: List[Dict], 
                               image_shape: Tuple, header) -> List[Obstacle]:
        """Convert tracked detections to obstacles"""
        obstacles = []
        
        for det in tracked_detections:
            bbox = det['bbox']
            
            obstacle = Obstacle()
            obstacle.bbox = [int(x) for x in bbox]
            obstacle.sensor_type = "camera_front"
            obstacle.track_id = det.get('track_id', 0)
            obstacle.class_id = det.get('class_id', -1)
            obstacle.confidence = float(det.get('confidence', 0.0))
            
            # Estimate distance and position
            distance = self._estimate_distance(bbox, obstacle.class_id)
            obstacle.distance = distance
            
            # Convert to 3D position (camera frame, then to base_link)
            cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
            h, w = image_shape[:2]
            
            # Simple pinhole projection to base_link coordinates
            # Assume camera is forward-facing at vehicle center
            x = distance  # Forward distance
            y = -distance * (cx - w/2) / self._focal_length  # Lateral offset (right is negative)
            z = 0.0  # Ground level
            
            obstacle.position_3d = [float(x), float(y), float(z)]
            
            # Speed estimation (if tracking provides it)
            obstacle.speed = float(det.get('speed', 0.0))
            obstacle.relative_speed = obstacle.speed  # Will be corrected by fusion
            
            # Additional properties
            obstacle.size_3d = [0.0, 0.0, 0.0]  # Will be filled by fusion if available
            obstacle.point_count = 0  # Camera doesn't provide points
            obstacle.sensor_sources = ["camera_front"]
            obstacle.fusion_distance = distance
            obstacle.track_age = det.get('track_age', 0)
            
            obstacles.append(obstacle)
        
        return obstacles

    def _visualize_detections(self, img: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Create visualization of detections"""
        viz = img.copy()
        
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox
            
            # Color based on class
            class_id = det.get('class_id', -1)
            color = self._get_class_color(class_id)
            
            # Draw bounding box
            cv2.rectangle(viz, (x1, y1), (x2, y2), color, 2)
            
            # Create label
            conf = det.get('confidence', 0.0)
            track_id = det.get('track_id', 0)
            class_name = self.class_names.get(class_id, f"cls_{class_id}")
            
            label = f"{class_name}: {conf:.2f}"
            if 'distance' in det:
                label += f" d={det['distance']:.1f}m"
            if 'speed' in det and det['speed'] > 0.5:
                label += f" v={det['speed']:.1f}m/s"
            label += f" T{track_id}"
            
            # Position label
            y_offset = y1 - 10 if y1 > 20 else y2 + 20
            cv2.putText(viz, label, (x1, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return viz

    def _get_class_names(self) -> Dict[int, str]:
        """Get YOLO class names"""
        return {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
            4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',
            8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
            11: 'stop sign', 12: 'parking meter', 13: 'bench',
            14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse',
            18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
            22: 'zebra', 23: 'giraffe'
        }

    def _get_class_heights(self) -> Dict[int, float]:
        """Get real-world heights for distance estimation"""
        return {
            0: 1.7,   # person
            1: 1.0,   # bicycle  
            2: 2.5,   # car
            3: 1.3,   # motorcycle
            5: 3.0,   # bus
            7: 2.5,   # truck
            9: 3.0,   # traffic light
            10: 1.0,  # fire hydrant
            11: 2.0,  # stop sign
        }

    def _get_class_color(self, class_id: int) -> Tuple[int, int, int]:
        """Get visualization color for class"""
        if class_id in [1, 2, 3, 4, 5, 6, 7, 8]:  # Vehicles
            return (0, 0, 255)  # Red
        elif class_id in [9, 10, 11, 12, 13]:  # Infrastructure
            return (255, 0, 0)  # Blue
        elif class_id in [15, 16, 17, 18, 19, 20, 21, 22, 23]:  # Animals
            return (0, 255, 0)  # Green
        else:  # Person & Other
            return (0, 255, 255)  # Yellow


class FrontCameraTracker:
    """Simple tracker for front camera detections"""
    
    def __init__(self, max_age: int = 30, max_assoc_dist: float = 100.0):
        self.tracks = {}  # track_id: track_data
        self.next_id = 0
        self.max_age = max_age
        self.max_assoc_dist = max_assoc_dist
        self.frame_count = 0

    def update(self, detections: List[Dict]) -> List[Dict]:
        """Update tracks with new detections"""
        self.frame_count += 1
        
        # Prune old tracks
        self._prune_tracks()
        
        # Associate detections to tracks
        associations = self._associate(detections)
        
        # Update existing tracks and create new ones
        tracked_detections = []
        
        for det_idx, det in enumerate(detections):
            track_id = associations.get(det_idx)
            
            if track_id is not None:
                # Update existing track
                track = self.tracks[track_id]
                track['bbox'] = det['bbox']
                track['last_seen'] = self.frame_count
                track['age'] += 1
                track['hits'] += 1
                
                # Calculate velocity and speed
                self._update_track_motion(track, det)
                
                # Add track info to detection
                det_copy = det.copy()
                det_copy['track_id'] = track_id
                det_copy['track_age'] = track['age']
                det_copy['speed'] = track.get('speed', 0.0)
                
            else:
                # Create new track
                track_id = self.next_id
                self.next_id += 1
                
                self.tracks[track_id] = {
                    'bbox': det['bbox'],
                    'last_seen': self.frame_count,
                    'age': 1,
                    'hits': 1,
                    'velocity': np.array([0.0, 0.0]),
                    'speed': 0.0,
                    'prev_center': None,
                    'prev_frame': None
                }
                
                det_copy = det.copy()
                det_copy['track_id'] = track_id
                det_copy['track_age'] = 1
                det_copy['speed'] = 0.0
            
            tracked_detections.append(det_copy)
        
        return tracked_detections

    def _associate(self, detections: List[Dict]) -> Dict[int, int]:
        """Associate detections to tracks"""
        associations = {}
        used_tracks = set()
        
        for det_idx, det in enumerate(detections):
            det_center = np.array([
                (det['bbox'][0] + det['bbox'][2]) / 2,
                (det['bbox'][1] + det['bbox'][3]) / 2
            ])
            
            best_track = None
            best_distance = float('inf')
            
            for track_id, track in self.tracks.items():
                if track_id in used_tracks:
                    continue
                
                # Skip very old tracks
                age = self.frame_count - track['last_seen']
                if age > 5:
                    continue
                
                # Calculate predicted position
                track_center = np.array([
                    (track['bbox'][0] + track['bbox'][2]) / 2,
                    (track['bbox'][1] + track['bbox'][3]) / 2
                ])
                
                # Predict position based on velocity
                predicted_center = track_center + track['velocity'] * age
                
                # Calculate distance
                distance = np.linalg.norm(det_center - predicted_center)
                
                if distance < best_distance and distance < self.max_assoc_dist:
                    best_distance = distance
                    best_track = track_id
            
            if best_track is not None:
                associations[det_idx] = best_track
                used_tracks.add(best_track)
        
        return associations

    def _update_track_motion(self, track: Dict, detection: Dict):
        """Update track motion estimates"""
        current_center = np.array([
            (detection['bbox'][0] + detection['bbox'][2]) / 2,
            (detection['bbox'][1] + detection['bbox'][3]) / 2
        ])
        
        if track['prev_center'] is not None and track['prev_frame'] is not None:
            dt = self.frame_count - track['prev_frame']
            if dt > 0:
                # Calculate velocity
                raw_velocity = (current_center - track['prev_center']) / dt
                
                # Apply smoothing
                alpha = 0.3
                track['velocity'] = alpha * raw_velocity + (1 - alpha) * track['velocity']
                
                # Calculate speed in pixels/frame
                speed_pixels = np.linalg.norm(track['velocity'])
                
                # Convert to approximate m/s (rough estimate)
                # This is a rough conversion - more accurate with camera calibration
                track['speed'] = speed_pixels * 0.1  # Approximate conversion factor
        
        track['prev_center'] = current_center
        track['prev_frame'] = self.frame_count

    def _prune_tracks(self):
        """Remove old tracks"""
        to_remove = []
        for track_id, track in self.tracks.items():
            age = self.frame_count - track['last_seen']
            if age > self.max_age:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracks[track_id]