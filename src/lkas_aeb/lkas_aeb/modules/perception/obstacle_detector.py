import cv2
import os
import numpy as np
from ultralytics import YOLO

"""
OBSTACLE DETECTOR CLASS
"""

class ObstacleDetector:
    """
        Real-time obstacle detection system using YOLOv8 with multi-object tracking.
        
        Features:
        - Object detection and classification
        - Distance estimation using object height
        - Velocity estimation with object tracking
        - Region of Interest (ROI) filtering for efficiency
        
        Input: Camera images from vehicle
        Output: Detected obstacles with position, distance, and velocity
    """
    
    def __init__(self, params):
        """
        Initialize obstacle detector with configuration parameters.
        
        Args:
            params (dict): Configuration dictionary containing 'perception' section
                - classes_of_interest: List of YOLO class IDs to detect
                - confidence_threshold: Minimum detection confidence
                - focal_length: Camera focal length for distance estimation
                - roi_width: Width of region of interest (0-1)
                - nms_iou_threshold: IoU threshold for Non-Maximum Suppression
        """
        # ========================
        # MODEL INITIALIZATION
        # ========================
        self.params = params['perception']
        self.model = YOLO(os.path.expanduser("~/Projects/lkas_aeb_ws/src/lkas_aeb/models/yolov8n.pt")) # .export(format='engine', half=True) TensorRT
        self.model.fuse() # Optimize model for inference
        self.model.to('cuda') # Use GPU if available
        
        # ========================
        # DETECTION CONFIGURATION
        # ========================
        self.classes = self.params['classes_of_interest']
        self.conf_thres = self.params['confidence_threshold']
        
        # ========================
        # TRACKING VARIABLES
        # ========================
        self.active_tracks = {} # track_id: (bbox, velocity, last_frame, track_age)
        self.next_track_id = 0
        self.max_tracking_age = 30 # frames
        self.max_association_distance = 100 # pixels
        self.frame_count = 0

        # ========================
        # DISTANCE ESTIMATION SETUP
        # ========================
        # Real-world heights of objects in meters for distance calculation
        self.class_real_heights = {
            0: 1.7, # person
            1: 1.0, # bicycle
            2: 2.5, # car
            3: 1.3, # motorcycle
            5: 3.0, # bus
            7: 2.5 # truck
        }

        # ========================
        # CAMERA PARAMETERS
        # ========================
        self.focal_length = self.params['focal_length']
        self.frame_rate = 30.0
        self.last_time = None
        
        # Get class names for visualization
        self.class_names = self.get_class_names()
        self.focal_length = self.params['focal_length']
        self.frame_rate = 30.0
        self.last_time = None
        
        self.class_names = self.get_class_names()

    def get_class_names(self):
        """
        Get YOLO class names for the first 24 classes.
        
        Returns:
            dict: Mapping of class_id to class_name
        """
        all_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 
            'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 
            'cow', 'elephant', 'bear', 'zebra', 'giraffe'
        ]
        return {i:name for i, name in enumerate(all_names[:24])}

    def get_class_color(self, class_id):
        """
        Get visualization color for different object classes.
        
        Args:
            class_id (int): YOLO class identifier
            
        Returns:
            tuple: BGR color tuple for OpenCV visualization
        """
        colors = [
            (0, 0, 255), # Red - Vehicles
            (255, 0, 0), # Blue - Infrastructure
            (0, 255, 0), # Green - Animals
            (0, 255, 255) # Yellow - Other
        ]

        if class_id in [1,2,3,4,5,6,7,8]: # Vehicles
            return colors[0]
        elif class_id in [9, 10, 11, 12, 13]: # Infrastructure
            return colors[1]
        elif class_id in [15,16,17,18,19,20,21,22,23]: # Animals
            return colors[2]
        else: # Person & Other
            return colors[3]
    
    def create_roi_mask(self, image_shape):
        """
        Create Region of Interest mask to focus processing on road area.
        
        Args:
            image_shape (tuple): Image dimensions (height, width, channels)
            
        Returns:
            tuple: (mask, roi_bounds) where:
                - mask: Binary mask for ROI
                - roi_bounds: (xmin, ymin, xmax, ymax) coordinates
        """
        h, w = image_shape[:2]
        mask = np.zeros((h,w), dtype=np.uint8)

        # Define ROI based on params
        roi_width = self.params['roi_width']
        xmin = int(w * (0.5 - roi_width / 2))
        xmax = int(w * (0.5 + roi_width / 2))

        # Focus on lower 2/3 of image (road area)
        ymin = int(h * 0.3)
        mask[ymin:h, xmin:xmax] = 255

        return mask, (xmin, ymin, xmax, h)
    
    def estimate_distance(self, box, class_id):
        """
        Estimate distance to object using pinhole camera model.
        Formula: distance = (focal_length * real_height) / pixel_height
        
        Args:
            box (tuple): Bounding box coordinates (x1, y1, x2, y2)
            class_id (int): Object class identifier
            
        Returns:
            float: Estimated distance in meters (clipped to 1-200m range)
        """
        real_height = self.class_real_heights.get(class_id, 1.5)
        _, y1, _, y2 = box
        pixel_height = y2 - y1 + 1e-5
        distance = (self.focal_length * real_height) / pixel_height

        return np.clip(distance, 1.0, 200.0)
    
    def cleanup_old_tracks(self):
        """
        Remove expired tracks to prevent memory leak.
        Tracks are expired if not updated for max_tracking_age frames.
        """
        current_frame = self.frame_count
        expired_ids = [
            track_id for track_id, data in self.active_tracks.items()
            if current_frame - data['last_frame'] > self.max_tracking_age
        ]
        for track_id in expired_ids:
            del self.active_tracks[track_id]
        
    def associate_detections(self, curr_boxes):
        """
        Associate current detections with existing tracks using nearest neighbor matching.
        
        Args:
            curr_boxes (list): List of current detection bounding boxes
            
        Returns:
            dict: Mapping of detection_index to track_id
        """
        if not self.active_tracks:
            # First frame or no existing tracks
            return {i: self.next_track_id + i for i in range(len(curr_boxes))}
        
        matches = {}
        used_track_ids = set()

        for i, curr_box in enumerate(curr_boxes):
            curr_center = np.array([(curr_box[0] + curr_box[2]) / 2, (curr_box[1] + curr_box[3]) / 2])
            
            best_match = None
            min_distance = float('inf')
            
            for track_id, track_data in self.active_tracks.items():
                if track_id in used_track_ids:
                    continue
                    
                # Skip very old tracks
                if self.frame_count - track_data['last_frame'] > 5:
                    continue

                prev_box = track_data['bbox']
                velocity = track_data['velocity']
                    
                prev_center = np.array([(prev_box[0] + prev_box[2]) / 2, (prev_box[1] + prev_box[3]) / 2])
                
                # Predict position based on velocity
                dt = max(1, self.frame_count - track_data['last_frame'])
                predicted_center = prev_center + velocity * dt
                
                # Calculate association cost (distance + size similarity)
                distance = np.linalg.norm(curr_center - predicted_center)
                
                # Size similarity check
                curr_area = (curr_box[2] - curr_box[0]) * (curr_box[3] - curr_box[1])
                prev_area = (prev_box[2] - prev_box[0]) * (prev_box[3] - prev_box[1])
                size_ratio = min(curr_area, prev_area) / max(curr_area, prev_area)
                
                # Combined cost (lower is better)
                cost = distance * (2 - size_ratio)  # Penalize size differences
                
                if cost < min_distance and distance < self.max_association_distance:
                    min_distance = cost
                    best_match = track_id

            if best_match is not None:
                matches[i] = best_match
                used_track_ids.add(best_match)
            else:
                # Create new track
                matches[i] = self.next_track_id
                self.next_track_id += 1

        return matches
    
    def calculate_velocity(self, curr_box, track_id, dt):
        """
        Calculate object velocity with exponential smoothing filter.
        
        Args:
            curr_box (tuple): Current bounding box coordinates
            track_id (int): Track identifier
            dt (float): Time delta in frames
            
        Returns:
            np.array: Smoothed velocity vector in pixels/frame
        """
        if track_id not in self.active_tracks:
            return np.array([0.0, 0.0])
        
        track_data = self.active_tracks[track_id]
        prev_box = track_data['bbox']
        prev_velocity = track_data['velocity']

        # Calculate center displacement
        curr_center = np.array([(curr_box[0] + curr_box[2]) / 2, (curr_box[1] + curr_box[3]) / 2])
        prev_center = np.array([(prev_box[0] + prev_box[2]) / 2, (prev_box[1] + prev_box[3]) / 2])

        # Raw velocity (in pixels / frame)
        raw_velocity = (curr_center - prev_center) / max(dt, 1)

        # Apply exponential smoothing filter
        alpha = 0.3 # Smoothing factor (0 = no update, 1 = no smoothing)
        smoothed_velocity = alpha * raw_velocity + (1 - alpha) * prev_velocity

        # Limit maximum velocity to reasonable bounds (max 20 pixels / frame)
        velocity_magnitude = np.linalg.norm(smoothed_velocity)
        if velocity_magnitude > 20:
            smoothed_velocity = smoothed_velocity * (20 / velocity_magnitude)
        
        return smoothed_velocity

    def calculate_speed_mps(self, velocity_pixels, box, class_id):
        """
        Convert pixel velocity to real-world speed in meters per second.
        
        Args:
            velocity_pixels (np.array): Velocity vector in pixels/frame
            box (tuple): Bounding box for scale estimation
            class_id (int): Object class for real-world size reference
            
        Returns:
            float: Speed in meters per second
        """
        # Esimtate object height in pixels
        pixel_height = max(box[3] - box[1], 1e-5)

        # Use class-specific real height
        real_height = self.class_real_heights.get(class_id, 1.5)
        
        # Convert pixels to meters / sec
        meter_per_pixels = real_height / pixel_height
        velocity_magnitude = np.linalg.norm(velocity_pixels) * self.frame_rate

        return velocity_magnitude * meter_per_pixels

    def detect(self, image, stamp):
        """
        Main detection function that processes image and returns obstacles with tracking.
        
        Args:
            image (np.array): Input camera image (BGR format)
            stamp: ROS timestamp for timing calculations
            
        Returns:
            tuple: (visualization_image, obstacles_list) where:
                - visualization_image: Image with detection overlays
                - obstacles_list: List of tuples (x1, y1, x2, y2, distance, speed, class_id, track_id)
        """
        # ========================
        # TIMING AND FRAME UPDATE
        # ========================
        self.frame_count += 1
        curr_time = stamp.sec + stamp.nanosec / 1e9
        dt = 1.0  # Default dt in frames
        if self.last_time is not None:
            frame_dt = (curr_time - self.last_time) * self.frame_rate
            dt = max(1, int(frame_dt))
        self.last_time = curr_time

        # Cleanup old tracks
        self.cleanup_old_tracks()
        
        # ========================
        # ROI PROCESSING
        # ========================
        # Create ROI mask for efficiency
        roi_mask, roi_bounds = self.create_roi_mask(image.shape)
        
        # ========================
        # YOLO INFERENCE
        # ========================
        # Run inference
        results = self.model(
            image,
            classes=self.classes,
            conf=self.conf_thres,
            verbose=False
        )

        # ========================
        # DETECTION PROCESSING
        # ========================
        res = results[0]
        if len(res.boxes) == 0:
            return image.copy(), []

        boxes = res.boxes.xyxy.cpu().numpy().astype(int)
        confidences = res.boxes.conf.cpu().numpy()
        class_ids = res.boxes.cls.cpu().numpy().astype(int)

        # ========================
        # ROI FILTERING
        # ========================
        roi_filtered_indices = []
        roi_filtered_boxes = []
        roi_filtered_confs = []
        roi_filtered_classes = []

        xmin, ymin, xmax, ymax = roi_bounds
        
        for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Check if detection center is within ROI
            if xmin <= center_x <= xmax and ymin <= center_y <= ymax:
                roi_filtered_indices.append(i)
                roi_filtered_boxes.append(box)
                roi_filtered_confs.append(conf)
                roi_filtered_classes.append(cls_id)

        if not roi_filtered_boxes:
            return image.copy(), []

        # ========================
        # NON-MAXIMUM SUPPRESSION
        # ========================
        indices = cv2.dnn.NMSBoxes(
            roi_filtered_boxes, roi_filtered_confs,
            self.conf_thres, self.params['nms_iou_threshold']
        )

        # ========================
        # FINAL DETECTION PROCESSING
        # ========================
        obstacles = []
        viz = image.copy()
        
        if len(indices) > 0:
            flat_indices = indices.flatten() if isinstance(indices, (list, np.ndarray)) else [indices]
            
            # Associate detections with existing tracks
            final_boxes = [roi_filtered_boxes[i] for i in flat_indices]
            matches = self.associate_detections(final_boxes)
            
            for i, box_idx in enumerate(flat_indices):
                box = roi_filtered_boxes[box_idx]
                cls_id = roi_filtered_classes[box_idx]
                conf = roi_filtered_confs[box_idx]
                track_id = matches[i]
                
                x1, y1, x2, y2 = box

                # Calculate distance using pinhole camera model
                distance = self.estimate_distance(box, cls_id)

                # Calculate velocity and convert to real-world speed
                velocity_pixels = self.calculate_velocity(box, track_id, dt)
                speed_mps = self.calculate_speed_mps(velocity_pixels, box, cls_id)

                # Update track information
                track_age = 1
                if track_id in self.active_tracks:
                    track_age = self.active_tracks[track_id]['track_age'] + 1
                self.active_tracks[track_id] = {
                    'bbox': box,
                    'velocity': velocity_pixels,
                    'last_frame': self.frame_count,
                    'track_age': track_age
                }

                # Store obstacle (format compatible with AEB Controller)
                obstacles.append((x1, y1, x2, y2, distance, speed_mps, cls_id, track_id))

                # ========================
                # VISUALIZATION
                # ========================
                color = self.get_class_color(cls_id)
                cv2.rectangle(viz, (x1, y1), (x2, y2), color, 2)
                
                # Created enhanced label with all information
                label = f"{self.class_names.get(cls_id, str(cls_id))}: {conf:.2f}"
                label += f" d={distance:.1f}m"
                if speed_mps > 0.5:  # Only show speed if significant
                    label += f" v={speed_mps:.1f}m/s"
                label += f" T{track_id}"
                
                # Position text to avoid overlap with bounding box
                y_offset = y1 - 10
                if y_offset < 20:
                    y_offset = y2 + 20
                    
                cv2.putText(viz, label, (x1, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw ROI bounds for debugging
        # if self.params.get('show_roi', False):
        #     xmin, ymin, xmax, ymax = roi_bounds
        #     cv2.rectangle(viz, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)

        return viz, obstacles