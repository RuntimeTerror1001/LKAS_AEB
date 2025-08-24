#!/usr/bin/env python3
"""
Rear Camera Processor (refactored, framework-consistent)

- Subclasses BaseCameraProcessor pattern (via BasePerceptionModule)
- **No dependency on ObstacleDetector** (YOLO) — safe to delete that module
- Optional naive motion-based detection to produce coarse bboxes
- Converts detections to ObstacleArray with rear-facing geometry
"""
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from lkas_aeb_msgs.msg import ObstacleArray, Obstacle

from .base_classes import BasePerceptionModule

_bridge = CvBridge()

class RearCameraProcessor(BasePerceptionModule):
    def __init__(self, params: Dict):
        self.processing_params = self._validate_params(params or {})
        # lazy init background subtractor if enabled
        self._bg = None
        if self.processing_params['detection']['enable_naive_motion_detector']:
            self._bg = cv2.createBackgroundSubtractorMOG2(
                history=int(self.processing_params['detection']['bg_history']),
                varThreshold=float(self.processing_params['detection']['bg_var_thresh']),
                detectShadows=False
            )
        super().__init__(self.processing_params, name="RearCameraProcessor")

    def _initialize(self) -> None:
        d = self.processing_params['detection']
        self._flip = bool(d.get('flip_rear_image', True))
        self._min_area = int(d.get('min_bbox_area', 1200))
        self._focal = float(d.get('rear_focal_length', 800.0))
        self._cam_h = float(d.get('rear_camera_height', 1.5))

    def process(self, msg: Image) -> ObstacleArray:
        start = self._last_update_time
        out = ObstacleArray()
        out.header = msg.header
        out.header.frame_id = "base_link"
        try:
            img = _bridge.imgmsg_to_cv2(msg, 'bgr8')
            if self._flip:
                img = cv2.flip(img, 1)

            detections = self._detect(img)  # list of (x1,y1,x2,y2)
            obstacles = self._detections_to_obstacles(detections, img.shape, msg.header)
            out.obstacles = obstacles
            self._update_stats(start, input_count=len(detections), output_count=len(obstacles))
            return out
        except Exception:
            self._update_stats(start, input_count=0, output_count=0, had_error=True)
            raise

    # ----------------------------- helpers --------------------------------
    def _validate_params(self, params: Dict) -> Dict:
        v = {'detection': {}}
        det = params.get('detection', {})
        v['detection'] = {
            'enable_naive_motion_detector': bool(det.get('enable_naive_motion_detector', True)),
            'bg_history': int(det.get('bg_history', 100)),
            'bg_var_thresh': float(det.get('bg_var_thresh', 25.0)),
            'min_bbox_area': int(det.get('min_bbox_area', 1200)),
            'flip_rear_image': bool(det.get('flip_rear_image', True)),
            'rear_focal_length': float(det.get('rear_focal_length', 800.0)),
            'rear_camera_height': float(det.get('rear_camera_height', 1.5)),
        }
        return v

    def _detect(self, img) -> List[Tuple[int,int,int,int]]:
        """Very simple motion-based ROI detection. Returns list of bboxes."""
        if self._bg is None:
            return []  # detector disabled → empty result
        fg = self._bg.apply(img)
        fg = cv2.medianBlur(fg, 5)
        _, th = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
        th = cv2.dilate(th, np.ones((5,5), np.uint8), iterations=1)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: List[Tuple[int,int,int,int]] = []
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            if w*h < self._min_area:
                continue
            boxes.append((x,y,x+w,y+h))
        return boxes

    def _bbox_to_world(self, bbox, image_shape) -> Optional[Tuple[float, float, float]]:
        # Use flat-ground and optical center offset (rear-facing camera).
        x1,y1,x2,y2 = bbox
        h,w = image_shape[:2]
        cx = w/2.0; cy = h/2.0
        u = ((x1 + x2) * 0.5) - cx
        v = y2 - cy  # bottom contact point relative to center
        if v <= 0:
            return None
        distance = (self._focal * self._cam_h) / v
        lateral = (u * distance) / self._focal
        # rear camera looks backward → -x in base_link
        return (-float(distance), float(lateral), 0.0)

    def _detections_to_obstacles(self, detections: List[Tuple[int,int,int,int]], image_shape, header) -> List[Obstacle]:
        obs: List[Obstacle] = []
        for i,box in enumerate(detections):
            world = self._bbox_to_world(box, image_shape)
            o = Obstacle()
            o.bbox = [int(v) for v in box]
            o.sensor_type = "camera_rear"
            o.track_id = i+1
            o.class_id = -1
            o.confidence = 0.4  # naive detector → modest confidence
            if world is not None:
                o.position_3d = [float(world[0]), float(world[1]), float(world[2])]
                o.distance = float(np.hypot(world[0], world[1]))
            else:
                o.position_3d = [0.0,0.0,0.0]
                o.distance = -1.0
            obs.append(o)
        return obs
