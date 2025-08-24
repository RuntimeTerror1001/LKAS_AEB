#!/usr/bin/env python3
"""
Rear Radar Processor (refactored, framework-consistent)

- Subclasses BasePerceptionModule for consistent stats/params handling
- Pure NumPy + sensor_msgs_py; no PCL/Open3D dependencies
- Left+Right rear radar inputs -> cluster -> lightweight tracking -> ObstacleArray
- Parameters validated via ParameterValidator-like pattern
"""
from typing import Dict, List, Tuple, Optional
import math
import numpy as np

from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from lkas_aeb_msgs.msg import Obstacle, ObstacleArray

from .base_classes import BasePerceptionModule, ParameterValidator
from lkas_aeb.util.perception_utils import validate_numeric


class _RadarTracker:
    """Minimal persistent-ID tracker for clustered radar hits."""
    def __init__(self, max_age:int=10, max_assoc_dist:float=5.0) -> None:
        self.tracks: Dict[int, Dict] = {}
        self.next_id = 0
        self.frame = 0
        self.max_age = max_age
        self.max_assoc_dist = max_assoc_dist

    def update(self, clusters: List[Tuple[float,float,float,float,int]]) -> List[Tuple[float,float,float,float,int,int]]:
        """Associate (x,y,vel,conf,point_count) -> (x,y,vel,conf,track_id,point_count)."""
        self.frame += 1
        # prune
        stale = [tid for tid,t in self.tracks.items() if (self.frame - t['last_seen']) > self.max_age]
        for tid in stale:
            del self.tracks[tid]

        out = []
        used: set[int] = set()
        for x,y,vel,conf,pcount in clusters:
            best_id, best_d = None, 1e9
            for tid,t in self.tracks.items():
                if tid in used:
                    continue
                tx,ty = t['pos']
                d = math.hypot(x-tx, y-ty)
                if d < best_d and d < self.max_assoc_dist:
                    best_id, best_d = tid, d
            if best_id is not None:
                tr = self.tracks[best_id]
                tr['pos'] = (x,y)
                tr['age'] += 1
                tr['last_seen'] = self.frame
                used.add(best_id)
                out.append((x,y,vel,conf,best_id,pcount))
            else:
                tid = self.next_id; self.next_id += 1
                self.tracks[tid] = {'pos':(x,y), 'age':1, 'last_seen':self.frame}
                out.append((x,y,vel,conf,tid,pcount))
        return out


class RearRadarProcessor(BasePerceptionModule):
    """Rear radar preprocessor that consumes LEFT+RIGHT PointCloud2 messages."""
    def __init__(self, params: Dict):
        # Validate & normalize params first
        self.processing_params = self._validate_params(params or {})
        self._tracker = _RadarTracker(
            max_age=int(self.processing_params['tracking']['max_age']),
            max_assoc_dist=float(self.processing_params['tracking']['max_association_distance'])
        )
        super().__init__(self.processing_params, name="RearRadarProcessor")

    # ------------------------------ lifecycle ------------------------------
    def _initialize(self) -> None:
        p = self.processing_params
        self._roi = p['roi']
        self._gating = p['gating']
        self._cluster = p['clustering']
        # Nothing heavy to init – pure NumPy pipeline

    def get_stats(self) -> Dict:
        return super().get_stats()

    # ------------------------------- API ----------------------------------
    def process(self, left_msg: PointCloud2, right_msg: PointCloud2) -> ObstacleArray:
        start = self._last_update_time
        out = ObstacleArray()
        # choose freshest header
        if left_msg and right_msg:
            if (left_msg.header.stamp.sec, left_msg.header.stamp.nanosec) >= (right_msg.header.stamp.sec, right_msg.header.stamp.nanosec):
                out.header = left_msg.header
            else:
                out.header = right_msg.header
        elif left_msg:
            out.header = left_msg.header
        elif right_msg:
            out.header = right_msg.header
        out.header.frame_id = "base_link"

        try:
            all_pts = []
            if left_msg and left_msg.width and left_msg.height:
                lp = self._read_points(left_msg)
                if lp.size: 
                    lp = self._gate_points(lp)
                    if lp.size:
                        lp = self._to_base_link(lp, sensor_side='left')
                        all_pts.append(lp)
            if right_msg and right_msg.width and right_msg.height:
                rp = self._read_points(right_msg)
                if rp.size:
                    rp = self._gate_points(rp)
                    if rp.size:
                        rp = self._to_base_link(rp, sensor_side='right')
                        all_pts.append(rp)

            if not all_pts:
                self._update_stats(start, input_count=0, output_count=0)
                return out

            pts = np.vstack(all_pts)  # (N,4) -> x,y,z,vel in base_link
            # ROI in base_link
            roi = self._apply_roi(pts)
            if roi.size == 0:
                self._update_stats(start, input_count=len(pts), output_count=0)
                return out

            # cluster in XY using grid-accelerated DBSCAN-like pass
            clusters = self._cluster_points(roi)
            if not clusters:
                self._update_stats(start, input_count=len(roi), output_count=0)
                return out

            tracked = self._tracker.update(clusters)
            for x,y,vel,conf,tid,pcount in tracked:
                o = Obstacle()
                o.sensor_type = "radar_rear"
                o.sensor_sources = ["radar_rear_left","radar_rear_right"]
                o.track_id = int(tid)
                o.class_id = 0
                o.distance = float(math.hypot(x,y))
                o.relative_speed = float(abs(vel))
                o.speed = 0.0
                o.position_3d = [float(x), float(y), 0.0]
                o.size_3d = [0.0,0.0,0.0]
                o.bbox = [-1,-1,-1,-1]
                o.point_count = int(pcount)
                o.confidence = float(conf)
                out.obstacles.append(o)

            self._update_stats(start, input_count=len(roi), output_count=len(out.obstacles))
            return out
        except Exception:
            # mark error and bubble up
            self._update_stats(start, input_count=0, output_count=0, had_error=True)
            raise

    # ----------------------------- helpers --------------------------------
    def _validate_params(self, params: Dict) -> Dict:
        # base shell
        v = {
            'roi': {},
            'gating': {},
            'clustering': {},
            'tracking': {}
        }
        # ROI (base_link coordinates; x forward, rear is negative x)
        roi = params.get('roi', {})
        v['roi'] = {
            'x_min': validate_numeric(roi.get('x_min', -60.0), -60.0),
            'x_max': validate_numeric(roi.get('x_max',  10.0),  10.0),
            'y_min': validate_numeric(roi.get('y_min',  -8.0),  -8.0),
            'y_max': validate_numeric(roi.get('y_max',   8.0),   8.0),
        }
        # sensor gating
        gat = params.get('gating', {})
        v['gating'] = {
            'min_range': validate_numeric(gat.get('min_range', 1.0), 1.0),
            'max_range': validate_numeric(gat.get('max_range', 100.0), 100.0),
            'min_elev_deg': validate_numeric(gat.get('min_elev_deg', -20.0), -20.0),
            'max_elev_deg': validate_numeric(gat.get('max_elev_deg',  20.0),  20.0),
            'min_abs_velocity': validate_numeric(gat.get('min_abs_velocity', 0.5), 0.5)
        }
        # clustering
        clu = params.get('clustering', {})
        v['clustering'] = {
            'eps': validate_numeric(clu.get('eps', 2.0), 2.0),
            'min_samples': int(clu.get('min_samples', 2))
        }
        # tracker
        tr = params.get('tracking', {})
        v['tracking'] = {
            'max_age': int(tr.get('max_age', 10)),
            'max_association_distance': validate_numeric(tr.get('max_association_distance', 5.0), 5.0)
        }
        return v

    def _read_points(self, msg: PointCloud2) -> np.ndarray:
        pts = pc2.read_points(msg, skip_nans=True)
        buf = []
        for p in pts:
            if len(p) >= 3:
                x,y,z = float(p[0]), float(p[1]), float(p[2])
                vel = float(p[3]) if len(p) > 3 else 0.0
                buf.append((x,y,z,vel))
        if not buf:
            return np.empty((0,4), dtype=np.float32)
        return np.asarray(buf, dtype=np.float32)

    def _gate_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        rng = np.linalg.norm(points[:,:3], axis=1)
        rmask = (rng >= self._gating['min_range']) & (rng <= self._gating['max_range'])
        elev = np.arcsin(points[:,2] / np.maximum(rng, 1e-6))
        emask = (elev >= math.radians(self._gating['min_elev_deg'])) & (elev <= math.radians(self._gating['max_elev_deg']))
        vmask = np.abs(points[:,3]) >= self._gating['min_abs_velocity']
        mask = rmask & emask & vmask
        return points[mask]

    def _to_base_link(self, pts: np.ndarray, sensor_side: str) -> np.ndarray:
        # Approximate mounting transform (left/right rear at ~±135° yaw), also translate a bit to bumper
        ang = math.radians(135.0 if sensor_side=='left' else -135.0)
        ca, sa = math.cos(ang), math.sin(ang)
        x = pts[:,0]*ca - pts[:,1]*sa
        y = pts[:,0]*sa + pts[:,1]*ca
        out = pts.copy()
        out[:,0], out[:,1] = x - 2.0, y + (-1.0 if sensor_side=='left' else 1.0)
        return out

    def _apply_roi(self, pts: np.ndarray) -> np.ndarray:
        x_min, x_max = self._roi['x_min'], self._roi['x_max']
        y_min, y_max = self._roi['y_min'], self._roi['y_max']
        m = (pts[:,0] >= x_min) & (pts[:,0] <= x_max) & (pts[:,1] >= y_min) & (pts[:,1] <= y_max)
        return pts[m]

    def _cluster_points(self, pts: np.ndarray) -> List[Tuple[float,float,float,float,int]]:
        if pts.size == 0:
            return []
        eps = float(self._cluster['eps'])
        min_samples = int(self._cluster['min_samples'])
        xy = pts[:,:2]
        vel = pts[:,3]
        # grid-accelerated neighborhood search
        cell = max(eps, 1e-3)
        inv = 1.0 / cell
        ij = np.floor(xy * inv).astype(np.int32)
        from collections import defaultdict, deque
        bucket = defaultdict(list)
        for idx,(i,j) in enumerate(ij):
            bucket[(int(i),int(j))].append(idx)
        visited_cells = set()
        visited_points = np.zeros(xy.shape[0], dtype=bool)
        clusters_idx: List[np.ndarray] = []
        nbrs = [(dx,dy) for dx in (-1,0,1) for dy in (-1,0,1) if not (dx==0 and dy==0)]
        for key in list(bucket.keys()):
            if key in visited_cells:
                continue
            q = deque([key]); visited_cells.add(key)
            candidate = []
            while q:
                ci,cj = q.popleft()
                candidate.extend(bucket[(ci,cj)])
                for dx,dy in nbrs:
                    nk = (ci+dx, cj+dy)
                    if nk in bucket and nk not in visited_cells:
                        visited_cells.add(nk)
                        q.append(nk)
            if not candidate:
                continue
            cand = np.array(candidate, dtype=np.int32)
            pts_local = xy[cand]
            used = np.zeros(len(cand), dtype=bool)
            for i in range(len(cand)):
                if used[i]:
                    continue
                d = np.linalg.norm(pts_local - pts_local[i], axis=1)
                neigh = np.where(d <= eps)[0]
                if len(neigh) >= min_samples:
                    used[neigh] = True
                    clusters_idx.append(cand[neigh])
        # summarize
        out: List[Tuple[float,float,float,float,int]] = []
        for idxs in clusters_idx:
            cp = pts[idxs]
            cx = float(cp[:,0].mean())
            cy = float(cp[:,1].mean())
            vmean = float(cp[:,3].mean())
            vstd = float(cp[:,3].std()) if cp.shape[0] > 1 else 0.0
            compact = 1.0 / (1.0 + vstd/5.0)
            conf = float(min(1.0, cp.shape[0]/12.0) * max(0.1, compact))
            out.append((cx, cy, vmean, conf, int(cp.shape[0])))
        return out
