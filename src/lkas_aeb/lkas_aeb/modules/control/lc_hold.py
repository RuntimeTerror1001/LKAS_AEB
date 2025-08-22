#!/usr/bin/env python3
"""
Lane Change Hold Module

Determines when to hold (cut throttle) during lane changes based on rear obstacles.
"""

from typing import Dict, Literal, List
import math
from lkas_aeb_msgs.msg import ObstacleArray, Obstacle

Side = Literal["left", "right", "none"]

def point_in_polygon(x: float, y: float, polygon: Dict) -> bool:
    """
    Check if a point is inside a rectangular polygon defined by min/max bounds.
    
    Args:
        x, y: Point coordinates
        polygon: Dictionary with keys 'x_min', 'x_max', 'y_min', 'y_max'
        
    Returns:
        True if point is inside polygon
    """
    return (polygon['x_min'] <= x <= polygon['x_max'] and
            polygon['y_min'] <= y <= polygon['y_max'])

def calculate_ttc(longitudinal_distance: float, relative_velocity: float) -> float:
    """
    Calculate time-to-collision for longitudinal approach.
    
    Args:
        longitudinal_distance: Distance along x-axis (positive = ahead, negative = behind)
        relative_velocity: Relative velocity (positive = closing/approaching)
        
    Returns:
        Time to collision in seconds (infinity if not approaching)
    """
    if relative_velocity <= 0:
        return float('inf')  # Not approaching
    
    if longitudinal_distance > 0:
        # Vehicle ahead approaching us
        return longitudinal_distance / relative_velocity
    else:
        # Vehicle behind approaching us  
        return abs(longitudinal_distance) / relative_velocity

def get_lane_polygon(side: Side, params: Dict) -> Dict:
    """
    Get the polygon defining the target lane area.
    
    Args:
        side: Target lane side ("left", "right", or "none")
        params: Configuration parameters
        
    Returns:
        Polygon dictionary or None if side is "none"
    """
    if side == "none":
        return None
    elif side == "left":
        return params.get('left_poly', {
            'x_min': -12.0,
            'x_max': 2.0,
            'y_min': -3.8,
            'y_max': -2.2
        })
    elif side == "right":
        return params.get('right_poly', {
            'x_min': -12.0,
            'x_max': 2.0,
            'y_min': 2.2,
            'y_max': 3.8
        })
    else:
        return None

def analyze_obstacle_threat(obstacle: Obstacle, target_polygon: Dict, params: Dict, ego_speed: float) -> Dict:
    """
    Analyze if an obstacle poses a threat for lane change.
    
    Args:
        obstacle: Detected obstacle
        target_polygon: Lane polygon to check
        params: Configuration parameters
        ego_speed: Ego vehicle speed
        
    Returns:
        Dictionary with threat analysis results
    """
    result = {
        'is_threat': False,
        'reason': 'safe',
        'distance': 0.0,
        'ttc': float('inf'),
        'in_blind_spot': False
    }
    
    # Get obstacle position
    if obstacle.position_3d and (obstacle.position_3d[0] != 0 or obstacle.position_3d[1] != 0):
        x, y = obstacle.position_3d[0], obstacle.position_3d[1]
    else:
        # Fallback: use distance and assume it's directly behind
        if obstacle.distance > 0:
            x = -obstacle.distance  # Behind vehicle
            y = 0.0
        else:
            return result  # Can't determine position
    
    # Check if obstacle is in target lane area
    if not point_in_polygon(x, y, target_polygon):
        result['reason'] = 'not_in_target_lane'
        return result
    
    # Get safety parameters
    D_front_min = params.get('D_front_min', 30.0)
    D_rear_min = params.get('D_rear_min', 25.0)
    TTC_min = params.get('TTC_min', 5.0)
    near_side_window = params.get('near_side_window', 8.0)
    
    # Calculate longitudinal distance (x-axis)
    d_rel = x  # Positive = ahead, negative = behind
    result['distance'] = d_rel
    
    # Calculate relative velocity
    # obstacle.relative_speed is typically closing speed (positive when approaching)
    v_rel = obstacle.relative_speed
    if v_rel == 0 and obstacle.speed > 0:
        # Estimate relative speed if not provided
        v_rel = max(0, obstacle.speed - ego_speed)
    
    # Safety checks
    threats = []
    
    # 1. Gap check: is vehicle too close?
    if -D_rear_min <= d_rel <= D_front_min:
        threats.append("gap_too_small")
        result['is_threat'] = True
    
    # 2. TTC check: is approaching vehicle going to reach us too soon?
    if d_rel > 0 and v_rel > 0:  # Vehicle ahead approaching us
        ttc = d_rel / max(v_rel, 1e-2)
        result['ttc'] = ttc
        if ttc < TTC_min:
            threats.append("ttc_too_low")
            result['is_threat'] = True
    
    # 3. Alongside check: is vehicle currently beside us?
    if abs(d_rel) < near_side_window:
        threats.append("vehicle_alongside")
        result['is_threat'] = True
        result['in_blind_spot'] = True
    
    # 4. Fast approaching from behind check
    if d_rel < 0 and v_rel > 0:  # Vehicle behind approaching
        ttc = abs(d_rel) / max(v_rel, 1e-2)
        result['ttc'] = ttc
        if ttc < TTC_min:
            threats.append("fast_approach_from_rear")
            result['is_threat'] = True
    
    # Set reason
    if threats:
        result['reason'] = ", ".join(threats)
    
    return result

def filter_relevant_obstacles(obstacles: List[Obstacle], params: Dict) -> List[Obstacle]:
    """
    Filter obstacles to those relevant for lane change decisions.
    
    Args:
        obstacles: List of detected obstacles
        params: Configuration parameters
        
    Returns:
        Filtered list of relevant obstacles
    """
    relevant = []
    
    # Filter parameters
    max_distance = params.get('lc_max_distance', 80.0)
    min_confidence = params.get('lc_min_confidence', 0.3)
    relevant_classes = params.get('lc_relevant_classes', [1, 2, 3])  # cars, trucks, bikes
    
    for obstacle in obstacles:
        # Distance filter
        if obstacle.distance > max_distance:
            continue
        
        # Confidence filter
        if obstacle.confidence < min_confidence:
            continue
        
        # Class filter (focus on vehicles)
        if obstacle.class_id not in relevant_classes:
            continue
        
        # Position filter (must be roughly to the rear or side)
        if obstacle.position_3d and obstacle.position_3d[0] > 10.0:
            continue  # Too far ahead
        
        relevant.append(obstacle)
    
    return relevant

def should_hold_for_lane_change(fused_rear: ObstacleArray,
                                target_side: Side,
                                ego_speed: float,
                                lane_width: float,
                                params: Dict) -> Dict:
    """
    Decide whether to HOLD (cut throttle to 0) while Pure Pursuit attempts a lane change.
    
    Args:
        fused_rear: Fused obstacle array from rear sensors
        target_side: Target lane ("left", "right", or "none")
        ego_speed: Current ego vehicle speed in m/s
        lane_width: Lane width in meters
        params: Configuration parameters
        
    Returns:
        Dictionary with decision and details:
        - hold: bool - Whether to hold throttle
        - offender_id: int - Track ID of threatening obstacle (-1 if none)
        - reason: str - Human-readable reason for decision
        - threat_count: int - Number of threatening obstacles
        - closest_threat: float - Distance to closest threat
    """
    
    result = {
        'hold': False,
        'offender_id': -1,
        'reason': 'no lane change',
        'threat_count': 0,
        'closest_threat': float('inf')
    }
    
    # No lane change requested
    if target_side == "none":
        return result
    
    # Get target lane polygon
    target_polygon = get_lane_polygon(target_side, params)
    if target_polygon is None:
        result['reason'] = 'invalid target side'
        return result
    
    # Filter relevant obstacles
    relevant_obstacles = filter_relevant_obstacles(fused_rear.obstacles, params)
    
    if not relevant_obstacles:
        result['reason'] = 'no relevant obstacles'
        return result
    
    # Analyze each obstacle for threats
    threats = []
    closest_distance = float('inf')
    most_dangerous_id = -1
    
    for obstacle in relevant_obstacles:
        threat_analysis = analyze_obstacle_threat(obstacle, target_polygon, params, ego_speed)
        
        if threat_analysis['is_threat']:
            threats.append({
                'obstacle': obstacle,
                'analysis': threat_analysis
            })
            
            # Track closest threat
            threat_distance = abs(threat_analysis['distance'])
            if threat_distance < closest_distance:
                closest_distance = threat_distance
                most_dangerous_id = obstacle.track_id
    
    # Make final decision
    result['threat_count'] = len(threats)
    result['closest_threat'] = closest_distance if threats else float('inf')
    
    if threats:
        result['hold'] = True
        result['offender_id'] = most_dangerous_id
        
        # Create detailed reason
        reasons = []
        for threat in threats:
            obs = threat['obstacle']
            analysis = threat['analysis']
            distance = analysis['distance']
            reason = analysis['reason']
            
            if analysis['in_blind_spot']:
                reasons.append(f"T{obs.track_id} alongside ({distance:.1f}m)")
            elif distance > 0:
                reasons.append(f"T{obs.track_id} ahead ({distance:.1f}m): {reason}")
            else:
                reasons.append(f"T{obs.track_id} behind ({abs(distance):.1f}m): {reason}")
        
        result['reason'] = f"UNSAFE - {'; '.join(reasons[:3])}"  # Limit to 3 threats
        
        # Add urgency indicator for very close threats
        if closest_distance < params.get('critical_distance', 5.0):
            result['reason'] = "CRITICAL - " + result['reason']
    else:
        result['hold'] = False
        result['reason'] = f'safe for {target_side} lane change'
    
    # Override for very low speeds (parking lot scenarios)
    if ego_speed < params.get('min_speed_for_lc_hold', 2.0):
        result['hold'] = False
        result['reason'] += ' (low speed override)'
    
    # Override for emergency situations (if explicitly requested)
    if params.get('emergency_lane_change', False):
        result['hold'] = False
        result['reason'] += ' (emergency override)'
    
    return result