import cv2 
import numpy as np

# ========================
# LANE DETECTOR
# ========================

class LaneDetector:
    """
    Advanced lane detection system using computer vision techniques.
    Implements perspective transformation, color masking, edge detection,
    and polynomial fitting with temporal smoothing for robust lane tracking.
    """

    def __init__(self, params):
        """
        Initialize lane detector with configuration parameters.
        
        Args:
            params (dict): Configuration dictionary containing lane_detection parameters
        
        Returns:
            None
        """
        self.params = params['lane_detection']

        # ========================
        # TEMPORAL SMOOTHING VARIABLES
        # ========================
        
        self.lane_history = []           # History of lane fits for smoothing
        self.max_history = 5            # Maximum frames to keep in history
        self.prev_left_fit = None       # Previous left lane polynomial fit
        self.prev_right_fit = None      # Previous right lane polynomial fit

        # ========================
        # VALIDATION PARAMETERS
        # ========================
        
        self.min_lane_width_pixels = 200   # Minimum expected lane width (pixels)
        self.max_lane_width_pixels = 800   # Maximum expected lane width (pixels)
        self.max_curvature_change = 0.005  # Maximum change in curvature per frame
    
    def validate_lane_fit(self, fit, prev_fit):
        """
        Validate polynomial fit for lane detection quality.
        
        Args:
            fit (np.array): Current polynomial coefficients [a, b, c] for ax²+bx+c
            prev_fit (np.array): Previous polynomial coefficients for comparison
        
        Returns:
            bool: True if fit is valid, False otherwise
        """
        if fit is None:
            return False

        # Check if curvature coefficient is reasonable (not too sharp)
        if abs(fit[0]) > 0.01:  # Curvature threshold
            return False
        
        # Check for smooth transition from previous frame
        if prev_fit is not None:
            if abs(fit[0] - prev_fit[0]) > self.max_curvature_change:
                return False
        
        return True
    
    def smooth_lanes(self, left_fit, right_fit):
        """
        Apply temporal smoothing to lane polynomial fits for stability.
        
        Args:
            left_fit (np.array): Current left lane polynomial coefficients
            right_fit (np.array): Current right lane polynomial coefficients
        
        Returns:
            tuple: (smoothed_left_fit, smoothed_right_fit)
        """
        # Validate current fits against previous ones
        left_valid = self.validate_lane_fit(left_fit, self.prev_left_fit)
        right_valid = self.validate_lane_fit(right_fit, self.prev_right_fit)

        # Use previous fits if current ones are invalid
        if not left_valid and self.prev_left_fit is not None:
            left_fit = self.prev_left_fit
        if not right_valid and self.prev_right_fit is not None:
            right_fit = self.prev_right_fit

        # Update history buffer
        if len(self.lane_history) >= self.max_history: 
            self.lane_history.pop(0)
        self.lane_history.append((left_fit, right_fit))

        # Calculate running average if we have sufficient history
        if len(self.lane_history) > 1:
            left_fits = [fit[0] for fit in self.lane_history if fit[0] is not None]
            right_fits = [fit[1] for fit in self.lane_history if fit[1] is not None]

            if left_fits:
                avg_left = np.mean(left_fits, axis=0)
            else:
                avg_left = left_fit

            if right_fits:
                avg_right = np.mean(right_fits, axis=0)
            else:
                avg_right = right_fit
            
            self.prev_left_fit = avg_left
            self.prev_right_fit = avg_right
            return avg_left, avg_right
        
        # Store current fits for next iteration
        self.prev_left_fit = left_fit
        self.prev_right_fit = right_fit
        return left_fit, right_fit
    
    def create_roi_mask(self, image):
        """
        Create region of interest mask to focus on road area.
        
        Args:
            image (np.array): Input image
        
        Returns:
            np.array: Binary mask highlighting road region
        """
        height, width = image.shape[:2]
        mask = np.zeros_like(image)

        # Define trapezoidal ROI focusing on road ahead
        roi_vertices = np.array([[
            (width * 0.1, height),      # Bottom left
            (width * 0.4, height * 0.6), # Top left
            (width * 0.6, height * 0.6), # Top right
            (width * 0.9, height)       # Bottom right
        ]], dtype=np.int32)

        cv2.fillPoly(mask, roi_vertices, 255)
        return mask

    def detect_lanes(self, image):
        """"
        Main lane detection pipeline combining perspective transform, color filtering,
        edge detection, line detection, and polynomial fitting.
        
        Args:
            image (np.array): Input BGR image from camera
        
        Returns:
            tuple: (visualization_image, lane_center, left_curvature, right_curvature, 
                   lane_width, confidence, distance)
        """
        height, width = image.shape[:2]
        
        # ========================
        # PERSPECTIVE TRANSFORMATION
        # ========================
        
        # Define source and destination points for bird's eye view
        src = np.float32([
            [width * 0.30, height * 0.70],  # Top left
            [width * 0.70, height * 0.70],  # Top right
            [width * 0.90, height * 0.95],  # Bottom right
            [width * 0.10, height * 0.95]   # Bottom left
        ])
        dst = np.float32([
            [width * 0.2, 0],               # Top left
            [width * 0.8, 0],               # Top right
            [width * 0.8, height],          # Bottom right
            [width * 0.2, height]           # Bottom left
        ])
        
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(image, M, (width, height))

        # ========================
        # COLOR-BASED LANE DETECTION
        # ========================
        
        # Convert to HSV for better color filtering
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        
        # Create masks for white and yellow lane markings
        white_mask = cv2.inRange(hsv, np.array(self.params['hsv_white_lower']), 
                                np.array(self.params['hsv_white_upper']))
        yellow_mask = cv2.inRange(hsv, np.array(self.params['hsv_yellow_lower']), 
                                 np.array(self.params['hsv_yellow_upper']))
        combined = cv2.bitwise_or(white_mask, yellow_mask)

        # Apply region of interest mask
        roi_mask = self.create_roi_mask(combined)
        combined = cv2.bitwise_and(combined, roi_mask)

        # ========================
        # EDGE DETECTION
        # ========================
        
        edges = cv2.Canny(combined, self.params['canny_threshold1'], 
                         self.params['canny_threshold2'])

        # ========================
        # LINE DETECTION
        # ========================
        
        # Hough transform to detect line segments
        lines = cv2.HoughLinesP(
            edges,
            rho=2,                    # Distance resolution
            theta=np.pi/180,          # Angular resolution
            threshold=30,             # Minimum vote threshold
            minLineLength=30,         # Minimum line length
            maxLineGap=150           # Maximum gap between line segments
        )

        # ========================
        # INITIALIZE VARIABLES
        # ========================
        
        lane_center = None
        left_curverad = None
        right_curverad = None
        lane_width = 3.7  # Default lane width in meters
        viz = warped.copy()
        left_lines, right_lines = [], []
        left_fit, right_fit = None, None
        
        # ========================
        # LINE CLASSIFICATION AND FITTING
        # ========================
        
        if lines is not None:
            # Classify lines as left or right lanes based on slope and position
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate slope with division by zero protection
                if abs(x2 - x1) < 1e-5:
                    slope = float('inf') if y2 > y1 else float('-inf')
                else:
                    slope = (y2 - y1) / (x2 - x1)
                    
                # Classify based on slope and position
                if slope < -0.2 and x1 < width/2 and x2 < width/2:
                    left_lines.append(line)
                elif slope > 0.2 and x1 > width/2 and x2 > width/2:
                    right_lines.append(line)

            # Collect points for polynomial fitting
            left_pts, right_pts = [], []
            for line in left_lines:
                x1, y1, x2, y2 = line[0]
                left_pts.extend([[x1, y1], [x2, y2]])
            for line in right_lines:
                x1, y1, x2, y2 = line[0]
                right_pts.extend([[x1, y1], [x2, y2]])
            
            # Fit second-order polynomials if sufficient points available
            if len(left_pts) > 10:
                left_pts = np.array(left_pts)
                try:
                    left_fit = np.polyfit(left_pts[:,1], left_pts[:,0], 2)
                    # Constrain curvature to reasonable values
                    left_fit[0] = np.clip(left_fit[0], -0.01, 0.01)
                except np.RankWarning:
                    left_fit = None
                    
            if len(right_pts) > 10:
                right_pts = np.array(right_pts)
                try:
                    right_fit = np.polyfit(right_pts[:,1], right_pts[:,0], 2)
                    # Constrain curvature to reasonable values
                    right_fit[0] = np.clip(right_fit[0], -0.01, 0.01)
                except np.RankWarning:
                    right_fit = None

            # Apply temporal smoothing to polynomial fits
            left_fit, right_fit = self.smooth_lanes(left_fit, right_fit)
            
            # ========================
            # LANE WIDTH VALIDATION
            # ========================
            
            if left_fit is not None and right_fit is not None:
                y_bottom = height * 0.9
                left_x = left_fit[0]*y_bottom**2 + left_fit[1]*y_bottom + left_fit[2]
                right_x = right_fit[0]*y_bottom**2 + right_fit[1]*y_bottom + right_fit[2]
                lane_width_pixels = abs(right_x - left_x)

                # Validate and correct unreasonable lane widths
                if (lane_width_pixels < self.min_lane_width_pixels or 
                    lane_width_pixels > self.max_lane_width_pixels):
                    # Estimate missing lane based on stronger detection
                    if len(left_pts) > len(right_pts):
                        right_fit = left_fit.copy()
                        right_fit[2] = left_fit[2] + width * 0.4
                    else:
                        left_fit = right_fit.copy()
                        left_fit[2] = right_fit[2] - width * 0.4
            
            # Handle single lane detection cases
            elif left_fit is not None and right_fit is None:
                # Estimate right lane from left lane
                right_fit = left_fit.copy()
                right_fit[2] = left_fit[2] + width * 0.4
                
            elif right_fit is not None and left_fit is None:
                # Estimate left lane from right lane
                left_fit = right_fit.copy()
                left_fit[2] = right_fit[2] - width * 0.4
                
            # ========================
            # LANE METRICS CALCULATION
            # ========================
            
            if left_fit is not None and right_fit is not None:
                # Calculate lane center at bottom of image
                y_bottom = height
                left_x = left_fit[0]*y_bottom**2 + left_fit[1]*y_bottom + left_fit[2]
                right_x = right_fit[0]*y_bottom**2 + right_fit[1]*y_bottom + right_fit[2]
                lane_center = (int((left_x + right_x)/2), int(y_bottom))
                
                # Calculate lane width in pixels
                lane_width_pixels = abs(right_x - left_x)
                
                # ========================
                # REAL-WORLD COORDINATE CONVERSION
                # ========================
                
                if lane_width_pixels > 0:
                    # Conversion factors from pixels to meters
                    xm_per_pix = 3.7 / lane_width_pixels  # Standard lane width
                    ym_per_pix = 30 / height              # 30m view ahead
                    
                    # Convert polynomial coefficients to real-world coordinates
                    left_fit_cr = np.array([
                        left_fit[0] * xm_per_pix / (ym_per_pix**2),  # Curvature term
                        left_fit[1] * xm_per_pix / ym_per_pix,       # Linear term
                        left_fit[2] * xm_per_pix                     # Constant term
                    ])
                    
                    right_fit_cr = np.array([
                        right_fit[0] * xm_per_pix / (ym_per_pix**2), # Curvature term
                        right_fit[1] * xm_per_pix / ym_per_pix,      # Linear term
                        right_fit[2] * xm_per_pix                    # Constant term
                    ])
                    
                    # ========================
                    # CURVATURE CALCULATION
                    # ========================
                    
                    # Calculate curvature at point 20m ahead
                    y_eval = min(height, height - 20/ym_per_pix) * ym_per_pix
                    
                    # Radius of curvature formula: R = (1 + (dy/dx)²)^1.5 / |d²y/dx²|
                    if abs(left_fit_cr[0]) > 1e-6:  # Avoid division by zero
                        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval + left_fit_cr[1])**2)**1.5) / abs(2*left_fit_cr[0])
                    else:
                        left_curverad = 10000  # Very large radius (straight line)
                        
                    if abs(right_fit_cr[0]) > 1e-6:  # Avoid division by zero
                        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval + right_fit_cr[1])**2)**1.5) / abs(2*right_fit_cr[0])
                    else:
                        right_curverad = 10000  # Very large radius (straight line)
                    
                    # Calculate actual lane width in meters
                    lane_width = lane_width_pixels * xm_per_pix

        # ========================
        # VISUALIZATION
        # ========================
        
        # Generate points for smooth curve visualization
        ploty = np.linspace(0, height-1, height)
        
        # Draw left lane if detected
        if left_fit is not None:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            # Only draw points within image bounds
            valid_indices = (left_fitx >= 0) & (left_fitx < width)
            for i in range(1, len(ploty)):
                if valid_indices[i-1] and valid_indices[i]:
                    cv2.line(viz, (int(left_fitx[i-1]), int(ploty[i-1])), 
                            (int(left_fitx[i]), int(ploty[i])), (0,255,0), 3)
        
        # Draw right lane if detected
        if right_fit is not None:
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            # Only draw points within image bounds
            valid_indices = (right_fitx >= 0) & (right_fitx < width)
            for i in range(1, len(ploty)):
                if valid_indices[i-1] and valid_indices[i]:
                    cv2.line(viz, (int(right_fitx[i-1]), int(ploty[i-1])), 
                            (int(right_fitx[i]), int(ploty[i])), (0,0,255), 3)
        
        # ========================
        # CONFIDENCE CALCULATION
        # ========================
        
        confidence = 0.0
        if lane_center is not None:
            # Base confidence on number of detected points
            total_points = len(left_pts) + len(right_pts) if 'left_pts' in locals() and 'right_pts' in locals() else 0
            point_confidence = min(100.0, total_points / 20.0 * 100.0)  # 20 points = 100% confidence
            
            # Reduce confidence for unusual lane widths
            width_confidence = 100.0
            if lane_width is not None:
                if lane_width < 2.5 or lane_width > 5.0:  # Unusual lane width range
                    width_confidence = max(20.0, 100.0 - abs(lane_width - 3.7) * 20.0)
            
            # Penalty for single lane detection
            if (left_fit is None) != (right_fit is None):  # XOR - only one lane
                single_lane_penalty = 0.6
            else:
                single_lane_penalty = 1.0
                
            confidence = min(100.0, point_confidence * (width_confidence / 100.0) * single_lane_penalty)
        
        # ========================
        # DISTANCE ESTIMATION
        # ========================
        
        # Estimate distance to lane center for control purposes
        estimated_distance = 15.0  # Default lookahead distance
        if lane_center is not None:
            # Adjust distance based on lane position in image
            center_y_ratio = lane_center[1] / height
            # Closer to bottom of image = closer distance
            estimated_distance = 5.0 + (1.0 - center_y_ratio) * 20.0  # 5-25m range

        # ========================
        # FINAL VISUALIZATION
        # ========================
        
        # Mark lane center
        if lane_center:
            cv2.circle(viz, lane_center, 10, (0, 255, 255), -1)
            
        # Add text overlays with detection metrics
        cv2.putText(viz, f'L-curv: {left_curverad:.0f}m' if left_curverad else 'L-curv: N/A', 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(viz, f'R-curv: {right_curverad:.0f}m' if right_curverad else 'R-curv: N/A', 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(viz, f'Width: {lane_width:.1f}m' if lane_width else 'Width: N/A', 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        return (
            viz, 
            lane_center, 
            float(left_curverad) if left_curverad is not None else None,
            float(right_curverad) if right_curverad is not None else None,
            float(lane_width) if lane_width is not None else None,
            float(confidence),
            float(estimated_distance)
        )