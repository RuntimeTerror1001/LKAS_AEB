import numpy as np

"""
SPEED PID CONTROLLER WITH ADVANCED FEATURES
"""
class SpeedPID:
    def __init__(self, params):
        """
        Initialize PID controller for vehicle speed control with advanced anti-windup and filtering.
        
        Args:
            params (dict): Control parameters containing PID gains and limits
        """
        self.params = params['control']

        # ========================
        # PID GAINS
        # ========================
        self.kp = self.params['speed_pid_gains']['p']  # Proportional gain
        self.ki = self.params['speed_pid_gains']['i']  # Integral gain
        self.kd = self.params['speed_pid_gains']['d']  # Derivative gain

        # ========================
        # OUTPUT LIMITS
        # ========================
        self.max_throttle = self.params['max_throttle']
        self.max_brake = self.params['max_brake']

        # ========================
        # STATE VARIABLES 
        # ========================
        self.integral = 0.0                # Integral accumulator
        self.prev_error = 0.0              # Previous error for derivative calculation
        self.last_time = None              # Previous timestamp
        self.prev_throttle = 0.0           # Previous throttle output for smoothing
        self.prev_brake = 0.0              # Previous brake output for smoothing
        
        # ========================
        # ADVANCED CONTROL FEATURES
        # ========================
        self.smoothing_factor = 0.7        # Output filtering coefficient
        self.dead_zone = 0.5               # Speed error dead zone (m/s)
        self.max_integral = 10.0           # Anti-windup integral limit
        self.derivative_filter = 0.2       # Low-pass filter for derivative term
        self.filtered_derivative = 0.0     # Filtered derivative value

        # ========================
        # ADAPTIVE CONTROL SYSTEM
        # ========================
        self.error_history = []            # Recent error history for adaptation
        self.max_error_history = 10        # Maximum error history length

        # ========================
        # SATURATION HANDLING
        # ========================
        self.is_saturated = False          # Current saturation state
        self.saturation_counter = 0        # Consecutive saturation frames

    def reset(self):
        """
        Reset all controller state variables to initial conditions.
        Used when starting new control session or after significant disturbance.
        """
        self.integral = 0.0
        self.prev_error = 0.0
        self.last_time = None
        self.prev_throttle = 0.0
        self.prev_brake = 0.0
        self.filtered_derivative = 0.0
        self.error_history = []
        self.is_saturated = False
        self.saturation_counter = 0
    
    def apply_dead_zone(self, error):
        """
        Apply dead zone to speed error to reduce oscillations around target speed.
        
        Args:
            error (float): Raw speed error in m/s
            
        Returns:
            float: Dead-zone adjusted error
        """
        if abs(error) < self.dead_zone:
            return 0.0
        elif error > 0:
            return error - self.dead_zone
        else:
            return error + self.dead_zone
    
    def update_error_history(self, error):
        """
        Maintain rolling history of speed errors for adaptive behavior.
        
        Args:
            error (float): Current speed error in m/s
        """
        self.error_history.append(abs(error))
        if len(self.error_history) > self.max_error_history:
            self.error_history.pop(0)
    
    def get_adaptive_gains(self, error):
        """
        Dynamically adjust PID gains based on error magnitude and recent performance.
        
        Args:
            error (float): Current speed error in m/s
            
        Returns:
            tuple: (kp, ki, kd) adjusted gains
        """
        kp, ki, kd = self.kp, self.ki, self.kd

        # Reduce gains for persistent large errors to prevent overshooting
        if len(self.error_history) >= 5:
            avg_error = np.mean(self.error_history[-5:])
            if avg_error > 3.0: # Large persistent error
                kp *= 0.8
                ki *= 0.6
                kd *= 0.9
            
        # Increase gains for small errors to improve precision
        if abs(error) < 1.0:
            kp *= 1.2
            ki *= 1.1
        
        return kp, ki, kd
    
    def get_speed_dependent_factor(self, curr_speed, target_speed):
        """
        Calculate speed-dependent throttle modulation for smoother control.
        
        Args:
            curr_speed (float): Current vehicle speed in m/s
            target_speed (float): Target vehicle speed in m/s
            
        Returns:
            float: Speed-dependent scaling factor (0.5 to 1.0)
        """
        # Reduce throttle at very low speeds to prevent jerky motion
        if curr_speed < 2.0:  # Very low speed
            return 0.7
        
        # Reduce throttle when approaching target speed to prevent overshoot
        if target_speed > 0:
            speed_ratio = curr_speed / target_speed
            if speed_ratio > 0.9:  # Near target speed
                return max(0.5, 1.0 - (speed_ratio - 0.9) * 5)
        
        return 1.0

    def update(self, target_speed, curr_speed, curr_time):
        """
        Main PID control update function with advanced features.
        
        Args:
            target_speed (float): Desired speed in m/s
            curr_speed (float): Current vehicle speed in m/s
            curr_time (float): Current timestamp in seconds
            
        Returns:
            tuple: (throttle_command, brake_command) both in range [0.0, max_value]
        """
        # ========================
        # ERROR CALCULATION
        # ========================
        raw_error = target_speed - curr_speed
        error = self.apply_dead_zone(raw_error)

        # Update error history
        self.update_error_history(raw_error)

        # ========================
        # TIME DELTA CALCULATION
        # ========================
        dt = 0.05  # Default for first run (20Hz)
        if self.last_time is not None:
            dt = curr_time - self.last_time
            dt = np.clip(dt, 0.01, 0.2)  # Prevent extreme dt values
        self.last_time = curr_time

        # ========================
        # ADAPTIVE GAIN CALCULATION
        # ========================
        kp, ki, kd = self.get_adaptive_gains(error)

        # ========================
        # PID TERM CALCULATION
        # ========================

        # Proportional Term
        proportional = kp * error

        # Integral term with conditional integration (anti-windup)
        should_integrate = True
        if self.is_saturated:
            # Don't integrate if saturated and integration would make it worse
            if (self.integral > 0 and error > 0) or (self.integral < 0 and error < 0):
                should_integrate = False

        if should_integrate:
            self.integral += error * dt
            self.integral = np.clip(self.integral, -self.max_integral, self.max_integral)

        integral = ki * self.integral

        # Derivative term with noise filtering
        raw_derivative = (error - self.prev_error) / dt
        self.filtered_derivative = (self.derivative_filter * raw_derivative + 
                                  (1 - self.derivative_filter) * self.filtered_derivative)
        derivative = kd * self.filtered_derivative
        self.prev_error = error

        # ========================
        # PID OUTPUT CALCULATION
        # ========================
        output = proportional + integral + derivative

        # Track Saturation State
        self.is_saturated = (output > self.max_throttle) or (output < -self.max_brake)
        if self.is_saturated:
            self.saturation_counter += 1
        else:
            self.saturation_counter = max(0, self.saturation_counter)

        # Apply output limits
        output = np.clip(output, -self.max_brake, self.max_throttle)

        # ========================
        # CONVERT TO THROTTLE/BRAKE COMMANDS
        # ========================
        if output >= 0:
            raw_throttle = output
            raw_brake = 0.0
        else:
            raw_throttle = 0.0
            raw_brake = -output

        # ========================
        # OUTPUT SMOOTHING AND FILTERING
        # ========================
        if raw_throttle > 0:
            # Smooth throttle application
            filtered_throttle =  (self.smoothing_factor * raw_throttle + (1 - self.smoothing_factor) * self.prev_throttle)
            self.prev_throttle = filtered_throttle

            # Quick brake release for responsiveness
            filtered_brake = 0.0
            self.prev_brake = 0.0
        else:
            # Quick throttle release for safety
            filtered_throttle = 0.0  
            self.prev_throttle = 0.0
            
            # Smooth brake application with faster response for safety
            brake_smoothing = 0.8  # Less smoothing for brakes (safety)
            filtered_brake = (brake_smoothing * raw_brake + 
                            (1 - brake_smoothing) * self.prev_brake)
            self.prev_brake = filtered_brake
        
        # ========================
        # SPEED-DEPENDENT ADJUSTMENTS
        # ========================
        speed_factor = self.get_speed_dependent_factor(curr_speed, target_speed)
        filtered_throttle *= speed_factor

        # ========================
        # FINAL SAFETY PROCESSING
        # ========================
        final_throttle = np.clip(filtered_throttle, 0.0, self.max_throttle)
        final_brake = np.clip(filtered_brake, 0.0, self.max_brake)

        # Ensure mutual exclusivity of throttle and brake
        if final_brake > 0.01:
            final_throttle = 0.0
        
        return final_throttle, final_brake
    
    def get_debug_info(self):
        """
        Return comprehensive debug information for controller monitoring and tuning.
        
        Returns:
            dict: Debug information containing all internal controller states
        """
        return {
            'integral': self.integral,
            'filtered_derivative': self.filtered_derivative,
            'prev_error': self.prev_error,
            'is_saturated': self.is_saturated,
            'saturation_counter': self.saturation_counter,
            'error_history_avg': np.mean(self.error_history) if self.error_history else 0.0,
            'prev_throttle': self.prev_throttle,
            'prev_brake': self.prev_brake
        }