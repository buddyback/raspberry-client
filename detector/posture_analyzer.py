"""
Posture analysis module for detecting posture issues and providing guidance.
"""

import math

from config.settings import (
    NECK_SCORE_MAP,
    SHOULDERS_SCORE_MAP,
    TORSO_SCORE_MAP,
)


def is_looking_at_camera(landmarks):
    """
    Determine if user is looking at camera based on facial landmarks
    """
    left_eye = landmarks[2]
    right_eye = landmarks[5]

    visibility = min(left_eye.visibility, right_eye.visibility)

    return visibility > 0.995  # todo make it a parameter


class PostureAnalyzer:
    """Analyzes posture based on landmark positions"""

    def __init__(self):
        """Initialize the posture analyzer"""
        self.same_side_frames = -1
        self.webcam_position = ""
        self.webcam_placement = "good"
        
        # Calibration state - captures baselines for ALL measurements
        self._calibrating = False
        self._calibration_samples = {
            "torso": [],
            "neck": [],
            "shoulders": [],
            "torso_length": [],  # Used for perspective correction
        }
        self._calibration_duration = 3.0  # seconds to collect samples
        
        # Baseline offsets to compensate for webcam angle
        self._baseline_torso_angle = 0.0
        self._baseline_neck_angle = 0.0  # For relative neck angle
        self._baseline_shoulders_offset = 0.0
        self._perspective_factor = 1.0  # Multiplier for angle deviations
        self._is_calibrated = False
    
    @property
    def is_calibrating(self) -> bool:
        """Return True if currently in calibration mode."""
        return self._calibrating
    
    @property
    def is_calibrated(self) -> bool:
        """Return True if calibration has been completed."""
        return self._is_calibrated
    
    @property
    def baseline_torso_angle(self) -> float:
        """Return the calibrated baseline torso angle."""
        return self._baseline_torso_angle
        
    @property
    def perspective_factor(self) -> float:
        """Return the calculated perspective correction factor."""
        return self._perspective_factor
    
    def start_calibration(self):
        """
        Start the calibration process.
        Call this when user triggers calibration (e.g., presses 'c').
        """
        self._calibrating = True
        self._calibration_samples = {
            "torso": [],
            "neck": [],
            "shoulders": [],
            "torso_length": [],
        }
        print("[Calibration] Started - please sit in your best posture...")
    
    def add_calibration_sample(self, torso_angle: float, neck_angle: float, shoulders_offset: float, torso_length: float):
        """
        Add samples during calibration.
        Called automatically during analyze_posture when calibrating.
        
        Args:
            torso_angle: The current torso angle measurement
            neck_angle: The current relative neck angle measurement
            shoulders_offset: The current shoulder offset measurement
            torso_length: The current torso pixel length (for normalization)
        """
        if self._calibrating:
            self._calibration_samples["torso"].append(torso_angle)
            self._calibration_samples["neck"].append(neck_angle)
            self._calibration_samples["shoulders"].append(shoulders_offset)
            self._calibration_samples["torso_length"].append(torso_length)
    
    def _median(self, values: list) -> float:
        """Calculate median of a list of values."""
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        mid = len(sorted_vals) // 2
        if len(sorted_vals) % 2 == 0:
            return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
        return sorted_vals[mid]
    
    def complete_calibration(self) -> bool:
        """
        Complete the calibration and compute baselines for all measurements.
        Also computes perspective correction factor.
        
        Returns:
            True if calibration succeeded, False if not enough samples
        """
        if not self._calibration_samples["torso"]:
            print("[Calibration] Failed - no samples collected")
            self._calibrating = False
            return False
        
        # Use median to be robust to outliers
        self._baseline_torso_angle = self._median(self._calibration_samples["torso"])
        self._baseline_neck_angle = self._median(self._calibration_samples["neck"])
        self._baseline_shoulders_offset = self._median(self._calibration_samples["shoulders"])
        
        baseline_torso_length = self._median(self._calibration_samples["torso_length"])
        
        # Calculate perspective factor
        # Ratio of shoulder width to torso length tells us how "frontal" the view is
        # Higher ratio = more frontal or more angled camera (seeing both shoulders)
        # Wait, if camera is perfectly side-on (90 deg), shoulders overlap -> offset is small -> ratio small
        # If camera is 45 deg, shoulders visible -> offset large -> ratio large
        # We want HIGHER sensitivity when ratio is LARGE (angled view) ??
        # No, wait:
        # Side view (90 deg): Shoulder offset ~ 0. Sensitivity is normal (movements map directly).
        # Front view (0 deg): Shoulder offset is MAX. Torso angle sensitivity is ZERO (can't see lean).
        # Angled view (45 deg): Shoulder offset is MEDIUM. Torso angle sensitivity is REDUCED.
        
        # Actually: 
        # Ideally, shoulder offset should be 0 for side view.
        # If offset > 0, it means we are seeing some front/back perspective.
        # This PERSPECTIVE shortens the observed angles of forward/backward leans.
        # So we want to BOOST the angles more as the shoulder offset increases.
        
        if baseline_torso_length > 0:
            shoulder_ratio = self._baseline_shoulders_offset / baseline_torso_length
            # Empirical tuning: 
            # If ratio is 0.0 (perfect side), factor = 1.0
            # If ratio is 0.5 (angled), factor = 1.0 + (4.0 * 0.5) = 3.0 (boost angles by 3x)
            self._perspective_factor = 1.0 + (4.0 * shoulder_ratio)
        else:
            self._perspective_factor = 1.0

        self._is_calibrated = True
        self._calibrating = False
        print(f"[Calibration] Complete!")
        print(f"  Torso baseline: {self._baseline_torso_angle:.1f}°")
        print(f"  Neck baseline: {self._baseline_neck_angle:.1f}°")
        print(f"  Shoulders baseline: {self._baseline_shoulders_offset:.1f}px")
        print(f"  Perspective Factor: {self._perspective_factor:.2f}x (Ratio: {shoulder_ratio:.2f})")
        return True

    
    def reset_calibration(self):
        """Reset calibration to default (no compensation)."""
        self._baseline_torso_angle = 0.0
        self._baseline_neck_angle = 0.0
        self._baseline_shoulders_offset = 0.0
        self._is_calibrated = False
        self._calibrating = False
        self._calibration_samples = {"torso": [], "neck": [], "shoulders": []}
        print("[Calibration] Reset to default")

    def calculate_distance(self, x1, y1, x2, y2):
        """
        Calculate Euclidean distance between two points

        Args:
            x1, y1: Coordinates of first point
            x2, y2: Coordinates of second point

        Returns:
            Float: Distance between the points
        """
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def calculate_angle(self, x1, y1, x2, y2):
        """
        Calculate angle between two points with respect to the vertical

        Args:
            x1, y1: Coordinates of first point
            x2, y2: Coordinates of second point

        Returns:
            Integer: Angle in degrees
        """
        # Avoid division by zero - check for equal y coordinates
        if y1 == y2:
            return 90
        
        # Avoid division by zero - check for y1 being zero
        if y1 == 0:
            return 0
        
        # Calculate denominator and check for zero
        dx = x2 - x1
        dy = y2 - y1
        denominator = math.sqrt(dx ** 2 + dy ** 2) * y1
        
        # Avoid division by zero in denominator
        if abs(denominator) < 1e-10:
            return 0
        
        # Calculate the angle with respect to vertical
        # Clamp the value to [-1, 1] to avoid math domain error in acos
        cos_value = (dy * (-y1)) / denominator
        cos_value = max(-1.0, min(1.0, cos_value))
        
        try:
            theta = math.acos(cos_value)
        except ValueError:
            return 0

        if x2 < x1:
            theta = -theta

        return int(180 / math.pi * theta)

    @staticmethod
    def compute_score(points_map, x):
        """
        Interpola linearmente un valore x in base a una mappa di punti {x: y}.

        Args:
            points_map (dict): Mappa {x: y} con x ordinabili (es. angoli) e y score.
            x (float): Valore da valutare.

        Returns:
            float: Score interpolato.
        """
        sorted_points = sorted(points_map.items())

        # Clamp a valori fuori dai bordi
        if x <= sorted_points[0][0]:
            return sorted_points[0][1]
        if x >= sorted_points[-1][0]:
            return sorted_points[-1][1]

        # Trova il segmento dove x si trova
        for i in range(1, len(sorted_points)):
            x0, y0 = sorted_points[i - 1]
            x1, y1 = sorted_points[i]
            if x0 <= x <= x1:
                # Avoid division by zero
                if abs(x1 - x0) < 1e-10:
                    return y0
                t = (x - x0) / (x1 - x0)
                return y0 + t * (y1 - y0)

    def analyze_posture(self, landmarks, sensitivity=-1, visibility_thresholds=None, uses_reliable_visibility=True):
        """
        Analyze posture based on the landmarks

        Args:
            landmarks: Dictionary of landmark coordinates
            sensitivity: Threshold for good posture detection
            visibility_thresholds: Model-specific thresholds for landmark visibility
            uses_reliable_visibility: If True, use visibility values to detect webcam side.
                                      If False (MediaPipe), use ear X positions instead.

        Returns:
            Dictionary: Results of posture analysis
        """
        results = {
            "neck_angle": None,
            "torso_angle": None,
            "shoulders_offset": None,
            "good_posture": False,
            "issues": {},
            "webcam_position": None,
            "relative_neck_angle": None,
            "is_head_tilted_back": False,
            "neck_score": 0,
            "torso_score": 0,
        }

        # Extract key landmarks
        l_shoulder = landmarks.get("l_shoulder")
        r_shoulder = landmarks.get("r_shoulder")
        l_ear = landmarks.get("l_ear")
        r_ear = landmarks.get("r_ear")
        l_hip = landmarks.get("l_hip")
        r_hip = landmarks.get("r_hip")

        # Get visibility information
        primary_ear = landmarks.get("primary_ear", "left")  # Default to left if not specified
        l_ear_vis = landmarks.get("l_ear_visibility", 0)
        r_ear_vis = landmarks.get("r_ear_visibility", 0)

        l_hip_vis = landmarks.get("l_hip_visibility", 0)
        r_hip_vis = landmarks.get("r_hip_visibility", 0)
        l_shoulder_vis = landmarks.get("l_shoulder_visibility", 0)
        r_shoulder_vis = landmarks.get("r_shoulder_visibility", 0)

        # Determine webcam position relative to the user
        # The ear that faces the camera is the one on the same side as the webcam
        
        if uses_reliable_visibility:
            # OpenPose/MoveNet case: visibility values reliably indicate which ear is visible
            # Hidden ear has low/zero visibility, visible ear has high visibility
            facing_ear = "right" if r_ear_vis > l_ear_vis else "left"
        else:
            # MediaPipe case: visibility ~1.0 for both ears (unreliable)
            # Use shoulder X positions instead - the shoulder closer to camera appears
            # more toward the left side of the image (lower X in a standard side profile)
            # 
            # When webcam is on user's RIGHT:
            # - User faces right → right shoulder is closer to camera → right shoulder has LOWER X
            # - l_shoulder.x > r_shoulder.x
            # 
            # When webcam is on user's LEFT:
            # - User faces left → left shoulder is closer to camera → left shoulder has LOWER X
            # - r_shoulder.x > l_shoulder.x
            if l_shoulder is not None and r_shoulder is not None:
                l_shoulder_x = l_shoulder[0] if isinstance(l_shoulder, tuple) else 0
                r_shoulder_x = r_shoulder[0] if isinstance(r_shoulder, tuple) else 0
                
                # If left shoulder has higher X, user is facing right → webcam on right
                if l_shoulder_x > r_shoulder_x:
                    facing_ear = "right"
                else:
                    facing_ear = "left"
            else:
                # Fallback to visibility if we don't have shoulders
                facing_ear = "right" if r_ear_vis > l_ear_vis else "left"
        
        # Update position with debouncing
        if self.same_side_frames == -1 or self.same_side_frames == 60:
            self.webcam_position = facing_ear
            self.same_side_frames = 0
        if self.same_side_frames < 60:
            self.same_side_frames += 1

        results["webcam_position"] = self.webcam_position

        # Use provided thresholds or defaults (calibrated for MediaPipe)
        if visibility_thresholds is None:
            visibility_thresholds = {
                "ear": 0.90,
                "hip": 0.75,
                "shoulder": 0.80,
            }

        results["webcam_placement"] = "good"
        ear_threshold = visibility_thresholds.get("ear", 0.90)
        # When webcam is on user's RIGHT, the user's RIGHT ear faces camera and should be visible
        # When webcam is on user's LEFT, the user's LEFT ear faces camera and should be visible
        if (results["webcam_position"] == "right" and r_ear_vis < ear_threshold) or (
            results["webcam_position"] == "left" and l_ear_vis < ear_threshold
        ):
            results["webcam_placement"] = "ear"

        hip_threshold = visibility_thresholds.get("hip", 0.75)
        # print(max(l_hip_vis, r_hip_vis))
        if max(l_hip_vis, r_hip_vis) < hip_threshold:
            results["webcam_placement"] = "hip"

        # For side-positioned cameras, only one shoulder needs to be visible
        # Changed from min() to max() since the opposite shoulder will be occluded
        shoulder_threshold = visibility_thresholds.get("shoulder", 0.80)
        if max(l_shoulder_vis, r_shoulder_vis) < shoulder_threshold:
            results["webcam_placement"] = "shoulder"

        if self.webcam_placement != results["webcam_placement"]:
            print(results["webcam_placement"])

        self.webcam_placement = results["webcam_placement"]

        # Check if all required landmarks are available
        if None in [l_shoulder, r_shoulder] or (l_ear is None and r_ear is None) or (l_hip is None and r_hip is None):
            return results

        # Unpack coordinates
        l_shldr_x, l_shldr_y = l_shoulder
        r_shldr_x, r_shldr_y = r_shoulder

        # Use the more visible ear for neck angle calculation
        if primary_ear == "left" and l_ear is not None:
            ear_x, ear_y = l_ear
            shoulder_x, shoulder_y = l_shoulder  # Use left shoulder with left ear
        elif r_ear is not None:
            ear_x, ear_y = r_ear
            shoulder_x, shoulder_y = r_shoulder  # Use right shoulder with right ear
        else:
            # Fallback to whatever ear is available
            if l_ear is not None:
                ear_x, ear_y = l_ear
                shoulder_x, shoulder_y = l_shoulder
            else:
                ear_x, ear_y = r_ear
                shoulder_x, shoulder_y = r_shoulder

        # Use the more visible hip
        if primary_ear == "left" and l_hip is not None:  # Assume if left ear is more visible, left hip might be too
            hip_x, hip_y = l_hip
        elif r_hip is not None:
            hip_x, hip_y = r_hip
        else:
            # Fallback to whatever hip is available
            if l_hip is not None:
                hip_x, hip_y = l_hip
            else:
                hip_x, hip_y = r_hip

        # Calculate shoulder-to-shoulder offset (width)
        results["shoulders_offset"] = self.calculate_distance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)
        
        # Calculate torso length (approx distance from mid-shoulders to mid-hips)
        mid_shoulder_x = (l_shldr_x + r_shldr_x) / 2
        mid_shoulder_y = (l_shldr_y + r_shldr_y) / 2
        mid_hip_x = (l_hip_x + r_hip_x) / 2   if 'l_hip_x' in locals() and 'r_hip_x' in locals() else hip_x # Approximate
        mid_hip_y = (l_hip_y + r_hip_y) / 2   if 'l_hip_y' in locals() and 'r_hip_y' in locals() else hip_y # Approximate
        
        # Better approximation if we have both hips
        if l_hip is not None and r_hip is not None:
             mid_hip_x = (l_hip[0] + r_hip[0]) / 2
             mid_hip_y = (l_hip[1] + r_hip[1]) / 2
        else:
             mid_hip_x, mid_hip_y = hip_x, hip_y
             
        torso_length = self.calculate_distance(mid_shoulder_x, mid_shoulder_y, mid_hip_x, mid_hip_y)

        # Calculate angles
        results["neck_angle"] = self.calculate_angle(shoulder_x, shoulder_y, ear_x, ear_y)
        results["torso_angle"] = self.calculate_angle(hip_x, hip_y, shoulder_x, shoulder_y)
        # results["reclination"] = self.calculate_angle(

        # Calculate relative angle between neck and torso
        results["relative_neck_angle"] = min(abs(results["neck_angle"] - results["torso_angle"]), results["neck_angle"])

        # Collect calibration samples if calibrating (before applying baseline adjustments)
        if self._calibrating:
            self.add_calibration_sample(
                results["torso_angle"],
                results["relative_neck_angle"],
                results["shoulders_offset"],
                torso_length
            )

        # Alternative condition: neck angle is smaller than torso angle (head is actually back)
        # This happens in a true reclined position
        # neck_behind_torso = results["neck_angle"] < results["torso_angle"] todo capire se serve ancora a qualcosa (non penso)

        # this helps a bit with reclined chairs, otherwise is too aggressive
        relative_neck_angle = results["relative_neck_angle"]
        if results["torso_angle"] <= -30:
            relative_neck_angle = int(relative_neck_angle / 1.5)

        # Compute scores - apply calibration baselines and perspective correction
        # This compensates for webcam angle offset and perspective distortion
        
        # Neck: deviation from calibrated baseline * perspective factor
        calibrated_neck_angle = abs(relative_neck_angle - self._baseline_neck_angle) * self._perspective_factor
        
        # Torso: deviation from calibrated baseline * perspective factor
        calibrated_torso_angle = abs(results["torso_angle"] - self._baseline_torso_angle) * self._perspective_factor
        
        # Shoulders: deviation from calibrated baseline (no perspective correction needed for this one?)
        # Actually shoulder offset deviation IS the signal for bad posture, usually we want closer to 0 (side view).
        # But if we calibrated a baseline, we want to stay close to that baseline.
        calibrated_shoulders = abs(results["shoulders_offset"] - self._baseline_shoulders_offset)

        results["neck_score"] = self.compute_score(NECK_SCORE_MAP, calibrated_neck_angle)
        results["torso_score"] = self.compute_score(TORSO_SCORE_MAP, calibrated_torso_angle)
        results["shoulders_score"] = self.compute_score(SHOULDERS_SCORE_MAP, calibrated_shoulders)

        results["good_posture"] = (
            results["neck_score"] >= sensitivity
            and results["neck_score"] >= sensitivity
            and results["neck_score"] >= sensitivity
        )

        return results
