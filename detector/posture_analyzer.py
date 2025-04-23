"""
Posture analysis module for detecting posture issues and providing guidance.
"""

import math

from config.settings import (
    NECK_ANGLE_THRESHOLD,
    RELATIVE_NECK_ANGLE_THRESHOLD,
    TORSO_ANGLE_THRESHOLD,
)


class PostureAnalyzer:
    """Analyzes posture based on landmark positions"""

    def __init__(self):
        """Initialize the posture analyzer"""
        pass

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
        if y1 == y2:  # Avoid division by zero
            return 90

        # Calculate the angle with respect to vertical
        theta = math.acos((y2 - y1) * (-y1) / (math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
        return int(180 / math.pi * theta)

    def analyze_posture(self, landmarks):
        """
        Analyze posture based on the landmarks

        Args:
            landmarks: Dictionary of landmark coordinates

        Returns:
            Dictionary: Results of posture analysis
        """
        results = {
            "neck_angle": None,
            "torso_angle": None,
            "shoulder_offset": None,
            "good_posture": False,
            "issues": {},
            "webcam_position": None,
            "relative_neck_angle": None,
            "is_head_tilted_back": False,
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

        # Determine webcam position relative to the user
        # Higher visibility on left side means webcam is on the right and vice versa
        if l_ear_vis > r_ear_vis:
            results["webcam_position"] = "right"  # If left ear is more visible, webcam is on right
        else:
            results["webcam_position"] = "left"  # If right ear is more visible, webcam is on left

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

        # Calculate shoulder offset
        results["shoulder_offset"] = self.calculate_distance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)

        # Calculate angles
        results["neck_angle"] = self.calculate_angle(shoulder_x, shoulder_y, ear_x, ear_y)
        results["torso_angle"] = self.calculate_angle(hip_x, hip_y, shoulder_x, shoulder_y)

        # Calculate relative angle between neck and torso
        results["relative_neck_angle"] = abs(results["neck_angle"] - results["torso_angle"])

        # Detect if the neck is tilted back (head leaned back)
        # Consider two key factors for a tilted back detection:
        #  1. Is the torso angle relatively large? (indicates leaning back)
        #  2. Is the neck angle reasonably aligned with the torso?
        torso_leaning_back = results["torso_angle"] > 20  # Moderate recline
        neck_aligned_with_torso = results["relative_neck_angle"] <= RELATIVE_NECK_ANGLE_THRESHOLD

        # Alternative condition: neck angle is smaller than torso angle (head is actually back)
        # This happens in a true reclined position
        neck_behind_torso = results["neck_angle"] < results["torso_angle"]

        # Mark as tilted back if EITHER:
        # - Torso is leaning back and neck is properly aligned with it, OR
        # - Neck is clearly positioned behind the torso line
        results["is_head_tilted_back"] = (torso_leaning_back and neck_aligned_with_torso) or neck_behind_torso

        # Determine posture quality based on context
        if results["is_head_tilted_back"]:
            # When head is tilted back, evaluate based on the relative angle
            results["good_posture"] = results["relative_neck_angle"] <= RELATIVE_NECK_ANGLE_THRESHOLD
        else:
            # In standard upright position, use the standard thresholds
            results["good_posture"] = (
                results["neck_angle"] < NECK_ANGLE_THRESHOLD and results["torso_angle"] < TORSO_ANGLE_THRESHOLD
            )

        # Generate guidance for issues
        if results["is_head_tilted_back"]:
            if results["relative_neck_angle"] > RELATIVE_NECK_ANGLE_THRESHOLD:
                results["issues"]["neck"] = "Align your neck with your torso"
        else:
            if results["neck_angle"] > NECK_ANGLE_THRESHOLD:
                results["issues"]["neck"] = "Straighten your neck"

            if results["torso_angle"] > TORSO_ANGLE_THRESHOLD:
                results["issues"]["torso"] = "Sit upright"

        if results["shoulder_offset"] >= 100:
            results["issues"]["shoulders"] = "Level your shoulders"

        return results
