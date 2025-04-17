"""
Posture analysis module for detecting posture issues and providing guidance.
"""
import math

from config.settings import NECK_ANGLE_THRESHOLD, TORSO_ANGLE_THRESHOLD


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
        theta = math.acos((y2 - y1) * (-y1) /
                          (math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
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
            'neck_angle': None,
            'torso_angle': None,
            'shoulder_offset': None,
            'good_posture': False,
            'issues': {}
        }

        # Extract key landmarks
        l_shoulder = landmarks.get('l_shoulder')
        r_shoulder = landmarks.get('r_shoulder')
        l_ear = landmarks.get('l_ear')
        l_hip = landmarks.get('l_hip')

        # Check if all required landmarks are available
        if None in [l_shoulder, l_ear, l_hip]:
            return results

        # Unpack coordinates
        l_shldr_x, l_shldr_y = l_shoulder
        l_ear_x, l_ear_y = l_ear
        l_hip_x, l_hip_y = l_hip

        # Calculate shoulder offset if right shoulder is available
        if r_shoulder is not None:
            r_shldr_x, r_shldr_y = r_shoulder
            results['shoulder_offset'] = self.calculate_distance(
                l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)

        # Calculate angles
        results['neck_angle'] = self.calculate_angle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
        results['torso_angle'] = self.calculate_angle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

        # Determine posture quality
        results['good_posture'] = (
                results['neck_angle'] < NECK_ANGLE_THRESHOLD and
                results['torso_angle'] < TORSO_ANGLE_THRESHOLD
        )

        # Generate guidance for issues
        if results['neck_angle'] >= NECK_ANGLE_THRESHOLD:
            results['issues']['neck'] = "Straighten your neck (chin up)"

        if results['torso_angle'] >= TORSO_ANGLE_THRESHOLD:
            results['issues']['torso'] = "Sit upright (back straight)"

        if results['shoulder_offset'] is not None and results['shoulder_offset'] >= 100:
            results['issues']['shoulders'] = "Level your shoulders"

        return results
