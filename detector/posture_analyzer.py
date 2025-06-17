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
                t = (x - x0) / (x1 - x0)
                return y0 + t * (y1 - y0)

    def analyze_posture(self, landmarks, sensitivity=-1):
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
        # Higher visibility on left side means webcam is on the right and vice versa
        if l_ear_vis > r_ear_vis:
            if self.same_side_frames == -1 or self.same_side_frames == 60:
                self.webcam_position = "right"  # If left ear is more visible, webcam is on right
                self.same_side_frames = 0
        else:
            if self.same_side_frames == -1 or self.same_side_frames == 60:
                self.webcam_position = "left"  # If right ear is more visible, webcam is on left
                self.same_side_frames = 0
        if self.same_side_frames < 60:
            self.same_side_frames += 1

        results["webcam_position"] = self.webcam_position

        results["webcam_placement"] = "good"
        if (results["webcam_position"] == "right" and r_ear_vis < 0.90) or (
            results["webcam_position"] == "left" and l_ear_vis < 0.90
        ):
            results["webcam_placement"] = "ear"

        if max(l_hip_vis, r_hip_vis) < 0.75:
            results["webcam_placement"] = "hip"

        if min(l_shoulder_vis, r_shoulder_vis) < 0.93:
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

        # Calculate shoulder offset
        results["shoulders_offset"] = self.calculate_distance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)

        # Calculate angles
        results["neck_angle"] = self.calculate_angle(shoulder_x, shoulder_y, ear_x, ear_y)
        results["torso_angle"] = self.calculate_angle(hip_x, hip_y, shoulder_x, shoulder_y)
        # results["reclination"] = self.calculate_angle(

        # Calculate relative angle between neck and torso
        results["relative_neck_angle"] = min(abs(results["neck_angle"] - results["torso_angle"]), results["neck_angle"])

        # print("-------------------")
        # print(results["torso_angle"])
        # print(results["relative_neck_angle"])

        # Alternative condition: neck angle is smaller than torso angle (head is actually back)
        # This happens in a true reclined position
        # neck_behind_torso = results["neck_angle"] < results["torso_angle"] todo capire se serve ancora a qualcosa (non penso)

        # this helps a bit with reclined chairs, otherwise is too aggressive
        relative_neck_angle = results["relative_neck_angle"]
        if results["torso_angle"] <= -30:
            relative_neck_angle = int(relative_neck_angle / 1.5)

        # compute scores
        positive_neck_angle = relative_neck_angle if relative_neck_angle >= 0 else -relative_neck_angle
        positive_torso_angle = results["torso_angle"] if results["torso_angle"] >= 0 else -results["torso_angle"]

        results["neck_score"] = self.compute_score(NECK_SCORE_MAP, positive_neck_angle)
        results["torso_score"] = self.compute_score(TORSO_SCORE_MAP, positive_torso_angle)
        results["shoulders_score"] = self.compute_score(SHOULDERS_SCORE_MAP, results["shoulders_offset"])

        results["good_posture"] = (
            results["neck_score"] >= sensitivity
            and results["neck_score"] >= sensitivity
            and results["neck_score"] >= sensitivity
        )

        return results
