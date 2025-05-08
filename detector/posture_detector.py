"""
Main posture detection module that integrates camera capture and posture analysis.
"""

import asyncio
import os
import signal
import sys
import time
from datetime import datetime, timedelta

import cv2
import mediapipe as mp

from config.settings import (
    BODY_COMPONENTS,
    CAMERA_FPS,
    COLORS,
    FONT_FACE,
    SEND_INTERVAL,
    SLIDING_WINDOW_DURATION,
    WARNING_COOLDOWN,
)
from detector.posture_analyzer import PostureAnalyzer
from utils.pigpio import PigpioClient
from utils.visualization import (
    draw_angle_text,
    draw_landmarks,
    draw_posture_guidance,
    draw_posture_indicator,
    draw_posture_lines,
    draw_status_bar,
    get_optimal_font_scale,
)


class PostureDetector:
    """Main class for posture detection"""

    def __init__(self, camera_manager, show_guidance=True, model_complexity=2, http_client=None):
        """
        Initialize posture detector

        Args:
            camera_manager: CameraManager instance for handling video capture
            show_guidance: Whether to show posture correction guidance
        """
        self.camera_manager = camera_manager
        self.show_guidance = show_guidance

        # Initialize frame counters
        self.good_frames = 0
        self.bad_frames = 0

        # Initialize MediaPipe pose detection
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=model_complexity, min_detection_confidence=0.7, min_tracking_confidence=0.7
        )

        # Initialize posture analyzer
        self.analyzer = PostureAnalyzer()

        # Warning timer
        self.last_alert_time = None

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.cleanup_and_exit)
        signal.signal(signal.SIGTERM, self.cleanup_and_exit)

        # Window resize control
        self.resize_mode = False
        self.window_name = "Posture Detection"

        self.old_posture = None

        self.last_sent_time = time.time()
        self.last_sent_posture = None
        self.SEND_INTERVAL = SEND_INTERVAL  # seconds

        self.history = []

        self.http_client = http_client

        self.gpio_client = PigpioClient()

    def _update_history(self, analysis_results):
        if analysis_results["webcam_placement"] != "good":
            return
        self.history.append((datetime.now(), analysis_results))

        # todo make it an async task
        # pop elements if SLIDING_WINDOW_DURATION (which is a duration in seconds) is reached
        while len(self.history) > 0:
            first_time, _ = self.history[0]
            now = datetime.now()
            diff = now - first_time
            if diff.total_seconds() > SLIDING_WINDOW_DURATION:
                self.history.pop(0)
            else:
                break

    def _get_average_score(self, seconds):
        # Get the current time
        now = datetime.now()
        # Calculate the time threshold
        time_threshold = now - timedelta(seconds=seconds)
        # Filter the history to include only entries within the time threshold
        filtered_history = [entry for entry in self.history if entry[0] >= time_threshold]
        # Calculate the average score for each component
        average_scores = {}

        for component_name, attributes in BODY_COMPONENTS.items():
            # Get the score name
            score_key = attributes["score"]
            # Get the scores for the filtered history
            scores = [entry[1][score_key] for entry in filtered_history]
            # Calculate the average score
            if len(scores) > 0:
                average_scores[score_key] = sum(scores) / len(scores)
            else:
                average_scores[score_key] = 0

        return average_scores

    def _maybe_send_posture(self, current_posture, analysis_results):
        if os.getenv("DISABLE_TELEMETRY", False).lower() in ["true", "1", "yes"]:
            return

        now = time.time()
        posture_changed = current_posture != self.last_sent_posture
        # todo we don't want to send data on each change, it does not make sense

        time_passed = now - self.last_sent_time > self.SEND_INTERVAL

        if posture_changed or time_passed:
            # self._prepare_data()

            components = self._get_average_score(SLIDING_WINDOW_DURATION)

            print(f"[Posture Update] Sending data (posture changed: {posture_changed})")
            self.http_client.send_posture_data(components)
            self.last_sent_time = now
            self.last_sent_posture = current_posture
            return True

        return False

    def cleanup_and_exit(self, signum=None, frame=None):
        """Clean up resources and exit the program"""
        print("\nShutting down posture detector...")
        if self.camera_manager.is_open():
            self.camera_manager.release()
        cv2.destroyAllWindows()
        sys.exit(0)

    def extract_landmarks(self, pose_landmarks, frame_width, frame_height):
        """
        Extract key landmarks from MediaPipe pose results

        Args:
            pose_landmarks: MediaPipe pose landmarks
            frame_width: Width of the frame
            frame_height: Height of the frame

        Returns:
            Dictionary: Key landmarks with coordinates
        """
        lm = pose_landmarks
        lmPose = self.mp_pose.PoseLandmark

        landmarks = {}

        try:
            # Left shoulder
            landmarks["l_shoulder"] = (
                int(lm.landmark[lmPose.LEFT_SHOULDER].x * frame_width),
                int(lm.landmark[lmPose.LEFT_SHOULDER].y * frame_height),
            )

            # Right shoulder
            landmarks["r_shoulder"] = (
                int(lm.landmark[lmPose.RIGHT_SHOULDER].x * frame_width),
                int(lm.landmark[lmPose.RIGHT_SHOULDER].y * frame_height),
            )

            # Both ears for better detection regardless of webcam position
            landmarks["l_ear"] = (
                int(lm.landmark[lmPose.LEFT_EAR].x * frame_width),
                int(lm.landmark[lmPose.LEFT_EAR].y * frame_height),
            )

            landmarks["r_ear"] = (
                int(lm.landmark[lmPose.RIGHT_EAR].x * frame_width),
                int(lm.landmark[lmPose.RIGHT_EAR].y * frame_height),
            )

            # Left hip
            landmarks["l_hip"] = (
                int(lm.landmark[lmPose.LEFT_HIP].x * frame_width),
                int(lm.landmark[lmPose.LEFT_HIP].y * frame_height),
            )

            # Right hip
            landmarks["r_hip"] = (
                int(lm.landmark[lmPose.RIGHT_HIP].x * frame_width),
                int(lm.landmark[lmPose.RIGHT_HIP].y * frame_height),
            )

            # Calculate visibility scores for ear landmarks
            # Higher score = more visible/reliable
            l_ear_vis = (
                lm.landmark[lmPose.LEFT_EAR].visibility if hasattr(lm.landmark[lmPose.LEFT_EAR], "visibility") else 0
            )
            r_ear_vis = (
                lm.landmark[lmPose.RIGHT_EAR].visibility if hasattr(lm.landmark[lmPose.RIGHT_EAR], "visibility") else 0
            )
            l_hip_vis = (
                lm.landmark[lmPose.LEFT_HIP].visibility if hasattr(lm.landmark[lmPose.LEFT_HIP], "visibility") else 0
            )
            r_hip_vis = (
                lm.landmark[lmPose.RIGHT_HIP].visibility if hasattr(lm.landmark[lmPose.RIGHT_HIP], "visibility") else 0
            )
            l_shoulder_vis = (
                lm.landmark[lmPose.LEFT_SHOULDER].visibility
                if hasattr(lm.landmark[lmPose.LEFT_SHOULDER], "visibility")
                else 0
            )
            r_shoulder_vis = (
                lm.landmark[lmPose.RIGHT_SHOULDER].visibility
                if hasattr(lm.landmark[lmPose.RIGHT_SHOULDER], "visibility")
                else 0
            )

            # Add information about which ear is more visible (useful for analyzing posture)
            landmarks["primary_ear"] = "left" if l_ear_vis >= r_ear_vis else "right"
            landmarks["l_ear_visibility"] = l_ear_vis
            landmarks["r_ear_visibility"] = r_ear_vis
            landmarks["l_hip_visibility"] = l_hip_vis
            landmarks["r_hip_visibility"] = r_hip_vis
            landmarks["l_shoulder_visibility"] = l_shoulder_vis
            landmarks["r_shoulder_visibility"] = r_shoulder_vis

            return landmarks

        except Exception as e:
            print(f"Error extracting landmarks: {e}")
            return {}

    async def process_frame(self, frame):
        """
        Process a single frame for posture detection

        Args:
            frame: Camera frame to process

        Returns:
            Processed frame with annotations
        """
        # Get height and width
        h, w = frame.shape[:2]
        font_scale = get_optimal_font_scale(w)
        thickness = max(1, int(w / 640))

        # Convert the BGR image to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image with MediaPipe
        result = self.pose.process(rgb_frame)

        # If no pose detected, return the original frame with message
        if not result.pose_landmarks:
            cv2.putText(
                frame,
                "No pose detected",
                (10, 30),
                FONT_FACE,
                font_scale,
                COLORS["red"],
                thickness,
            )
            return frame

        # Extract landmarks
        landmarks = self.extract_landmarks(result.pose_landmarks, w, h)

        # If landmarks extraction failed, return original frame
        if not landmarks:
            cv2.putText(
                frame,
                "Landmark extraction failed",
                (10, 30),
                FONT_FACE,
                font_scale,
                COLORS["red"],
                thickness,
            )
            return frame

        # Analyze posture
        analysis_results = self.analyzer.analyze_posture(landmarks, self.http_client.last_sensitivity)

        self._update_history(analysis_results)

        # todo if person not visible, show to display

        if not analysis_results["good_posture"]:
            scores = self._get_average_score(SLIDING_WINDOW_DURATION)
            sensitivity = self.http_client.last_sensitivity

            for component, score in scores.items():
                if score < sensitivity:
                    print("your average is very bad bro:", component, "is", score)
                    now = datetime.now()
                    if self.last_alert_time is None or now - self.last_alert_time > timedelta(seconds=WARNING_COOLDOWN):
                        await self.gpio_client.long_alert()
                        # asyncio.create_task(self.gpio_client.long_alert()) # todo decide if we want to use this
                        self.last_alert_time = now
                        # todo alert to display

        draw_landmarks(frame, landmarks)

        # Update frame counters based on posture quality
        if analysis_results["good_posture"]:
            self.bad_frames = 0
            self.good_frames += 1

            # Draw lines with good posture color
            draw_posture_lines(frame, landmarks, COLORS["green"])
        else:
            self.good_frames = 0
            self.bad_frames += 1

            # Draw lines with bad posture color
            draw_posture_lines(frame, landmarks, COLORS["red"])

        # Calculate timing values for UI
        analysis_results["good_time"] = (1 / CAMERA_FPS) * self.good_frames
        analysis_results["bad_time"] = (1 / CAMERA_FPS) * self.bad_frames

        # Draw posture angles
        color = COLORS["light_green"] if analysis_results["good_posture"] else COLORS["red"]

        # Update landmarks with head tilted back status for visualization
        landmarks["is_head_tilted_back"] = analysis_results["is_head_tilted_back"]

        draw_angle_text(
            frame,
            landmarks,
            analysis_results["neck_score"],
            analysis_results["torso_score"],
            color,
        )

        # Add main angle text at top
        angle_text = f'Neck: {int(analysis_results["neck_score"])}°  Torso: {int(analysis_results["torso_score"])}°'
        cv2.putText(frame, angle_text, (10, 30), FONT_FACE, font_scale, color, thickness)

        # Draw posture indicator (GOOD/BAD)
        draw_posture_indicator(frame, analysis_results["good_posture"])

        # Draw status bar
        draw_status_bar(frame, analysis_results)

        # Draw posture correction guidance if enabled
        if self.show_guidance and not analysis_results["good_posture"]:
            frame = draw_posture_guidance(frame, analysis_results)

        return frame

    def handle_keyboard_input(self, key):
        """
        Handle keyboard input during the application

        Args:
            key: Key pressed by user

        Returns:
            Boolean: True to continue, False to exit
        """
        if key == ord("q"):
            return False
        elif key == ord("r"):
            # Toggle resize mode
            self.resize_mode = not self.resize_mode
            mode_text = "ON" if self.resize_mode else "OFF"
            print(f"Resize mode: {mode_text}")
            if self.resize_mode:
                print("Use arrow keys to resize the window. Press 'r' again to exit resize mode.")
        elif key == ord("f"):
            # Toggle fullscreen
            current_prop = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN)
            if current_prop == cv2.WINDOW_NORMAL:
                cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                print("Fullscreen mode enabled")
            else:
                cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                print("Fullscreen mode disabled")
        elif self.resize_mode:
            # Handle resize control
            width, height = (
                self.camera_manager.frame_width,
                self.camera_manager.frame_height,
            )

            # Adjust dimensions based on arrow keys
            if key == 82 or key == ord("w"):  # Up arrow or 'w'
                height = int(height * 1.1)
            elif key == 84 or key == ord("s"):  # Down arrow or 's'
                height = int(height * 0.9)
            elif key == 83 or key == ord("d"):  # Right arrow or 'd'
                width = int(width * 1.1)
            elif key == 81 or key == ord("a"):  # Left arrow or 'a'
                width = int(width * 0.9)

            # Apply new dimensions if changed
            if width != self.camera_manager.frame_width or height != self.camera_manager.frame_height:
                actual_width, actual_height = self.camera_manager.resize_frame(width, height)
                print(f"Resized camera frame to {actual_width}x{actual_height}")

        return True

    async def run(self):
        """Main function to run the posture detection"""
        try:
            # Initialize webcam
            frame_width, frame_height = self.camera_manager.initialize()
            print(f"Camera initialized with resolution {frame_width}x{frame_height}")

            # Create named window for display with more window control
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

            # Set initial window size to something reasonable
            init_window_width = min(1280, frame_width * 1.5)
            init_window_height = int(init_window_width * frame_height / frame_width)

            cv2.resizeWindow(self.window_name, int(init_window_width), int(init_window_height))

            print("Posture detector running.")
            print("- Press 'q' to quit")
            print("- Press 'r' to enter resize mode, then use arrow keys to resize camera frame")
            print("- Press 'f' to toggle fullscreen mode")

            while True:
                while not self.http_client.last_session_status:
                    await asyncio.sleep(1)
                # Read frame from webcam
                success, frame = self.camera_manager.read_frame()

                if not success:
                    print("Error: Failed to capture image from webcam")
                    break

                # Process the frame
                processed_frame = await self.process_frame(frame)

                # Display the processed frame
                cv2.imshow(self.window_name, processed_frame)

                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                if not self.handle_keyboard_input(key):
                    break

                # Check if window was closed
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("Window closed by user")
                    break

                # Give other tasks a chance to run
                await asyncio.sleep(0.1)
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            # print the stacktrace
            raise e
        finally:
            # Ensure cleanup always happens
            self.cleanup_and_exit()
