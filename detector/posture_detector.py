"""
Main posture detection module that integrates camera capture and posture analysis.
"""

import asyncio
import multiprocessing
import os
import signal
import time
from datetime import datetime, timedelta

import cv2
import mediapipe as mp
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QApplication

from config.settings import (
    ALERT_SLIDING_WINDOW_DURATION,
    COLORS,
    SEND_INTERVAL,
    SLIDING_WINDOW_DURATION,
    WARNING_COOLDOWN,
)
from config.settings import BODY_COMPONENTS
from detector.posture_analyzer import PostureAnalyzer, is_looking_at_camera
from utils.pigpio import PigpioClient
from utils.raspi_screen import set_screen_cooldown, turn_on_screen
from utils.visualization import (
    draw_landmarks,
    get_optimal_font_scale,
)


class PostureDetector(QObject):
    """Main class for posture detection"""

    def __init__(
        self, camera_manager, show_guidance=True, model_complexity=2, websocket_client=None, app_controller=None
    ):
        """
        Initialize posture detector

        Args:
            camera_manager: CameraManager instance for handling video capture
            show_guidance: Whether to show posture correction guidance
            model_complexity: Complexity of the MediaPipe pose model (0, 1, or 2)
            websocket_client: WebSocket client for sending/receiving data
            app_controller: Controller for the PyQt application
        """
        super().__init__()
        self.camera_manager = camera_manager
        self.show_guidance = show_guidance
        self.posture_data_updated = pyqtSignal(dict)

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

        self.old_posture = None

        self.last_sent_time = time.time()
        self.last_sent_posture = None
        self.SEND_INTERVAL = SEND_INTERVAL  # seconds

        self.history = []
        self.app_controller = app_controller
        self.websocket_client = websocket_client
        self.settings = {}

        # Store last frame data for UI updates
        self._last_landmarks = {}
        self._last_analysis_results = {}

        if os.getenv("DISABLE_VIBRATION", False).lower() not in ["true", "1", "yes"]:
            self.gpio_client = PigpioClient()

    def _update_history(self, analysis_results):
        if analysis_results["webcam_placement"] != "good":
            return
        self.history.append((datetime.now(), analysis_results))

        # todo make it an async task
        # pop elements if ALERT_SLIDING_WINDOW_DURATION (which is a duration in seconds) is reached
        while len(self.history) > 0:
            first_time, _ = self.history[0]
            now = datetime.now()
            diff = now - first_time
            if diff.total_seconds() > ALERT_SLIDING_WINDOW_DURATION:
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
                average_scores[score_key] = int(sum(scores) / len(scores))
            else:
                average_scores[score_key] = 0

        return average_scores

    def _maybe_send_posture(self, analysis_results):
        if os.getenv("DISABLE_TELEMETRY", False).lower() in ["true", "1", "yes"]:
            return

        now = time.time()
        time_passed = now - self.last_sent_time > self.SEND_INTERVAL

        if time_passed:
            # self._prepare_data()
            components = self._get_average_score(self.SEND_INTERVAL)
            print(f"[Posture Update] Sending data: {components}")
            asyncio.create_task(self.websocket_client.send_posture_data(components))
            self.last_sent_time = now
            return True

        return False

    def cleanup_and_exit(self, signum=None, frame=None):
        """Clean up resources and exit the program"""
        print("\nShutting down posture detector...")
        # Release camera
        if self.camera_manager.is_open():
            self.camera_manager.release()

        # Hide PyQt windows
        if self.app_controller:
            if hasattr(self.app_controller, "main_screen") and self.app_controller.main_screen:
                self.app_controller.main_screen.hide()
            if hasattr(self.app_controller, "posture_window") and self.app_controller.posture_window:
                self.app_controller.posture_window.hide()

        # Cancel all tasks
        try:
            for task in asyncio.all_tasks():
                if task != asyncio.current_task():
                    task.cancel()
        except Exception as e:
            print(f"Error canceling tasks: {e}")

        # Exit forcefully to ensure complete termination
        os._exit(0)

    def handle_keyboard_input(self, key):
        """
        Handle keyboard input during the application
        Note: This is a simplified version since we're not using OpenCV windows

        Args:
            key: Key pressed by user

        Returns:
            Boolean: True to continue, False to exit
        """
        if key == ord("q"):
            return False

        # All window resize functionality has been removed since we're not using OpenCV windows
        return True

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
        if not result.pose_landmarks:
            webcam_placement_text = f"Person is not visible"
            self.app_controller.posture_window.show_alert(
                webcam_placement_text
            )
            return frame

        # Extract landmarks
        landmarks = self.extract_landmarks(result.pose_landmarks, w, h)
        draw_landmarks(frame, landmarks)

        sensitivity = self.settings.get("sensitivity", -1)
        # Analyze posture
        analysis_results = self.analyzer.analyze_posture(landmarks, sensitivity)

        self._update_history(analysis_results)
        self._maybe_send_posture(analysis_results)

        last_scores = self._get_average_score(SLIDING_WINDOW_DURATION)

        webcam_placement = analysis_results.get("webcam_placement", "unknown")
        # todo if is sitted for long, start idle stuff

        results = {"scores": last_scores, "issues": dict()}

        if webcam_placement == "good":
            if last_scores[BODY_COMPONENTS["neck"]["score"]] < sensitivity:
                results["issues"]["neck"] = "Straighten your neck"

            if last_scores[BODY_COMPONENTS["torso"]["score"]] < sensitivity:
                results["issues"]["torso"] = "Sit upright"

            if last_scores[BODY_COMPONENTS["shoulders"]["score"]] < sensitivity:
                results["issues"]["shoulders"] = "Face the screen"
        else:
            results = {}
        colors = self.get_colors(SLIDING_WINDOW_DURATION)
        self.app_controller.posture_window.update_results(results, colors)

        if os.getenv("RASPI_DISPLAY", False).lower() in ["true", "1", "yes"]:
            user_looking = is_looking_at_camera(result.pose_landmarks.landmark)
            if user_looking:
                turn_on_screen()  # wake up the screen if user is looking at it

        if os.getenv("DISABLE_VIBRATION", False).lower() not in ["true", "1", "yes"]:
            # If the last posture is bad then...
            if not analysis_results["good_posture"]:
                scores = self._get_average_score(ALERT_SLIDING_WINDOW_DURATION)
                # For each component, check if the score is below the sensitivity threshold to trigger alert
                for component, score in scores.items():
                    if score < sensitivity:
                        print("bad avg score:", component, "is", score)
                        now = datetime.now()
                        if self.last_alert_time is None or now - self.last_alert_time > timedelta(
                            seconds=WARNING_COOLDOWN
                        ):
                            print("alert successfully sent")
                            p = multiprocessing.Process(target=self.gpio_client.long_alert_thread, args=(self.settings.get("vibration_intensity", 100),))
                            # Show alert in the posture window
                            webcam_placement_text = ""
                            match component:
                                case "neck_score":
                                    webcam_placement_text = "Straighten your neck"
                                case "torso_score":
                                    webcam_placement_text = "Sit upright"
                                case "shoulders_score":
                                    webcam_placement_text = "Face the desk"
                            self.app_controller.posture_window.show_alert(
                                webcam_placement_text, 5000
                            )
                            p.start()
                            self.last_alert_time = now


        # Update landmarks with head tilted back status for visualization
        landmarks["is_head_tilted_back"] = analysis_results["is_head_tilted_back"]

        # Store landmarks and analysis results for UI updates
        self._last_landmarks = landmarks
        self._last_analysis_results = analysis_results

        # Add main angle text at top
        if webcam_placement != "good":
            webcam_placement_text = f"{webcam_placement.upper()} is not visible"
            self.app_controller.posture_window.show_alert(
                webcam_placement_text
            )

        return frame

    def get_colors(self, sliding_window_size):
        """
        Get the colors for the posture components based on the sliding window size
        """
        scores = self._get_average_score(sliding_window_size)
        components = {}
        for component, attributes in BODY_COMPONENTS.items():
            score = scores.get(attributes["score"])
            if self.settings.get("sensitivity", 75) - score >= 10:
                components[component] = COLORS["red"]
            elif self.settings.get("sensitivity", 75) - score >= 0:
                components[component] = COLORS["yellow"]
            else:
                components[component] = COLORS["green"]
        return components

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

    async def update_settings(self):
        """
        Continuously pull the latest settings from the websocket client
        and update self.settings, without ever exiting.
        """
        while True:
            try:
                # If websocket isn't ready yet, just wait and retry
                # if not self.websocket_client.websocket or not self.websocket_client.websocket.open:
                #     await asyncio.sleep(1)
                #     continue
                # print("here")
                # Actually fetch the settings
                new_settings = await self.websocket_client.get_settings()
                if new_settings:
                    self.settings = new_settings
                    # (optional) print or log
                    print(f"[settings] updated → {self.settings}")

            except Exception as e:
                # Log the error, but keep the loop alive
                print(f"Error updating settings: {e}")
                await asyncio.sleep(1)  # Wait before retrying

            await asyncio.sleep(1)

    async def run(self):
        """Main function to run the posture detection"""
        try:
            # Initialize webcam
            frame_width, frame_height = self.camera_manager.initialize()
            print(f"Camera initialized with resolution {frame_width}x{frame_height}")

            print("Posture detector running.")
            print("- Press 'q' to quit")

            self.settings = await self.websocket_client.get_settings()
            print(
                f"Active session status: {'🟢 ACTIVE' if self.settings.get('has_active_session', False) else '🔴 INACTIVE'}"
            )

            # Start a background task for user commands
            # asyncio.create_task(self.websocket_client.process_user_commands())

            # Start a background task for sending heartbeats
            asyncio.create_task(self.websocket_client.send_heartbeats())

            # Listen for updates
            print("Listening for real-time updates (press Ctrl+C to stop)...")
            print("=" * 50)
            print("Commands:")
            print("  'data' - Send single posture data sample")
            print("=" * 50)
            print(f"DEBUG: Waiting for messages at {time.strftime('%H:%M:%S')}")

            # Start a task to continuously update settings
            asyncio.create_task(self.update_settings())

            # Get initial session state
            initial_session_active = self.settings.get("has_active_session", False)
            print(f"Initial session status: {'🟢 ACTIVE' if initial_session_active else '🔴 INACTIVE'}")

            # Ensure the main application window is shown
            if self.app_controller and hasattr(self.app_controller, "main_screen") and self.app_controller.main_screen:
                self.app_controller.main_screen.show()
            else:
                print("ERROR: AppController or main_screen is not available. UI might not function correctly.")

            # Set the initial view based on session state
            if initial_session_active:
                if self.app_controller:
                    self.app_controller.activate_session()
            else:
                if self.app_controller:
                    self.app_controller.end_session()

            # Track the current session state
            current_session_active = initial_session_active

            # Give the UI a moment to properly initialize
            await asyncio.sleep(0.2)

            while True:
                # Check if session state changed
                session_active_from_settings = self.settings.get("has_active_session", False)

                # Handle session state change
                if current_session_active != session_active_from_settings:
                    print(f"Session state changed: {'🟢 ACTIVE' if session_active_from_settings else '🔴 INACTIVE'}")

                    if self.app_controller:
                        if session_active_from_settings:
                            self.app_controller.activate_session()
                        else:
                            self.app_controller.end_session()

                    # Turn on the screen if session started
                    if os.getenv("RASPI_DISPLAY", False).lower() in ["true", "1", "yes"]:
                        if session_active_from_settings:
                            set_screen_cooldown(10800)  # 3 hours cooldown
                            turn_on_screen()
                        else:
                            set_screen_cooldown(5)

                    # Update tracking variable
                    current_session_active = session_active_from_settings

                    # Give the UI a moment to process the content transition
                    await asyncio.sleep(0.2)

                # If no active session, just wait and check again
                if not current_session_active:
                    # Process events while waiting to ensure UI remains responsive
                    QApplication.processEvents()
                    await asyncio.sleep(1)  # Check settings periodically
                    continue

                # Read frame from webcam
                success, frame = self.camera_manager.read_frame()

                if not success:
                    print("Error: Failed to capture image from webcam")
                    break

                # Process the frame
                processed_frame = await self.process_frame(frame)

                # Also update the posture window's webcam feed when session is active
                if current_session_active:
                    self.app_controller.posture_window.update_frame(
                        frame=processed_frame,
                        landmarks=getattr(self, "_last_landmarks", {}),
                        analysis_results=getattr(self, "_last_analysis_results", {}),
                        colors=self.get_colors(3),
                    )

                # Process Qt events to keep the UI responsive
                QApplication.processEvents()

                # Give other tasks a chance to run
                await asyncio.sleep(0.01)
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            raise e
        finally:
            # Ensure cleanup always happens
            self.cleanup_and_exit()
