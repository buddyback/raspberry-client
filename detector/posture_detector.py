"""
Main posture detection module that integrates camera capture and posture analysis.
"""
import signal
import sys
import time

import cv2
import mediapipe as mp

from config.settings import (
    WARNING_TIME_THRESHOLD, WARNING_COOLDOWN, COLORS, FONT_FACE, CAMERA_FPS
)
from detector.posture_analyzer import PostureAnalyzer
from utils.visualization import (
    draw_landmarks, draw_posture_lines, draw_angle_text,
    draw_posture_guidance, draw_status_bar, draw_posture_indicator,
    get_optimal_font_scale
)


class PostureDetector:
    """Main class for posture detection"""

    def __init__(self, camera_manager, show_guidance=True):
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
        self.pose = self.mp_pose.Pose(model_complexity=2)

        # Initialize posture analyzer
        self.analyzer = PostureAnalyzer()

        # Warning timer
        self.last_warning_time = 0

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.cleanup_and_exit)
        signal.signal(signal.SIGTERM, self.cleanup_and_exit)

        # Window resize control
        self.resize_mode = False
        self.window_name = 'Posture Detection'

    def cleanup_and_exit(self, signum=None, frame=None):
        """Clean up resources and exit the program"""
        print("\nShutting down posture detector...")
        if self.camera_manager.is_open():
            self.camera_manager.release()
        cv2.destroyAllWindows()
        sys.exit(0)

    def send_warning(self):
        """Send alert when bad posture is detected for too long"""
        current_time = time.time()
        if current_time - self.last_warning_time > WARNING_COOLDOWN:
            print("\a")  # System beep
            print("WARNING: Bad posture detected! Please correct your posture.")
            self.last_warning_time = current_time

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
            landmarks['l_shoulder'] = (
                int(lm.landmark[lmPose.LEFT_SHOULDER].x * frame_width),
                int(lm.landmark[lmPose.LEFT_SHOULDER].y * frame_height)
            )

            # Right shoulder
            landmarks['r_shoulder'] = (
                int(lm.landmark[lmPose.RIGHT_SHOULDER].x * frame_width),
                int(lm.landmark[lmPose.RIGHT_SHOULDER].y * frame_height)
            )

            # Left ear
            landmarks['l_ear'] = (
                int(lm.landmark[lmPose.LEFT_EAR].x * frame_width),
                int(lm.landmark[lmPose.LEFT_EAR].y * frame_height)
            )

            # Left hip
            landmarks['l_hip'] = (
                int(lm.landmark[lmPose.LEFT_HIP].x * frame_width),
                int(lm.landmark[lmPose.LEFT_HIP].y * frame_height)
            )

            # Right hip
            landmarks['r_hip'] = (
                int(lm.landmark[lmPose.RIGHT_HIP].x * frame_width),
                int(lm.landmark[lmPose.RIGHT_HIP].y * frame_height)
            )

            return landmarks

        except Exception as e:
            print(f"Error extracting landmarks: {e}")
            return {}

    def process_frame(self, frame):
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
            cv2.putText(frame, "No pose detected", (10, 30), FONT_FACE,
                        font_scale, COLORS['red'], thickness)
            return frame

        # Extract landmarks
        landmarks = self.extract_landmarks(result.pose_landmarks, w, h)

        # If landmarks extraction failed, return original frame
        if not landmarks:
            cv2.putText(frame, "Landmark extraction failed", (10, 30),
                        FONT_FACE, font_scale, COLORS['red'], thickness)
            return frame

        # Analyze posture
        analysis_results = self.analyzer.analyze_posture(landmarks)

        # Draw landmarks
        draw_landmarks(frame, landmarks)

        # Update frame counters based on posture quality
        if analysis_results['good_posture']:
            self.bad_frames = 0
            self.good_frames += 1

            # Draw lines with good posture color
            draw_posture_lines(frame, landmarks, COLORS['green'])
        else:
            self.good_frames = 0
            self.bad_frames += 1

            # Draw lines with bad posture color
            draw_posture_lines(frame, landmarks, COLORS['red'])

            # Check if warning should be sent
            bad_time = self.bad_frames / CAMERA_FPS
            if bad_time > WARNING_TIME_THRESHOLD:
                self.send_warning()

        # Calculate timing values for UI
        analysis_results['good_time'] = (1 / CAMERA_FPS) * self.good_frames
        analysis_results['bad_time'] = (1 / CAMERA_FPS) * self.bad_frames

        # Draw posture angles
        color = COLORS['light_green'] if analysis_results['good_posture'] else COLORS['red']
        draw_angle_text(
            frame, landmarks,
            analysis_results['neck_angle'],
            analysis_results['torso_angle'],
            color
        )

        # Add main angle text at top
        angle_text = f'Neck: {int(analysis_results["neck_angle"])}°  Torso: {int(analysis_results["torso_angle"])}°'
        cv2.putText(frame, angle_text, (10, 30), FONT_FACE, font_scale, color, thickness)

        # Draw posture indicator (GOOD/BAD)
        draw_posture_indicator(frame, analysis_results['good_posture'])

        # Draw status bar
        draw_status_bar(frame, analysis_results)

        # Draw posture correction guidance if enabled
        if self.show_guidance and not analysis_results['good_posture']:
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
        if key == ord('q'):
            return False
        elif key == ord('r'):
            # Toggle resize mode
            self.resize_mode = not self.resize_mode
            mode_text = "ON" if self.resize_mode else "OFF"
            print(f"Resize mode: {mode_text}")
            if self.resize_mode:
                print("Use arrow keys to resize the window. Press 'r' again to exit resize mode.")
        elif key == ord('f'):
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
            width, height = self.camera_manager.frame_width, self.camera_manager.frame_height

            # Adjust dimensions based on arrow keys
            if key == 82 or key == ord('w'):  # Up arrow or 'w'
                height = int(height * 1.1)
            elif key == 84 or key == ord('s'):  # Down arrow or 's'
                height = int(height * 0.9)
            elif key == 83 or key == ord('d'):  # Right arrow or 'd'
                width = int(width * 1.1)
            elif key == 81 or key == ord('a'):  # Left arrow or 'a'
                width = int(width * 0.9)

            # Apply new dimensions if changed
            if width != self.camera_manager.frame_width or height != self.camera_manager.frame_height:
                actual_width, actual_height = self.camera_manager.resize_frame(width, height)
                print(f"Resized camera frame to {actual_width}x{actual_height}")

        return True

    def run(self):
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
                # Read frame from webcam
                success, frame = self.camera_manager.read_frame()

                if not success:
                    print("Error: Failed to capture image from webcam")
                    break

                # Process the frame
                processed_frame = self.process_frame(frame)

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

        except Exception as e:
            print(f"Error occurred: {str(e)}")
        finally:
            # Ensure cleanup always happens
            self.cleanup_and_exit()
