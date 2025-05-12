"""
Visualization utilities for the posture detector.
"""

import cv2

from config.settings import COLORS, FONT_FACE, PANEL_OPACITY, PANEL_PADDING, TEXT_PADDING


def get_optimal_font_scale(frame_width):
    """
    Calculate optimal font scale based on frame width

    Args:
        frame_width: Width of the frame

    Returns:
        Float: Font scale suitable for the frame size
    """
    base_width = 640.0  # Base width for reference
    base_scale = 0.7  # Base font scale

    # Scale font proportionally to frame width
    return max(0.5, base_scale * (frame_width / base_width))


def draw_landmarks(frame, landmarks, color=COLORS["yellow"]):
    """
    Draw landmarks on the frame

    Args:
        frame: Image frame to draw on
        landmarks: Dictionary of landmark coordinates
        color: Color to use for drawing
    """
    for name, value in landmarks.items():
        # Only process coordinate pairs (x,y), skip metadata fields
        if name in ["primary_ear", "l_ear_visibility", "r_ear_visibility"]:
            continue

        # Handle coordinate pairs
        if isinstance(value, tuple) and len(value) == 2:
            x, y = value
            if x is not None and y is not None:
                radius = max(3, int(frame.shape[1] / 100))  # Scale circle radius to frame
                cv2.circle(frame, (x, y), radius, color, -1)


def draw_posture_lines(frame, landmarks, color):
    """
    Draw lines connecting landmarks to visualize posture

    Args:
        frame: Image frame to draw on
        landmarks: Dictionary of landmark coordinates
        color: Color to use for drawing
    """
    # Extract coordinates
    l_shldr = landmarks.get("l_shoulder", (None, None))
    r_shldr = landmarks.get("r_shoulder", (None, None))
    l_ear = landmarks.get("l_ear", (None, None))
    r_ear = landmarks.get("r_ear", (None, None))
    l_hip = landmarks.get("l_hip", (None, None))
    r_hip = landmarks.get("r_hip", (None, None))

    # Get the primary ear (more visible) for drawing lines
    primary_ear = landmarks.get("primary_ear", "left")

    # Set line thickness based on frame size
    thickness = max(2, int(frame.shape[1] / 320))

    # Create vertical reference points - adjust vertical distance based on frame size
    vert_offset = int(frame.shape[0] / 5)

    # Choose which side to use for visualization based on visibility
    ear = None
    shoulder = None
    hip = None

    if primary_ear == "left":
        # Use left side if available
        ear = l_ear if l_ear[0] is not None else r_ear
        shoulder = l_shldr if l_shldr[0] is not None else r_shldr
        hip = l_hip if l_hip[0] is not None else r_hip
    else:
        # Use right side if available
        ear = r_ear if r_ear[0] is not None else l_ear
        shoulder = r_shldr if r_shldr[0] is not None else l_shldr
        hip = r_hip if r_hip[0] is not None else l_hip

    # Skip if we couldn't find valid landmarks
    if None in [ear, shoulder, hip]:
        return

    # Skip if any of the landmarks have None coordinates
    if None in ear or None in shoulder or None in hip:
        return

    # Create reference points for the selected side
    shoulder_ref = (shoulder[0], shoulder[1] - vert_offset)
    hip_ref = (hip[0], hip[1] - vert_offset)

    # Draw reference points
    radius = max(3, int(frame.shape[1] / 100))
    cv2.circle(frame, shoulder_ref, radius, COLORS["yellow"], -1)
    cv2.circle(frame, hip_ref, radius, COLORS["yellow"], -1)

    # Draw lines
    line_pairs = [
        (shoulder, ear),
        (shoulder, shoulder_ref),
        (hip, shoulder),
        (hip, hip_ref),
    ]

    for start, end in line_pairs:
        cv2.line(frame, start, end, color, thickness)

    # Draw shoulder line to show alignment if both shoulders are available
    if all(x is not None for x in l_shldr) and all(x is not None for x in r_shldr):
        cv2.line(frame, l_shldr, r_shldr, color, thickness)


def draw_angle_text(frame, landmarks, neck_angle, torso_angle, color):
    """
    Draw angle measurements on the frame

    Args:
        frame: Image frame to draw on
        landmarks: Dictionary of landmark coordinates
        neck_angle: Calculated neck angle
        torso_angle: Calculated torso angle
        color: Color to use for drawing
    """
    h, w = frame.shape[:2]
    font_scale = get_optimal_font_scale(w)
    thickness = max(1, int(w / 640))

    # Get the primary ear (more visible) for text positioning
    primary_ear = landmarks.get("primary_ear", "left")

    # Choose which side to use for visualization based on visibility
    if primary_ear == "left":
        shoulder = landmarks.get("l_shoulder")
        hip = landmarks.get("l_hip")
    else:
        shoulder = landmarks.get("r_shoulder")
        hip = landmarks.get("r_hip")

    # If preferred side isn't available, try the other side
    if shoulder is None:
        shoulder = landmarks.get("r_shoulder" if primary_ear == "left" else "l_shoulder")

    if hip is None:
        hip = landmarks.get("r_hip" if primary_ear == "left" else "l_hip")

    # Display angles next to landmarks with proper positioning
    if shoulder is not None and all(x is not None for x in shoulder):
        # Ensure text stays within frame boundaries
        x_pos = min(shoulder[0] + 10, w - 40)
        cv2.putText(
            frame,
            f"{int(neck_angle)}°",
            (x_pos, shoulder[1]),
            FONT_FACE,
            font_scale,
            color,
            thickness,
        )

    if hip is not None and all(x is not None for x in hip):
        x_pos = min(hip[0] + 10, w - 40)
        cv2.putText(
            frame,
            f"{int(torso_angle)}°",
            (x_pos, hip[1]),
            FONT_FACE,
            font_scale,
            color,
            thickness,
        )

    # Display relative angle if head is tilted back
    is_head_tilted_back = landmarks.get("is_head_tilted_back", False)

    if is_head_tilted_back and shoulder is not None and hip is not None:
        relative_angle = abs(neck_angle - torso_angle)
        midpoint_y = (shoulder[1] + hip[1]) // 2
        midpoint_x = (shoulder[0] + hip[0]) // 2
        x_pos = min(midpoint_x + 10, w - 120)
        cv2.putText(
            frame,
            f"Rel: {int(relative_angle)}°",
            (x_pos, midpoint_y),
            FONT_FACE,
            font_scale,
            COLORS["yellow"],
            thickness,
        )


def draw_posture_guidance(frame, analysis_results):
    """
    Draw posture correction guidance on the frame

    Args:
        frame: Image frame to draw on
        analysis_results: Dictionary containing posture analysis results
    """
    h, w = frame.shape[:2]
    font_scale = get_optimal_font_scale(w)
    thickness = max(1, int(w / 640))

    # Extract posture issues
    issues = analysis_results.get("issues", [])
    if not issues:
        return frame

    # Create a semi-transparent overlay panel - scale with frame size
    panel_width = min(w // 2, 400)
    line_height = int(30 * (w / 640))
    panel_height = len(issues) * line_height + PANEL_PADDING * 2

    # Ensure panel fits within frame
    panel_x1 = max(0, w - panel_width - PANEL_PADDING)
    panel_x2 = min(w - PANEL_PADDING, w)
    panel_y1 = PANEL_PADDING
    panel_y2 = min(h - PANEL_PADDING, panel_height + PANEL_PADDING)

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), COLORS["dark_blue"], -1)

    # Add title
    title_y = panel_y1 + PANEL_PADDING + int(20 * font_scale)
    cv2.putText(
        overlay,
        "Posture Correction Guide:",
        (panel_x1 + TEXT_PADDING, title_y),
        FONT_FACE,
        font_scale,
        COLORS["white"],
        thickness,
    )

    # Add correction instructions
    for i, (issue, correction) in enumerate(issues.items()):
        y_pos = title_y + int((i + 1) * line_height)
        if y_pos >= panel_y2 - 5:  # Stop if we're going beyond panel
            break
        cv2.putText(
            overlay,
            f"• {correction}",
            (panel_x1 + TEXT_PADDING, y_pos),
            FONT_FACE,
            font_scale * 0.9,
            COLORS["white"],
            thickness,
        )

    # Apply the overlay with transparency
    cv2.addWeighted(overlay, PANEL_OPACITY, frame, 1 - PANEL_OPACITY, 0, frame)

    return frame


def draw_status_bar(frame, analysis_results):
    """
    Draw status information at the bottom of the frame

    Args:
        frame: Image frame to draw on
        analysis_results: Dictionary containing posture analysis results
    """
    h, w = frame.shape[:2]
    font_scale = get_optimal_font_scale(w)
    thickness = max(1, int(w / 640))

    # Extract timing information
    good_time = analysis_results.get("good_time", 0)
    bad_time = analysis_results.get("bad_time", 0)
    is_head_tilted_back = analysis_results.get("is_head_tilted_back", False)

    # Scale status bar height based on frame size
    status_height = int(h / 12)

    # Background for the status bar
    cv2.rectangle(frame, (0, h - status_height), (w, h), (0, 0, 0), -1)

    # Display posture time
    y_pos = h - int(status_height / 2)
    if good_time > 0:
        time_string = f"Good Posture Time: {round(good_time, 1)}s"
        cv2.putText(
            frame,
            time_string,
            (10, y_pos),
            FONT_FACE,
            font_scale,
            COLORS["green"],
            thickness,
        )
    else:
        time_string = f"Bad Posture Time: {round(bad_time, 1)}s"
        cv2.putText(
            frame,
            time_string,
            (10, y_pos),
            FONT_FACE,
            font_scale,
            COLORS["red"],
            thickness,
        )

    # Display alignment status
    alignment = analysis_results.get("shoulders_offset", 0)
    if alignment < 100:
        align_text = f"Shoulders Aligned ({int(alignment)})"
        align_color = COLORS["green"]
    else:
        align_text = f"Shoulders Not Aligned ({int(alignment)})"
        align_color = COLORS["red"]

    # Position text on right side, accounting for text length
    text_size = cv2.getTextSize(align_text, FONT_FACE, font_scale, thickness)[0]
    x_pos = w - text_size[0] - 10
    cv2.putText(
        frame,
        align_text,
        (max(10, x_pos), y_pos),
        FONT_FACE,
        font_scale,
        align_color,
        thickness,
    )

    # Display webcam position at the bottom-center
    webcam_pos = analysis_results.get("webcam_position", "unknown")

    # Create status text with head tilt information
    status_text = "HEAD BACK" if is_head_tilted_back else ""

    if webcam_pos != "unknown":
        if status_text:
            pos_text = f"Webcam: {webcam_pos.upper()} | {status_text}"
        else:
            pos_text = f"Webcam position: {webcam_pos.upper()}"

        # Calculate text position to center it
        text_size = cv2.getTextSize(pos_text, FONT_FACE, font_scale * 0.8, thickness)[0]
        x_pos = (w - text_size[0]) // 2
        y_pos_webcam = h - int(status_height / 4)  # Position below the main status text
        cv2.putText(
            frame,
            pos_text,
            (x_pos, y_pos_webcam),
            FONT_FACE,
            font_scale * 0.8,
            COLORS["yellow"],
            thickness,
        )


def draw_posture_indicator(frame, good_posture):
    """
    Draw a prominent indicator showing if posture is good or bad

    Args:
        frame: Image frame to draw on
        good_posture: Boolean indicating if posture is good
    """
    h, w = frame.shape[:2]
    font_scale = get_optimal_font_scale(w)
    thickness = max(1, int(w / 640))

    # Draw indicator in top-left corner
    if good_posture:
        status_text = "GOOD POSTURE"
        color = COLORS["green"]
    else:
        status_text = "BAD POSTURE"
        color = COLORS["red"]

    # Calculate text size to make proper background
    text_size = cv2.getTextSize(status_text, FONT_FACE, font_scale, thickness)[0]

    # Background for indicator
    padding = int(5 * (w / 640))
    cv2.rectangle(
        frame,
        (10, 40),
        (10 + text_size[0] + padding * 2, 40 + text_size[1] + padding),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        frame,
        status_text,
        (10 + padding, 40 + text_size[1]),
        FONT_FACE,
        font_scale,
        color,
        thickness,
    )

import sys
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QProgressBar
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt

class StatusWidget(QWidget):
    def __init__(self, image_path, label_text, score):
        super().__init__()

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Icon
        icon_label = QLabel()
        pixmap = QPixmap(image_path).scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        icon_label.setPixmap(pixmap)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon_label)

        # Label text
        text_label = QLabel(label_text)
        text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        text_label.setStyleSheet("color: black; font-weight: bold;")
        layout.addWidget(text_label)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setFixedWidth(100)
        self.progress.setValue(score)
        self.progress.setTextVisible(True)
        self.progress.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Determine color based on score
        if score > 60:
            color = "green"
        elif score > 30:
            color = "yellow"
        else:
            color = "red"

        # Stylesheet with black border and text
        self.progress.setStyleSheet(f"""
            QProgressBar {{
                border: 2px solid black;
                border-radius: 5px;
                text-align: center;
                color: black;
                background-color: #f0f0f0;
            }}
            QProgressBar::chunk {{
                background-color: {color};
            }}
        """)

        layout.addWidget(self.progress)
        self.setLayout(layout)

class MainAppController:
    def __init__(self):
        self.main_screen = MainScreen(controller=self)
        self.posture_window = PostureWindow()

    def start(self):
        self.main_screen.show()

    def activate_session(self):
        self.main_screen.hide()
        self.posture_window.show()

    def end_session(self):
        self.posture_window.hide()
        self.main_screen.show()

class MainScreen(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.setWindowTitle("BuddyBack")
        self.setStyleSheet("background-color: black;")
        self.setFixedSize(400, 400)

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        logo = QLabel()
        pixmap = QPixmap("../images/logo.png").scaled(150, 150, Qt.AspectRatioMode.KeepAspectRatio)
        logo.setPixmap(pixmap)
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)

        label = QLabel("Session is not active")
        label.setStyleSheet("color: white; font-size: 18px;")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(logo)
        layout.addWidget(label)
        self.setLayout(layout)


class PostureWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Posture Status")
        self.setStyleSheet("background-color: white;")
        self.setFixedSize(400, 400)

        # Layout setup
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)

        # Torso widget
        self.torso_widget = StatusWidget("../images/torso.png", "Torso", 0)
        self.torso_widget.setFixedSize(150, 150)
        main_layout.addWidget(self.torso_widget, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Shoulders and neck
        lower_row = QHBoxLayout()
        lower_row.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.shoulders_widget = StatusWidget("../images/shoulders.png", "Shoulders", 0)
        self.neck_widget = StatusWidget("../images/neck.png", "Neck", 0)

        lower_row.addWidget(self.shoulders_widget)
        lower_row.addSpacing(20)
        lower_row.addWidget(self.neck_widget)

        main_layout.addLayout(lower_row)
        self.setLayout(main_layout)

        # Posture detector

    def update_scores(self, scores):
        self.torso_widget.progress.setValue(scores.get("torso_score", 0))
        self.shoulders_widget.progress.setValue(scores.get("shoulders_score", 0))
        self.neck_widget.progress.setValue(scores.get("neck_score", 0))

        # Optional: dynamically update bar color
        self.update_progress_style(self.torso_widget.progress, scores.get("torso_score", 0))
        self.update_progress_style(self.shoulders_widget.progress, scores.get("shoulders_score", 0))
        self.update_progress_style(self.neck_widget.progress, scores.get("neck_score", 0))

    def update_progress_style(self, progress_bar, score):
        if score > 60:
            color = "green"
        elif score > 30:
            color = "yellow"
        else:
            color = "red"

        progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 2px solid black;
                border-radius: 5px;
                text-align: center;
                color: black;
                background-color: #f0f0f0;
            }}
            QProgressBar::chunk {{
                background-color: {color};
            }}
        """)
