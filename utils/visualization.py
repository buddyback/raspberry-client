"""
Visualization utilities for the posture detector.
"""

import os

import cv2
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QIcon, QImage, QPixmap
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from config.settings import BODY_COMPONENTS, COLORS, FONT_FACE, PANEL_OPACITY, PANEL_PADDING, TEXT_PADDING


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


def draw_posture_lines(frame, landmarks, colors):
    """
    Draw lines connecting landmarks to visualize posture

    Args:
        frame: Image frame to draw on
        landmarks: Dictionary of landmark coordinates
        colors: Colors to use for drawing
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
    line_pairs = {
        "neck": {
            "points": (shoulder, ear),
            "color": colors["neck"],
        },
        "shoulders": {
            "points": (shoulder, shoulder_ref),
            "color": colors["shoulders"],
        },
        "torso": {
            "points": (hip, shoulder),
            "color": colors["torso"],
        },
        # "hips": {
        #     "points": (hip, hip_ref),
        #     "color": colors["hips"],
        # },
    }

    for component, data in line_pairs.items():
        points = data["points"]
        cv2.line(frame, points[0], points[1], data["color"], thickness)

    # Draw shoulder line to show alignment if both shoulders are available
    if all(x is not None for x in l_shldr) and all(x is not None for x in r_shldr):
        cv2.line(frame, l_shldr, r_shldr, colors["shoulders"], thickness)


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

    is_head_tilted_back = analysis_results.get("is_head_tilted_back", False)

    # Scale status bar height based on frame size
    status_height = int(h / 12)

    # Background for the status bar
    cv2.rectangle(frame, (0, h - status_height), (w, h), (0, 0, 0), -1)

    # Display posture time
    y_pos = h - int(status_height / 2)

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


class StatusWidget(QWidget):
    def __init__(self, image_path, label_text, score):
        super().__init__()

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Icon
        icon_label = QLabel()
        pixmap = QPixmap(image_path).scaled(
            100, 100, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        icon_label.setPixmap(pixmap)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon_label)

        # Label text
        text_label = QLabel(label_text)
        text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        text_label.setStyleSheet("color: white; font-weight: bold;")
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
        self.progress.setStyleSheet(
            f"""
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
        """
        )

        layout.addWidget(self.progress)
        self.setLayout(layout)


class MainAppController:
    def __init__(self):
        # Main window
        self.window = QMainWindow()
        self.window.setWindowTitle("BuddyBack")
        self.window.setFixedSize(800, 480)
        if os.getenv("HIDE_TITLEBAR", 0) in ["1", "true", "True"]:
            self.window.setWindowFlags(self.window.windowFlags() | Qt.WindowType.FramelessWindowHint)
        # Main layout
        central_widget = QWidget()
        self.main_layout = QVBoxLayout()
        central_widget.setLayout(self.main_layout)
        self.window.setCentralWidget(central_widget)

        # Navigation buttons at the top
        self.button_container = QWidget()
        self.button_bar = QHBoxLayout()
        self.button_container.setLayout(self.button_bar)
        self.main_layout.addWidget(self.button_container)

        self.btn_posture = QPushButton()
        self.btn_webcam = QPushButton()

        # Set icons (replace with actual paths to your icon files)
        self.btn_posture.setIcon(QIcon("images/posture.png"))  # Choose a posture icon
        self.btn_webcam.setIcon(QIcon("images/camera.png"))  # Camera icon

        # Set size and style
        for btn in [self.btn_posture, self.btn_webcam]:
            btn.setIconSize(QSize(32, 32))  # Set icon size
            btn.setFixedSize(48, 48)  # Total button size
            btn.setStyleSheet(
                """
                QPushButton {
                    background-color: white;
                    border: 1px solid #ccc;
                    border-radius: 6px;
                }
                QPushButton:hover {
                    background-color: #f0f0f0;
                }
            """
            )

        self.button_bar.addWidget(self.btn_posture)
        self.button_bar.addWidget(self.btn_webcam)

        # Stack of views
        self.stacked_widget = QStackedWidget()
        self.main_layout.addWidget(self.stacked_widget)

        self.inactive_view = MainScreen(controller=self)  # "Session is not active" view
        self.active_view = PostureWindow()  # Posture status view
        self.webcam_view = WebcamWindow()

        self.stacked_widget.addWidget(self.inactive_view)  # index 0
        self.stacked_widget.addWidget(self.active_view)  # index 1
        self.stacked_widget.addWidget(self.webcam_view)  # index 2

        # Connect buttons
        self.btn_posture.clicked.connect(lambda: self.stacked_widget.setCurrentWidget(self.active_view))
        self.btn_webcam.clicked.connect(lambda: self.stacked_widget.setCurrentWidget(self.webcam_view))

        # For compatibility with PostureDetector's existing references
        self.main_screen = self.window  # app_controller.main_screen will refer to the QMainWindow
        self.posture_window = (
            self.active_view
        )  # app_controller.posture_window refers to the active_view for score updates
        self.window.stacked_widget = self.stacked_widget

    def start(self):
        """Shows the main window and sets the initial view to inactive."""
        self.button_container.hide()  # Hide the button bar initially
        self.window.show()
        self.stacked_widget.setCurrentWidget(self.inactive_view)  # Default to inactive view

    def activate_session(self):
        """Switches the view to the active session (posture tracking)."""
        self.stacked_widget.setCurrentWidget(self.active_view)
        self.button_container.show()
        print("UI: Switched to active_view with webcam and posture metrics")

    def end_session(self):
        """Switches the view to the inactive session (main screen)."""
        self.stacked_widget.setCurrentWidget(self.inactive_view)
        print("UI: Switched to inactive_view (MainScreen)")
        self.button_container.hide()


class MainScreen(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.setWindowTitle("BuddyBack")
        self.setStyleSheet("background-color: black;")
        self.setFixedSize(800, 480)

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


class WebcamWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(800, 400)

        # Create layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Create label to display the webcam feed
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image_label)

        # Current frame and analysis results
        self.current_frame = None
        self.analysis_results = {}
        self.landmarks = {}
        self.neck_angle = 0
        self.torso_angle = 0
        self.good_posture = False

    def update_frame(self, frame):
        """
        Update the widget with a new frame and visualization data

        Args:
            frame: The video frame (numpy array)
            landmarks: Dictionary of detected landmarks
            analysis_results: Dictionary of posture analysis results
        """
        if frame is None:
            return

        # Store data for visualization
        self.current_frame = frame.copy()

        # Convert to Qt format and display
        self._display_frame()

    def _display_frame(self):
        """Convert the processed frame to Qt format and display it"""
        if self.current_frame is None:
            return

        # Convert the frame to RGB (from BGR)
        rgb_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)

        # Create QImage from the frame
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        # Scale the image to fit the label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(qt_image)
        pixmap = pixmap.scaled(
            self.image_label.width(),
            self.image_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        # Display the image
        self.image_label.setPixmap(pixmap)

    def resizeEvent(self, event):
        """Handle resize events to update the displayed frame"""
        super().resizeEvent(event)
        if self.current_frame is not None:
            self._display_frame()


class PostureWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Posture Status")
        self.setFixedSize(800, 480)
        self.issues = {}

        # Layout setup
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)

        # New status widget for webcam alert or results summary
        self.status_widget = QLabel("Please fix webcam placement")
        self.status_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_widget.setStyleSheet(
            """
            font-size: 18px; 
            font-weight: bold; 
            color: red; 
            background-color: #f0f0f0;
            padding: 10px;
            border-radius: 5px;
            margin: 10px;
        """
        )
        main_layout.addWidget(self.status_widget)

        # Container for posture components
        self.posture_container = QWidget()
        posture_layout = QVBoxLayout(self.posture_container)
        posture_layout.setContentsMargins(0, 0, 0, 0)

        # Torso widget
        self.torso_widget = StatusWidget("images/icons/red-torso.png", "Torso", 0)
        self.torso_widget.setFixedSize(180, 180)
        self.torso_widget.setCursor(Qt.CursorShape.PointingHandCursor)
        self.torso_widget.mousePressEvent = lambda event: self.handle_widget_click("torso")
        posture_layout.addWidget(self.torso_widget, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Shoulders and neck
        lower_row = QHBoxLayout()
        lower_row.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.shoulders_widget = StatusWidget("images/icons/red-shoulders.png", "Shoulders", 0)
        self.shoulders_widget.setCursor(Qt.CursorShape.PointingHandCursor)
        self.shoulders_widget.mousePressEvent = lambda event: self.handle_widget_click("shoulders")

        self.neck_widget = StatusWidget("images/icons/red-neck.png", "Neck", 0)
        self.neck_widget.setCursor(Qt.CursorShape.PointingHandCursor)
        self.neck_widget.mousePressEvent = lambda event: self.handle_widget_click("neck")

        lower_row.addWidget(self.shoulders_widget)
        lower_row.addSpacing(20)
        lower_row.addWidget(self.neck_widget)

        posture_layout.addLayout(lower_row)
        main_layout.addWidget(self.posture_container)

        # Initially hide posture container
        self.posture_container.hide()

        self.setLayout(main_layout)

    def update_results(self, results):
        if not results or not results.get("scores"):
            self.status_widget.show()
            # No valid results - show webcam message
            self.posture_container.hide()
            return

        # Valid results - show posture container and update status
        self.posture_container.show()
        self.status_widget.hide()

        # Update scores and status widgets
        if scores := results.get("scores"):
            self.torso_widget.progress.setValue(scores.get(BODY_COMPONENTS["torso"]["score"], 0))
            self.shoulders_widget.progress.setValue(scores.get(BODY_COMPONENTS["shoulders"]["score"], 0))
            self.neck_widget.progress.setValue(scores.get(BODY_COMPONENTS["neck"]["score"], 0))

            self.update_progress_style(self.torso_widget.progress, scores.get(BODY_COMPONENTS["torso"]["score"], 0))
            self.update_progress_style(
                self.shoulders_widget.progress, scores.get(BODY_COMPONENTS["shoulders"]["score"], 0)
            )
            self.update_progress_style(self.neck_widget.progress, scores.get(BODY_COMPONENTS["neck"]["score"], 0))

        if issues := results.get("issues"):
            self.issues["shoulders"] = issues.get("shoulders", None)
            self.issues["neck"] = issues.get("neck", None)
            self.issues["torso"] = issues.get("torso", None)

    # Other methods remain unchanged

    def handle_widget_click(self, component):
        if issue := self.issues.get(component):
            alert = QMessageBox()
            alert.setWindowTitle(f"Informazioni {component.capitalize()}")
            alert.setText(issue)
            alert.setIcon(QMessageBox.Icon.Information)
            alert.exec()

    def update_progress_style(self, progress_bar, score):
        if score > 60:
            color = "green"
        elif score > 30:
            color = "yellow"
        else:
            color = "red"

        progress_bar.setStyleSheet(
            f"""
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
        """
        )
