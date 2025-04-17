"""
Visualization utilities for the posture detector.
"""
import cv2

from config.settings import COLORS, FONT_FACE, FONT_SCALE, FONT_THICKNESS
from config.settings import PANEL_PADDING, PANEL_OPACITY, TEXT_PADDING


def draw_landmarks(frame, landmarks, color=COLORS['yellow']):
    """
    Draw landmarks on the frame

    Args:
        frame: Image frame to draw on
        landmarks: Dictionary of landmark coordinates
        color: Color to use for drawing
    """
    for name, (x, y) in landmarks.items():
        if x is not None and y is not None:
            cv2.circle(frame, (x, y), 7, color, -1)


def draw_posture_lines(frame, landmarks, color):
    """
    Draw lines connecting landmarks to visualize posture

    Args:
        frame: Image frame to draw on
        landmarks: Dictionary of landmark coordinates
        color: Color to use for drawing
    """
    # Extract coordinates
    l_shldr = landmarks.get('l_shoulder', (None, None))
    l_ear = landmarks.get('l_ear', (None, None))
    l_hip = landmarks.get('l_hip', (None, None))

    # Create vertical reference points
    l_shldr_ref = (l_shldr[0], l_shldr[1] - 100) if all(x is not None for x in l_shldr) else (None, None)
    l_hip_ref = (l_hip[0], l_hip[1] - 100) if all(x is not None for x in l_hip) else (None, None)

    # Draw reference points
    if all(x is not None for x in l_shldr_ref):
        cv2.circle(frame, l_shldr_ref, 7, COLORS['yellow'], -1)
    if all(x is not None for x in l_hip_ref):
        cv2.circle(frame, l_hip_ref, 7, COLORS['yellow'], -1)

    # Draw lines
    line_pairs = [(l_shldr, l_ear), (l_shldr, l_shldr_ref), (l_hip, l_shldr), (l_hip, l_hip_ref)]

    for start, end in line_pairs:
        if all(x is not None for x in start) and all(x is not None for x in end):
            cv2.line(frame, start, end, color, 4)


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
    l_shldr = landmarks.get('l_shoulder')
    l_hip = landmarks.get('l_hip')

    # Display angles next to landmarks
    if l_shldr is not None:
        cv2.putText(frame, str(int(neck_angle)), (l_shldr[0] + 10, l_shldr[1]), FONT_FACE, FONT_SCALE, color,
                    FONT_THICKNESS)

    if l_hip is not None:
        cv2.putText(frame, str(int(torso_angle)), (l_hip[0] + 10, l_hip[1]), FONT_FACE, FONT_SCALE, color,
                    FONT_THICKNESS)


def draw_posture_guidance(frame, analysis_results):
    """
    Draw posture correction guidance on the frame

    Args:
        frame: Image frame to draw on
        analysis_results: Dictionary containing posture analysis results
    """
    h, w = frame.shape[:2]

    # Extract posture issues
    issues = analysis_results.get('issues', [])
    if not issues:
        return frame

    # Create a semi-transparent overlay panel
    panel_width = w // 3
    panel_height = len(issues) * 40 + PANEL_PADDING * 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (w - panel_width - PANEL_PADDING, PANEL_PADDING),
                  (w - PANEL_PADDING, panel_height + PANEL_PADDING), COLORS['dark_blue'], -1)

    # Add title
    cv2.putText(overlay, "Posture Correction Guide:",
                (w - panel_width - PANEL_PADDING + TEXT_PADDING, PANEL_PADDING * 2), FONT_FACE, FONT_SCALE,
                COLORS['white'], FONT_THICKNESS)

    # Add correction instructions
    for i, (issue, correction) in enumerate(issues.items()):
        y_pos = PANEL_PADDING * 2 + 30 + i * 40
        cv2.putText(overlay, f"• {correction}", (w - panel_width - PANEL_PADDING + TEXT_PADDING, y_pos), FONT_FACE,
                    FONT_SCALE, COLORS['white'], FONT_THICKNESS)

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

    # Extract timing information
    good_time = analysis_results.get('good_time', 0)
    bad_time = analysis_results.get('bad_time', 0)

    # Background for the status bar
    cv2.rectangle(frame, (0, h - 40), (w, h), (0, 0, 0), -1)

    # Display posture time
    if good_time > 0:
        time_string = f'Good Posture Time: {round(good_time, 1)}s'
        cv2.putText(frame, time_string, (10, h - 15), FONT_FACE, FONT_SCALE, COLORS['green'], FONT_THICKNESS)
    else:
        time_string = f'Bad Posture Time: {round(bad_time, 1)}s'
        cv2.putText(frame, time_string, (10, h - 15), FONT_FACE, FONT_SCALE, COLORS['red'], FONT_THICKNESS)

    # Display alignment status
    alignment = analysis_results.get('shoulder_offset', 0)
    if alignment < 100:
        align_text = f"Shoulders Aligned ({int(alignment)})"
        align_color = COLORS['green']
    else:
        align_text = f"Shoulders Not Aligned ({int(alignment)})"
        align_color = COLORS['red']

    cv2.putText(frame, align_text, (w - 250, h - 15), FONT_FACE, FONT_SCALE, align_color, FONT_THICKNESS)


def draw_posture_indicator(frame, good_posture):
    """
    Draw a prominent indicator showing if posture is good or bad

    Args:
        frame: Image frame to draw on
        good_posture: Boolean indicating if posture is good
    """
    h, w = frame.shape[:2]

    # Draw indicator in top-left corner
    if good_posture:
        status_text = "GOOD POSTURE"
        color = COLORS['green']
    else:
        status_text = "BAD POSTURE"
        color = COLORS['red']

    # Background for indicator
    cv2.rectangle(frame, (10, 40), (200, 70), (0, 0, 0), -1)
    cv2.putText(frame, status_text, (15, 65), FONT_FACE, FONT_SCALE, color, FONT_THICKNESS)
