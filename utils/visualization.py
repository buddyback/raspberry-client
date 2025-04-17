"""
Visualization utilities for the posture detector.
"""
import cv2

from config.settings import COLORS, FONT_FACE
from config.settings import PANEL_PADDING, PANEL_OPACITY, TEXT_PADDING


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
    l_shldr = landmarks.get('l_shoulder', (None, None))
    l_ear = landmarks.get('l_ear', (None, None))
    l_hip = landmarks.get('l_hip', (None, None))

    # Set line thickness based on frame size
    thickness = max(2, int(frame.shape[1] / 320))

    # Create vertical reference points - adjust vertical distance based on frame size
    vert_offset = int(frame.shape[0] / 5)
    l_shldr_ref = (l_shldr[0], l_shldr[1] - vert_offset) if all(x is not None for x in l_shldr) else (None, None)
    l_hip_ref = (l_hip[0], l_hip[1] - vert_offset) if all(x is not None for x in l_hip) else (None, None)

    # Draw reference points
    if all(x is not None for x in l_shldr_ref):
        radius = max(3, int(frame.shape[1] / 100))
        cv2.circle(frame, l_shldr_ref, radius, COLORS['yellow'], -1)
    if all(x is not None for x in l_hip_ref):
        radius = max(3, int(frame.shape[1] / 100))
        cv2.circle(frame, l_hip_ref, radius, COLORS['yellow'], -1)

    # Draw lines
    line_pairs = [(l_shldr, l_ear), (l_shldr, l_shldr_ref), (l_hip, l_shldr), (l_hip, l_hip_ref)]

    for start, end in line_pairs:
        if all(x is not None for x in start) and all(x is not None for x in end):
            cv2.line(frame, start, end, color, thickness)


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

    l_shldr = landmarks.get('l_shoulder')
    l_hip = landmarks.get('l_hip')

    # Display angles next to landmarks with proper positioning
    if l_shldr is not None:
        # Ensure text stays within frame boundaries
        x_pos = min(l_shldr[0] + 10, w - 40)
        cv2.putText(frame, str(int(neck_angle)), (x_pos, l_shldr[1]), FONT_FACE, font_scale, color, thickness)

    if l_hip is not None:
        x_pos = min(l_hip[0] + 10, w - 40)
        cv2.putText(frame, str(int(torso_angle)), (x_pos, l_hip[1]), FONT_FACE, font_scale, color, thickness)


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
    issues = analysis_results.get('issues', [])
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
    cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), COLORS['dark_blue'], -1)

    # Add title
    title_y = panel_y1 + PANEL_PADDING + int(20 * font_scale)
    cv2.putText(overlay, "Posture Correction Guide:", (panel_x1 + TEXT_PADDING, title_y), FONT_FACE, font_scale,
                COLORS['white'], thickness)

    # Add correction instructions
    for i, (issue, correction) in enumerate(issues.items()):
        y_pos = title_y + int((i + 1) * line_height)
        if y_pos >= panel_y2 - 5:  # Stop if we're going beyond panel
            break
        cv2.putText(overlay, f"• {correction}", (panel_x1 + TEXT_PADDING, y_pos), FONT_FACE, font_scale * 0.9,
                    COLORS['white'], thickness)

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
    good_time = analysis_results.get('good_time', 0)
    bad_time = analysis_results.get('bad_time', 0)

    # Scale status bar height based on frame size
    status_height = int(h / 12)

    # Background for the status bar
    cv2.rectangle(frame, (0, h - status_height), (w, h), (0, 0, 0), -1)

    # Display posture time
    y_pos = h - int(status_height / 2)
    if good_time > 0:
        time_string = f'Good Posture Time: {round(good_time, 1)}s'
        cv2.putText(frame, time_string, (10, y_pos), FONT_FACE, font_scale, COLORS['green'], thickness)
    else:
        time_string = f'Bad Posture Time: {round(bad_time, 1)}s'
        cv2.putText(frame, time_string, (10, y_pos), FONT_FACE, font_scale, COLORS['red'], thickness)

    # Display alignment status
    alignment = analysis_results.get('shoulder_offset', 0)
    if alignment < 100:
        align_text = f"Shoulders Aligned ({int(alignment)})"
        align_color = COLORS['green']
    else:
        align_text = f"Shoulders Not Aligned ({int(alignment)})"
        align_color = COLORS['red']

    # Position text on right side, accounting for text length
    text_size = cv2.getTextSize(align_text, FONT_FACE, font_scale, thickness)[0]
    x_pos = w - text_size[0] - 10
    cv2.putText(frame, align_text, (max(10, x_pos), y_pos), FONT_FACE, font_scale, align_color, thickness)


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
        color = COLORS['green']
    else:
        status_text = "BAD POSTURE"
        color = COLORS['red']

    # Calculate text size to make proper background
    text_size = cv2.getTextSize(status_text, FONT_FACE, font_scale, thickness)[0]

    # Background for indicator
    padding = int(5 * (w / 640))
    cv2.rectangle(frame, (10, 40), (10 + text_size[0] + padding * 2, 40 + text_size[1] + padding), (0, 0, 0), -1)

    cv2.putText(frame, status_text, (10 + padding, 40 + text_size[1]), FONT_FACE, font_scale, color, thickness)
