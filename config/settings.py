"""
Configuration settings for the posture detector application.
"""

# Camera settings
DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480

# Score thresholds
NECK_SCORE_MAP = {0: 100, 25: 75, 40: 20, 50: 0}

TORSO_SCORE_MAP = {0: 100, 15: 75, 30: 10, 40: 0}

SHOULDERS_SCORE_MAP = {0: 100, 100: 50, 200: 0}

# Warning settings
WARNING_COOLDOWN = 300  # seconds

# Colors in BGR format
COLORS = {
    "blue": (255, 127, 0),
    "red": (50, 50, 255),
    "green": (127, 255, 0),
    "dark_blue": (127, 20, 0),
    "light_green": (127, 233, 100),
    "yellow": (0, 255, 255),
    "pink": (255, 0, 255),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "translucent_black": (0, 0, 0, 128),  # For overlay backgrounds
}

# Font settings
FONT_FACE = 0  # FONT_HERSHEY_SIMPLEX
FONT_THICKNESS = 2
DEFAULT_SENSITIVITY = 75

BODY_COMPONENTS = {
    "neck": {"parameter": "neck_angle", "score": "neck_score"},
    "torso": {"parameter": "torso_angle", "score": "torso_score"},
    "shoulders": {"parameter": "shoulders_offset", "score": "shoulders_score"},
}

ALERT_SLIDING_WINDOW_DURATION = 120  # seconds
SLIDING_WINDOW_DURATION = 30  # seconds

SEND_INTERVAL = 30  # seconds

# GPIO
VIBRATION_PIN = 14

# Instruction panel settings
PANEL_PADDING = 10
PANEL_OPACITY = 0.7
TEXT_PADDING = 5
