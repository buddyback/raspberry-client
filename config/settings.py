"""
Configuration settings for the posture detector application.
"""

# Camera settings
DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Posture thresholds
NECK_ANGLE_THRESHOLD = 15
TORSO_ANGLE_THRESHOLD = 10
MAX_SHOULDERS_DISTANCE = 120

# Warning settings
WARNING_TIME_THRESHOLD = 60  # seconds
WARNING_COOLDOWN = 60  # seconds

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

BODY_COMPONENTS = {
    "neck": {
        "parameter": "neck_angle",
        "reference": NECK_ANGLE_THRESHOLD,
    },
    "torso": {
        "parameter": "torso_angle",
        "reference": NECK_ANGLE_THRESHOLD,
    },
    "shoulders": {
        "parameter": "shoulder_offset",
        "reference": MAX_SHOULDERS_DISTANCE,
    },
}

SEND_INTERVAL = 60  # seconds

# Instruction panel settings
PANEL_PADDING = 10
PANEL_OPACITY = 0.7
TEXT_PADDING = 5
