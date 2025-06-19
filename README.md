# Posture Detection System

This application uses computer vision to detect and provide feedback on your sitting posture using a webcam.

# [Watch video presentation](https://www.youtube.com/watch?v=6V_hrUXwh0k&feature=youtu.be)

## Features

- Real-time posture monitoring
- Visual feedback on neck and torso angles
- Guidance for correcting poor posture
- Resizable camera frame
- Audible warnings for prolonged poor posture

## Requirements

- Python 3.6+
- OpenCV
- MediaPipe
- Webcam

## Installation

1. Clone this repository:

```
git clone https://github.com/yourusername/posture-detector.git
cd posture-detector
```

2. Install the required dependencies:

```
pip install opencv-python mediapipe
```

## Usage

Run the application with default settings:

```
python -m main
```

### Command Line Options

- `--width`: Set camera frame width (default: 640)
- `--height`: Set camera frame height (default: 480)
- `--camera`: Select camera index (default: 0)
- `--no-guidance`: Disable posture correction guidance

Example:

```
python -m main --width 800 --height 600 --camera 1
```

### Controls

- Press `q` to quit the application
- Press `r` to enter resize mode
    - In resize mode, use arrow keys to adjust the frame size
    - Press `r` again to exit resize mode

## How It Works

The application uses MediaPipe's pose detection to identify key body landmarks:

- Shoulders
- Ears
- Hips

It then calculates:

1. Neck angle (between shoulder and ear)
2. Torso angle (between hip and shoulder)

Good posture is defined as:

- Neck angle < 40°
- Torso angle < 10°

When poor posture is detected, the application provides visual guidance on how to correct it.

## Project Structure

```
posture_detector/
├── __init__.py
├── main.py
├── detector/
│   ├── __init__.py
│   ├── posture_detector.py
│   └── posture_analyzer.py
├── utils/
│   ├── __init__.py
│   ├── visualization.py
│   └── camera.py
└── config/
    ├── __init__.py
    └── settings.py
```

## Customization

You can adjust posture thresholds and other settings in `config/settings.py`.