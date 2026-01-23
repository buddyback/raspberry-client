#!/usr/bin/env python3
"""
Main entry point for the Posture Detector application.
"""
# Fix protobuf compatibility issue between TensorFlow and MediaPipe
# Must be set before any imports that use protobuf
import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import argparse
import asyncio
import sys

import websockets
from dotenv import load_dotenv
from qasync import QApplication as QAsyncApplication
from qasync import QEventLoop

from config.settings import DEFAULT_CAMERA_HEIGHT, DEFAULT_CAMERA_WIDTH
from detector.pose_estimators import EstimatorType, create_estimator
from detector.posture_detector import PostureDetector
from utils.camera import CameraManager
from utils.visualization import MainAppController
from utils.websocket_client import WebSocketClient

# Load environment variables from .env file
load_dotenv()


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Posture Detection System")
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_CAMERA_WIDTH,
        help=f"Camera frame width (default: {DEFAULT_CAMERA_WIDTH})",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_CAMERA_HEIGHT,
        help=f"Camera frame height (default: {DEFAULT_CAMERA_HEIGHT})",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--no-guidance", action="store_true", help="Disable posture correction guidance")
    parser.add_argument(
        "--rotate",
        type=int,
        default=0,
        choices=[0, 90, 180, 270],
        help="Rotate webcam image by specified degrees (default: 0)",
    )
    parser.add_argument(
        "--estimator",
        type=str,
        default="mediapipe",
        choices=["mediapipe", "movenet_lightning", "movenet_thunder", "posenet", "openpose"],
        help="Pose estimation model to use (default: mediapipe)",
    )
    parser.add_argument(
        "--model-complexity",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="MediaPipe model complexity, only used with --estimator=mediapipe (default: 2)",
    )

    return parser.parse_args()


async def main():
    """Main function to run the posture detector"""
    args = parse_arguments()

    try:
        # Initialize Qt application with qasync integration
        app = QAsyncApplication(sys.argv)
        # set background color to black
        app.setStyleSheet("QWidget { background-color: #000000; }")
        loop = QEventLoop(app)
        asyncio.set_event_loop(loop)

        # Initialize camera with specified dimensions
        camera_manager = CameraManager(
            camera_index=args.camera,
            frame_width=args.width,
            frame_height=args.height,
            rotation=args.rotate,
        )
        websocket_client = WebSocketClient(
            base_url=os.getenv("WEBSOCET_BASE_URL"),
            api_key=os.getenv("API_KEY"),
            device_id=os.getenv("DEVICE_ID"),
        )

        async with websockets.connect(websocket_client.uri) as websocket:
            websocket_client.websocket = websocket

            # Initialize app controller first
            app_controller = MainAppController()

            # Create pose estimator based on command line argument
            estimator_kwargs = {}
            if args.estimator == "mediapipe":
                estimator_kwargs["model_complexity"] = args.model_complexity
            
            pose_estimator = create_estimator(args.estimator, **estimator_kwargs)
            pose_estimator.initialize()
            print(f"Using pose estimator: {pose_estimator.name}")

            # Initialize posture detector with injected estimator
            detector = PostureDetector(
                camera_manager=camera_manager,
                show_guidance=not args.no_guidance,
                pose_estimator=pose_estimator,
                websocket_client=websocket_client,
                app_controller=app_controller,
            )

            # Start the app controller
            app_controller.start()

            # Run the detector task
            detector_task = asyncio.create_task(detector.run())

            # Wait for either the detector to finish or KeyboardInterrupt
            try:
                # This ensures the event loop keeps running
                await detector_task
            except KeyboardInterrupt:
                print("\nApplication terminated by user")
                detector.cleanup_and_exit()

    except websockets.exceptions.ConnectionClosed as e:
        print(f"Connection closed with code {e.code}: {e.reason}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {str(e)}")
        import traceback

        traceback.print_exc()

    return 0


if __name__ == "__main__":
    asyncio.run(main())
