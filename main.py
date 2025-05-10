#!/usr/bin/env python3
"""
Main entry point for the Posture Detector application.
"""
import argparse
import asyncio
import os

import websockets
from dotenv import load_dotenv

from config.settings import DEFAULT_CAMERA_HEIGHT, DEFAULT_CAMERA_WIDTH
from detector.posture_detector import PostureDetector
from utils.camera import CameraManager
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
        "--model",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="MediaPipe model complexity  (default: 2)",
    )

    return parser.parse_args()


async def main():
    """Main function to run the posture detector"""
    args = parse_arguments()

    try:
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

        # Initialize HTTP client for sending data
            # http_client = HttpClient(
            #     api_key=os.getenv("API_KEY"), base_url=os.getenv("API_BASE_URL"), device_id=os.getenv("DEVICE_ID")
            # )
            # http_client.start_polling()

            # Initialize posture detector
            detector = PostureDetector(
                camera_manager=camera_manager,
                show_guidance=not args.no_guidance,
                model_complexity=args.model,
                # http_client=http_client,
                websocket_client=websocket_client
            )

            # Run the detector as a task
            detector_task = asyncio.create_task(detector.run())

            # Wait for either the detector to finish or KeyboardInterrupt
            try:
                await detector_task
            except KeyboardInterrupt:
                print("\nApplication terminated by user")
                detector.cleanup_and_exit()

    except websockets.exceptions.ConnectionClosed as e:
        print(f"Connection closed with code {e.code}: {e.reason}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {str(e)}")

    return 0


if __name__ == "__main__":
    asyncio.run(main())
