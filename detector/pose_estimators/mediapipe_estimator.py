"""
MediaPipe Pose Estimator implementation.

This module wraps Google's MediaPipe PoseLandmarker (Tasks API) to conform to the
PoseEstimator interface.

Note: MediaPipe 0.10+ uses the new Tasks API instead of the legacy Solutions API.
"""

import os
import urllib.request
from typing import List

import cv2
import numpy as np

from .base import Landmark, PoseEstimator, PoseResult


class MediaPipePoseEstimator(PoseEstimator):
    """
    Pose estimator using Google MediaPipe PoseLandmarker (Tasks API).
    
    MediaPipe Pose is a ML pipeline for 33 full-body pose landmarks.
    It's optimized for real-time performance and works well on various devices.
    
    Args:
        model_complexity: Which model to use:
            0 = pose_landmarker_lite.task (fastest)
            1 = pose_landmarker_full.task (balanced)
            2 = pose_landmarker_heavy.task (most accurate)
        min_detection_confidence: Minimum confidence for detection [0.0, 1.0]
        min_tracking_confidence: Minimum confidence for tracking [0.0, 1.0]
    """
    
    # Model URLs from MediaPipe
    MODEL_URLS = {
        0: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        1: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
        2: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    }
    
    MODEL_NAMES = {
        0: "lite",
        1: "full", 
        2: "heavy",
    }
    
    # Mapping from landmark indices to our standard names
    # MediaPipe PoseLandmarker uses these indices
    LANDMARK_MAPPING = {
        0: "nose",
        1: "l_eye_inner",
        2: "l_eye",
        3: "l_eye_outer",
        4: "r_eye_inner",
        5: "r_eye",
        6: "r_eye_outer",
        7: "l_ear",
        8: "r_ear",
        9: "mouth_left",
        10: "mouth_right",
        11: "l_shoulder",
        12: "r_shoulder",
        13: "l_elbow",
        14: "r_elbow",
        15: "l_wrist",
        16: "r_wrist",
        17: "l_pinky",
        18: "r_pinky",
        19: "l_index",
        20: "r_index",
        21: "l_thumb",
        22: "r_thumb",
        23: "l_hip",
        24: "r_hip",
        25: "l_knee",
        26: "r_knee",
        27: "l_ankle",
        28: "r_ankle",
        29: "l_heel",
        30: "r_heel",
        31: "l_foot_index",
        32: "r_foot_index",
    }
    
    def __init__(
        self,
        model_complexity: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._model_complexity = model_complexity
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._landmarker = None
        self._mp = None
        self._vision = None
    
    @property
    def name(self) -> str:
        model_name = self.MODEL_NAMES.get(self._model_complexity, "unknown")
        return f"MediaPipe ({model_name})"
    
    @property
    def supported_landmarks(self) -> List[str]:
        return list(self.LANDMARK_MAPPING.values())
    
    def _download_model(self) -> str:
        """Download the model file if not available locally."""
        cache_dir = os.path.expanduser("~/.cache/pose_estimators/mediapipe")
        os.makedirs(cache_dir, exist_ok=True)
        
        model_name = self.MODEL_NAMES.get(self._model_complexity, "full")
        model_filename = f"pose_landmarker_{model_name}.task"
        model_path = os.path.join(cache_dir, model_filename)
        
        if not os.path.exists(model_path):
            url = self.MODEL_URLS.get(self._model_complexity, self.MODEL_URLS[1])
            print(f"[MediaPipe] Downloading model to {model_path}...")
            urllib.request.urlretrieve(url, model_path)
            print("[MediaPipe] Download complete")
        
        return model_path
    
    def initialize(self) -> None:
        """Initialize the MediaPipe PoseLandmarker."""
        if self._initialized:
            return
        
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            self._mp = mp
            self._vision = vision
            
            # Download model if needed
            model_path = self._download_model()
            
            # Create options
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,  # For single frame processing
                min_pose_detection_confidence=self._min_detection_confidence,
                min_tracking_confidence=self._min_tracking_confidence,
            )
            
            # Create the landmarker
            self._landmarker = vision.PoseLandmarker.create_from_options(options)
            
            self._initialized = True
            model_name = self.MODEL_NAMES.get(self._model_complexity, "unknown")
            print(f"[MediaPipe] Initialized with model={model_name}")
            
        except ImportError as e:
            raise RuntimeError(
                "MediaPipe is required. Install with: pip install mediapipe"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to initialize MediaPipe PoseLandmarker: {e}")
    
    def process(self, frame: np.ndarray) -> PoseResult:
        """
        Process a frame using MediaPipe PoseLandmarker.
        
        Args:
            frame: BGR image (OpenCV format)
        
        Returns:
            PoseResult with detected landmarks
        """
        if not self._initialized:
            raise RuntimeError("MediaPipe not initialized. Call initialize() first.")
        
        # Get frame dimensions
        h, w = frame.shape[:2]
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect pose landmarks
        result = self._landmarker.detect(mp_image)
        
        # Check if pose was detected
        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return PoseResult(
                landmarks={},
                raw_output=result,
                success=False,
                error_message="No pose detected in frame"
            )
        
        # Use the first detected pose
        pose_landmarks = result.pose_landmarks[0]
        
        # Extract landmarks
        landmarks = {}
        for idx, name in self.LANDMARK_MAPPING.items():
            if idx < len(pose_landmarks):
                lm = pose_landmarks[idx]
                landmarks[name] = Landmark(
                    x=int(lm.x * w),
                    y=int(lm.y * h),
                    visibility=lm.visibility if hasattr(lm, 'visibility') else 1.0,
                    name=name
                )
        
        return PoseResult(
            landmarks=landmarks,
            raw_output=result,
            success=True,
            error_message=None
        )
    
    def cleanup(self) -> None:
        """Release MediaPipe resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
        self._initialized = False
        print("[MediaPipe] Cleaned up resources")
