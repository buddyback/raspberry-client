"""
MediaPipe Pose Estimator implementation.

This module wraps Google's MediaPipe Pose (Legacy Solutions API) to conform to the
PoseEstimator interface.

Note: Using the legacy Solutions API for better compatibility with TensorFlow/protobuf.
The newer Tasks API has compatibility issues with protobuf >= 5.x.
"""

from typing import List

import cv2
import numpy as np

from .base import Landmark, PoseEstimator, PoseResult


class MediaPipePoseEstimator(PoseEstimator):
    """
    Pose estimator using Google MediaPipe Pose (Legacy Solutions API).
    
    MediaPipe Pose is a ML pipeline for 33 full-body pose landmarks.
    It's optimized for real-time performance and works well on various devices.
    
    Args:
        model_complexity: Model complexity (0=lite, 1=full, 2=heavy)
        min_detection_confidence: Minimum confidence for detection [0.0, 1.0]
        min_tracking_confidence: Minimum confidence for tracking [0.0, 1.0]
    """
    
    MODEL_NAMES = {
        0: "lite",
        1: "full", 
        2: "heavy",
    }
    
    # Mapping from landmark indices to our standard names
    # MediaPipe Pose uses these indices (same as PoseLandmark enum)
    # MediaPipe uses anatomical left/right (person's perspective)
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
        self._pose = None
        self._mp_pose = None
    
    @property
    def name(self) -> str:
        model_name = self.MODEL_NAMES.get(self._model_complexity, "unknown")
        return f"MediaPipe ({model_name})"
    
    @property
    def supported_landmarks(self) -> List[str]:
        return list(self.LANDMARK_MAPPING.values())
    
    @property
    def uses_reliable_visibility(self) -> bool:
        """
        MediaPipe reports ~1.0 visibility for all landmarks even when occluded.
        Visibility cannot be used to determine which side faces the camera.
        """
        return False
    
    @property
    def uses_smoothing(self) -> bool:
        """
        MediaPipe already applies internal smoothing (smooth_landmarks=True).
        No additional smoothing needed.
        """
        return False
    
    def initialize(self) -> None:
        """Initialize the MediaPipe Pose (Legacy Solutions API)."""
        if self._initialized:
            return
        
        try:
            import mediapipe as mp
            
            self._mp_pose = mp.solutions.pose
            
            # Create the pose estimator using Legacy Solutions API
            self._pose = self._mp_pose.Pose(
                static_image_mode=False,  # For video/webcam processing
                model_complexity=self._model_complexity,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=self._min_detection_confidence,
                min_tracking_confidence=self._min_tracking_confidence,
            )
            
            self._initialized = True
            model_name = self.MODEL_NAMES.get(self._model_complexity, "unknown")
            print(f"[MediaPipe] Initialized with model={model_name} (Solutions API)")
            
        except ImportError as e:
            raise RuntimeError(
                "MediaPipe is required. Install with: pip install mediapipe"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to initialize MediaPipe Pose: {e}")
    
    def process(self, frame: np.ndarray) -> PoseResult:
        """
        Process a frame using MediaPipe Pose.
        
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
        
        # Process the frame
        result = self._pose.process(rgb_frame)
        
        # Check if pose was detected
        if result.pose_landmarks is None:
            return PoseResult(
                landmarks={},
                raw_output=result,
                success=False,
                error_message="No pose detected in frame"
            )
        
        # Extract landmarks
        landmarks = {}
        for idx, name in self.LANDMARK_MAPPING.items():
            if idx < len(result.pose_landmarks.landmark):
                lm = result.pose_landmarks.landmark[idx]
                landmarks[name] = Landmark(
                    x=int(lm.x * w),
                    y=int(lm.y * h),
                    visibility=lm.visibility,
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
        if self._pose is not None:
            self._pose.close()
            self._pose = None
        self._initialized = False
        print("[MediaPipe] Cleaned up resources")
