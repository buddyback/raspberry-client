"""
MoveNet Pose Estimator implementation.

This module wraps Google's MoveNet model from TensorFlow Hub to conform to the
PoseEstimator interface. MoveNet is optimized for real-time pose estimation.

Two variants are available:
- Lightning: Faster, optimized for latency-critical applications
- Thunder: More accurate, slightly slower
"""

from typing import List, Literal

import cv2
import numpy as np

from .base import Landmark, PoseEstimator, PoseResult


# MoveNet keypoint indices
MOVENET_KEYPOINTS = [
    "nose",          # 0
    "l_eye",         # 1
    "r_eye",         # 2
    "l_ear",         # 3
    "r_ear",         # 4
    "l_shoulder",    # 5
    "r_shoulder",    # 6
    "l_elbow",       # 7
    "r_elbow",       # 8
    "l_wrist",       # 9
    "r_wrist",       # 10
    "l_hip",         # 11
    "r_hip",         # 12
    "l_knee",        # 13
    "r_knee",        # 14
    "l_ankle",       # 15
    "r_ankle",       # 16
]


class MoveNetPoseEstimator(PoseEstimator):
    """
    Pose estimator using Google MoveNet from TensorFlow Hub.
    
    MoveNet is an ultra-fast and accurate pose detection model that detects
    17 keypoints of a body. It's specifically designed for real-time applications.
    
    Args:
        variant: Model variant - 'lightning' (faster) or 'thunder' (more accurate)
    """
    
    MODEL_URLS = {
        "lightning": "https://tfhub.dev/google/movenet/singlepose/lightning/4",
        "thunder": "https://tfhub.dev/google/movenet/singlepose/thunder/4",
    }
    
    INPUT_SIZES = {
        "lightning": 192,
        "thunder": 256,
    }
    
    def __init__(
        self,
        variant: Literal["lightning", "thunder"] = "lightning",
        **kwargs
    ):
        super().__init__(**kwargs)
        self._variant = variant.lower()
        if self._variant not in self.MODEL_URLS:
            raise ValueError(f"Invalid variant '{variant}'. Must be 'lightning' or 'thunder'")
        
        self._model = None
        self._input_size = self.INPUT_SIZES[self._variant]
        self._tf = None  # TensorFlow module (lazy import)
        self._hub = None  # TensorFlow Hub module (lazy import)
    
    @property
    def name(self) -> str:
        return f"MoveNet ({self._variant})"
    
    @property
    def supported_landmarks(self) -> List[str]:
        return MOVENET_KEYPOINTS.copy()
    
    @property
    def visibility_thresholds(self) -> dict:
        """
        MoveNet-specific visibility thresholds.
        
        MoveNet outputs lower confidence values than MediaPipe (typically 0.1-0.5
        for visible keypoints vs 0.9-1.0 for MediaPipe). These thresholds are
        calibrated for MoveNet's native output range.
        """
        return {
            "ear": 0.10,
            "hip": 0.10,
            "shoulder": 0.15,
        }
    
    def initialize(self) -> None:
        """Initialize the MoveNet model from TensorFlow Hub."""
        if self._initialized:
            return
        
        try:
            # Force CPU-only mode to avoid GPU compatibility issues
            # This is especially important for Raspberry Pi (no GPU) and
            # newer GPUs that may not have compatible CUDA kernels
            import os
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            
            # Lazy import TensorFlow to avoid loading it if not needed
            import tensorflow as tf
            import tensorflow_hub as hub
            
            self._tf = tf
            self._hub = hub
            
            print(f"[MoveNet] Loading model: {self._variant}... (CPU mode)")
            model_url = self.MODEL_URLS[self._variant]
            
            # Load the model from TensorFlow Hub
            self._model = hub.load(model_url)
            self._movenet = self._model.signatures['serving_default']
            
            self._initialized = True
            print(f"[MoveNet] Initialized {self._variant} (input size: {self._input_size}x{self._input_size})")
            
        except ImportError as e:
            raise RuntimeError(
                "TensorFlow and tensorflow-hub are required for MoveNet. "
                "Install with: pip install tensorflow tensorflow-hub"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to initialize MoveNet: {e}")
    
    def _preprocess_frame(self, frame: np.ndarray) -> "tf.Tensor":
        """
        Preprocess frame for MoveNet input.
        
        Args:
            frame: BGR image (OpenCV format)
        
        Returns:
            Preprocessed tensor ready for model input
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        resized = cv2.resize(rgb_frame, (self._input_size, self._input_size))
        
        # Convert to tensor and add batch dimension
        input_tensor = self._tf.cast(resized, dtype=self._tf.int32)
        input_tensor = self._tf.expand_dims(input_tensor, axis=0)
        
        return input_tensor
    
    def process(self, frame: np.ndarray) -> PoseResult:
        """
        Process a frame using MoveNet.
        
        Args:
            frame: BGR image (OpenCV format)
        
        Returns:
            PoseResult with detected landmarks
        """
        if not self._initialized:
            raise RuntimeError("MoveNet not initialized. Call initialize() first.")
        
        # Get frame dimensions for coordinate scaling
        h, w = frame.shape[:2]
        
        # Preprocess the frame
        input_tensor = self._preprocess_frame(frame)
        
        # Run inference
        outputs = self._movenet(input_tensor)
        
        # Extract keypoints
        # Output shape: [1, 1, 17, 3] - (batch, person, keypoints, [y, x, confidence])
        keypoints = outputs['output_0'].numpy()[0, 0, :, :]
        
        # Check if any keypoints were detected with reasonable confidence
        max_confidence = keypoints[:, 2].max()
        if max_confidence < 0.1:
            return PoseResult(
                landmarks={},
                raw_output=keypoints,
                success=False,
                error_message="No pose detected with sufficient confidence"
            )
        
        # Convert to our landmark format
        landmarks = {}
        for idx, name in enumerate(MOVENET_KEYPOINTS):
            y_norm, x_norm, confidence = keypoints[idx]
            landmarks[name] = Landmark(
                x=int(x_norm * w),
                y=int(y_norm * h),
                visibility=float(confidence),  # Use raw confidence
                name=name
            )
        
        return PoseResult(
            landmarks=landmarks,
            raw_output=keypoints,
            success=True,
            error_message=None
        )
    
    def cleanup(self) -> None:
        """Release MoveNet resources."""
        self._model = None
        self._movenet = None
        self._initialized = False
        
        # Clear TensorFlow session if possible
        if self._tf is not None:
            try:
                self._tf.keras.backend.clear_session()
            except Exception:
                pass
        
        print(f"[MoveNet] Cleaned up resources")
