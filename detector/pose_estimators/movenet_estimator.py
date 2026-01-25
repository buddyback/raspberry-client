"""
MoveNet Pose Estimator implementation using TensorFlow Lite.

This module wraps Google's MoveNet model using TensorFlow Lite for efficient
inference on edge devices like Raspberry Pi.

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
    Pose estimator using Google MoveNet via TensorFlow Lite.
    
    MoveNet is an ultra-fast and accurate pose detection model that detects
    17 keypoints of a body. This implementation uses TensorFlow Lite for 
    efficient inference on edge devices like Raspberry Pi.
    
    Args:
        variant: Model variant - 'lightning' (faster) or 'thunder' (more accurate)
        model_path: Optional path to a custom TFLite model file
    """
    
    # TFLite model URLs (official Google models)
    MODEL_URLS = {
        "lightning": "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4?lite-format=tflite",
        "thunder": "https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/int8/4?lite-format=tflite",
    }
    
    INPUT_SIZES = {
        "lightning": 192,
        "thunder": 256,
    }
    
    def __init__(
        self,
        variant: Literal["lightning", "thunder"] = "lightning",
        model_path: str = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._variant = variant.lower()
        if self._variant not in self.MODEL_URLS:
            raise ValueError(f"Invalid variant '{variant}'. Must be 'lightning' or 'thunder'")
        
        self._model_path = model_path
        self._input_size = self.INPUT_SIZES[self._variant]
        self._interpreter = None
        self._input_details = None
        self._output_details = None
    
    @property
    def name(self) -> str:
        return f"MoveNet ({self._variant})"

    @property
    def uses_reliable_visibility(self) -> bool:
        return True

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
            "hip": 0.07,
            "shoulder": 0.15,
        }
    
    def _download_model(self) -> str:
        """Download the MoveNet TFLite model if not available locally."""
        import os
        import urllib.request
        
        cache_dir = os.path.expanduser("~/.cache/pose_estimators")
        os.makedirs(cache_dir, exist_ok=True)
        
        model_filename = f"movenet_{self._variant}.tflite"
        model_path = os.path.join(cache_dir, model_filename)
        
        if not os.path.exists(model_path):
            print(f"[MoveNet] Downloading {self._variant} model to {model_path}...")
            try:
                urllib.request.urlretrieve(self.MODEL_URLS[self._variant], model_path)
                print("[MoveNet] Download complete")
            except Exception as e:
                # Fallback: try alternative URL format
                alt_urls = {
                    "lightning": "https://storage.googleapis.com/movenet/models/movenet_singlepose_lightning_int8_4.tflite",
                    "thunder": "https://storage.googleapis.com/movenet/models/movenet_singlepose_thunder_int8_4.tflite",
                }
                print(f"[MoveNet] Primary download failed, trying alternative URL...")
                urllib.request.urlretrieve(alt_urls[self._variant], model_path)
                print("[MoveNet] Download complete (alternative URL)")
        
        return model_path
    
    def initialize(self) -> None:
        """Initialize the MoveNet TFLite interpreter."""
        if self._initialized:
            return
        
        try:
            # Force CPU-only mode
            import os
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            
            # Use TensorFlow Lite for inference
            import tensorflow as tf
            
            # Get model path
            if self._model_path is None:
                self._model_path = self._download_model()
            
            print(f"[MoveNet] Loading model from {self._model_path}... (CPU mode, TFLite)")
            
            # Create interpreter
            self._interpreter = tf.lite.Interpreter(model_path=self._model_path)
            self._interpreter.allocate_tensors()
            
            # Get input and output details
            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()
            
            # Update input size based on model
            input_shape = self._input_details[0]['shape']
            self._input_size = input_shape[1]  # Assuming square input
            
            self._initialized = True
            print(f"[MoveNet] Initialized {self._variant} (input size: {self._input_size}x{self._input_size}, TFLite)")
            
        except ImportError as e:
            raise RuntimeError(
                "TensorFlow is required for MoveNet. "
                "Install with: pip install tensorflow"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to initialize MoveNet: {e}")
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for MoveNet input.
        
        Args:
            frame: BGR image (OpenCV format)
        
        Returns:
            Preprocessed numpy array ready for model input
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        resized = cv2.resize(rgb_frame, (self._input_size, self._input_size))
        
        # Add batch dimension and convert to appropriate dtype
        input_dtype = self._input_details[0]['dtype']
        if input_dtype == np.uint8:
            input_data = np.expand_dims(resized.astype(np.uint8), axis=0)
        else:
            # For float models, normalize to [0, 1]
            input_data = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)
        
        return input_data
    
    def process(self, frame: np.ndarray) -> PoseResult:
        """
        Process a frame using MoveNet TFLite.
        
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
        input_data = self._preprocess_frame(frame)
        
        # Set input tensor
        self._interpreter.set_tensor(self._input_details[0]['index'], input_data)
        
        # Run inference
        self._interpreter.invoke()
        
        # Get output tensor
        # Output shape: [1, 1, 17, 3] - (batch, person, keypoints, [y, x, confidence])
        keypoints = self._interpreter.get_tensor(self._output_details[0]['index'])
        keypoints = np.squeeze(keypoints)  # Remove batch dimension
        
        # Handle different output shapes
        if keypoints.ndim == 2:
            # Shape is [17, 3]
            pass
        elif keypoints.ndim == 3:
            # Shape is [1, 17, 3]
            keypoints = keypoints[0]
        
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
                visibility=float(confidence),
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
        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self._initialized = False
        print(f"[MoveNet] Cleaned up resources")
