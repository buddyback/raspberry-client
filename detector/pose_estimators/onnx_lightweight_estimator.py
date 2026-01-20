"""
ONNX Lightweight Pose Estimator implementation.

This module provides an ultra-lightweight pose estimator using MobileNetV2-based
ONNX models optimized for edge devices like Raspberry Pi.

Uses the MoveNet SinglePose Lightning TFLite model converted to ONNX,
or similar lightweight pose estimation models.

Requirements:
    - onnxruntime (pip install onnxruntime)
"""

import os
import urllib.request
from typing import List

import cv2
import numpy as np

from .base import Landmark, PoseEstimator, PoseResult


# Standard COCO keypoints (17 keypoints) - same as MoveNet
LIGHTWEIGHT_KEYPOINTS = [
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


class ONNXLightweightPoseEstimator(PoseEstimator):
    """
    Ultra-lightweight pose estimator using ONNX Runtime.
    
    This implementation uses MobileNetV2-based pose estimation models
    optimized for real-time inference on CPU and edge devices.
    
    Suitable for Raspberry Pi and other resource-constrained environments.
    
    Args:
        model_path: Path to custom ONNX model (optional)
        input_size: Input size for the model (default: 192 for speed)
    """
    
    # Ultra-lightweight pose model URL (MoveNet Lightning ONNX conversion)
    MODEL_URL = "https://github.com/PINTO0309/PINTO_model_zoo/raw/main/115_MoveNet/resources/saved_model_movenet_singlepose_lightning_192x192_integer_quant/model_float32.onnx"
    
    def __init__(
        self,
        model_path: str = None,
        input_size: int = 192,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._model_path = model_path
        self._input_size = input_size
        self._session = None
        self._input_name = None
        self._output_name = None
    
    @property
    def name(self) -> str:
        return "ONNX Lightweight"
    
    @property
    def supported_landmarks(self) -> List[str]:
        return LIGHTWEIGHT_KEYPOINTS.copy()
    
    @property
    def visibility_thresholds(self) -> dict:
        """
        ONNX Lightweight-specific visibility thresholds.
        
        This model outputs confidence values similar to MoveNet since it's
        based on MoveNet architecture. Thresholds are calibrated accordingly.
        """
        return {
            "ear": 0.25,
            "hip": 0.15,
            "shoulder": 0.20,
        }
    

    def _download_model(self) -> str:
        """Download the lightweight ONNX model if not available."""
        cache_dir = os.path.expanduser("~/.cache/pose_estimators/onnx_lightweight")
        os.makedirs(cache_dir, exist_ok=True)
        
        model_path = os.path.join(cache_dir, "movenet_lightning.onnx")
        
        if not os.path.exists(model_path):
            print(f"[ONNX Lightweight] Downloading model to {model_path}...")
            try:
                urllib.request.urlretrieve(self.MODEL_URL, model_path)
                print("[ONNX Lightweight] Download complete")
            except Exception as e:
                print(f"[ONNX Lightweight] Failed to download from primary URL: {e}")
                # Fallback: create a simple message
                raise RuntimeError(
                    "Failed to download ONNX model. Please download manually from:\n"
                    f"{self.MODEL_URL}\n"
                    f"And place it at: {model_path}"
                )
        
        return model_path
    
    def initialize(self) -> None:
        """Initialize the ONNX Lightweight model."""
        if self._initialized:
            return
        
        try:
            import onnxruntime as ort
            
            # Get model path
            if self._model_path is None:
                self._model_path = self._download_model()
            
            print(f"[ONNX Lightweight] Loading model from {self._model_path}...")
            
            # Create ONNX Runtime session with optimizations
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Use CPU provider (optimized for edge devices)
            providers = ['CPUExecutionProvider']
            
            self._session = ort.InferenceSession(
                self._model_path, 
                sess_options=sess_options,
                providers=providers
            )
            
            # Get input/output names
            inputs = self._session.get_inputs()
            outputs = self._session.get_outputs()
            
            self._input_name = inputs[0].name
            self._output_name = outputs[0].name
            
            # Update input size based on model
            input_shape = inputs[0].shape
            if len(input_shape) >= 3:
                self._input_size = input_shape[1] if isinstance(input_shape[1], int) else self._input_size
            
            self._initialized = True
            print(f"[ONNX Lightweight] Initialized (input size: {self._input_size}x{self._input_size})")
            
        except ImportError as e:
            raise RuntimeError(
                "onnxruntime is required for ONNX Lightweight. "
                "Install with: pip install onnxruntime"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ONNX Lightweight: {e}")
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for model input.
        
        Args:
            frame: BGR image (OpenCV format)
        
        Returns:
            Preprocessed numpy array ready for model input
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        resized = cv2.resize(rgb_frame, (self._input_size, self._input_size))
        
        # Normalize to int32 (MoveNet expects integer input)
        input_tensor = resized.astype(np.int32)
        
        # Add batch dimension
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor
    
    def process(self, frame: np.ndarray) -> PoseResult:
        """
        Process a frame using ONNX Lightweight model.
        
        Args:
            frame: BGR image (OpenCV format)
        
        Returns:
            PoseResult with detected landmarks
        """
        if not self._initialized:
            raise RuntimeError("ONNX Lightweight not initialized. Call initialize() first.")
        
        # Get frame dimensions
        h, w = frame.shape[:2]
        
        # Preprocess the frame
        input_tensor = self._preprocess_frame(frame)
        
        # Run inference
        outputs = self._session.run([self._output_name], {self._input_name: input_tensor})
        
        # Parse output - shape is typically [1, 1, 17, 3] (batch, person, keypoints, [y, x, confidence])
        keypoints = outputs[0]
        
        # Handle different output shapes
        if keypoints.ndim == 4:
            keypoints = keypoints[0, 0]  # [17, 3]
        elif keypoints.ndim == 3:
            keypoints = keypoints[0]  # [17, 3]
        
        # Check if any keypoints were detected
        max_confidence = keypoints[:, 2].max() if keypoints.shape[1] >= 3 else 0
        if max_confidence < 0.1:
            return PoseResult(
                landmarks={},
                raw_output=outputs,
                success=False,
                error_message="No pose detected with sufficient confidence"
            )
        
        # Convert to landmarks
        landmarks = {}
        for idx, name in enumerate(LIGHTWEIGHT_KEYPOINTS):
            if idx >= keypoints.shape[0]:
                break
                
            y_norm, x_norm = keypoints[idx, 0], keypoints[idx, 1]
            confidence = keypoints[idx, 2] if keypoints.shape[1] >= 3 else 0.5
            
            
            landmarks[name] = Landmark(
                x=int(x_norm * w),
                y=int(y_norm * h),
                visibility=float(confidence),  # Use raw confidence
                name=name
            )
        
        return PoseResult(
            landmarks=landmarks,
            raw_output=outputs,
            success=True,
            error_message=None
        )
    
    def cleanup(self) -> None:
        """Release ONNX Lightweight resources."""
        self._session = None
        self._initialized = False
        print("[ONNX Lightweight] Cleaned up resources")
