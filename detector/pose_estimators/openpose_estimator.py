"""
OpenPose Pose Estimator implementation using Lightweight OpenPose ONNX.

This module wraps the Lightweight OpenPose model (ONNX format) for efficient
pose estimation without heavy dependencies like Caffe.

Lightweight OpenPose is based on:
- Paper: "Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose"
- GitHub: https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch

Requirements:
    - onnxruntime (pip install onnxruntime)
"""

import os
import urllib.request
from typing import List

import cv2
import numpy as np

from .base import Landmark, PoseEstimator, PoseResult


# OpenPose COCO keypoints (18 keypoints)
OPENPOSE_KEYPOINTS = [
    "nose",          # 0
    "neck",          # 1
    "r_shoulder",    # 2
    "r_elbow",       # 3
    "r_wrist",       # 4
    "l_shoulder",    # 5
    "l_elbow",       # 6
    "l_wrist",       # 7
    "r_hip",         # 8
    "r_knee",        # 9
    "r_ankle",       # 10
    "l_hip",         # 11
    "l_knee",        # 12
    "l_ankle",       # 13
    "r_eye",         # 14
    "l_eye",         # 15
    "r_ear",         # 16
    "l_ear",         # 17
]

# Map OpenPose keypoints to standard names
OPENPOSE_TO_STANDARD = {
    "nose": "nose",
    "neck": "neck",
    "r_shoulder": "r_shoulder",
    "r_elbow": "r_elbow",
    "r_wrist": "r_wrist",
    "l_shoulder": "l_shoulder",
    "l_elbow": "l_elbow",
    "l_wrist": "l_wrist",
    "r_hip": "r_hip",
    "r_knee": "r_knee",
    "r_ankle": "r_ankle",
    "l_hip": "l_hip",
    "l_knee": "l_knee",
    "l_ankle": "l_ankle",
    "r_eye": "r_eye",
    "l_eye": "l_eye",
    "r_ear": "r_ear",
    "l_ear": "l_ear",
}


class OpenPosePoseEstimator(PoseEstimator):
    """
    Pose estimator using Lightweight OpenPose via ONNX Runtime.
    
    This implementation uses the Lightweight OpenPose model which is optimized
    for real-time inference on CPU. It detects 18 body keypoints.
    
    Args:
        model_path: Path to custom ONNX model (optional, will download if not provided)
        input_height: Input height for the model (default: 256)
        input_width: Input width for the model (default: 456)
    """
    
    # Lightweight OpenPose ONNX model URL from PINTO's model zoo
    # Original URL is no longer available, using alternative
    MODEL_URL = "https://github.com/PINTO0309/PINTO_model_zoo/raw/main/084_LightWeight_OpenPose/model_float32.onnx"
    
    def __init__(
        self,
        model_path: str = None,
        input_height: int = 256,
        input_width: int = 456,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._model_path = model_path
        self._input_height = input_height
        self._input_width = input_width
        self._session = None
        self._input_name = None
        self._output_names = None
    
    @property
    def name(self) -> str:
        return "OpenPose (Lightweight ONNX)"
    
    @property
    def supported_landmarks(self) -> List[str]:
        return OPENPOSE_KEYPOINTS.copy()
    
    @property
    def visibility_thresholds(self) -> dict:
        """
        OpenPose-specific visibility thresholds.
        
        OpenPose outputs confidence values in a similar range to MoveNet/PoseNet.
        These thresholds are calibrated for OpenPose's native output range.
        """
        return {
            "ear": 0.25,
            "hip": 0.15,
            "shoulder": 0.20,
        }
    

    def _download_model(self) -> str:
        """Download the Lightweight OpenPose ONNX model if not available."""
        cache_dir = os.path.expanduser("~/.cache/pose_estimators/openpose")
        os.makedirs(cache_dir, exist_ok=True)
        
        model_path = os.path.join(cache_dir, "lightweight_openpose.onnx")
        
        if not os.path.exists(model_path):
            print(f"[OpenPose] Downloading model to {model_path}...")
            urllib.request.urlretrieve(self.MODEL_URL, model_path)
            print("[OpenPose] Download complete")
        
        return model_path
    
    def initialize(self) -> None:
        """Initialize the OpenPose ONNX model."""
        if self._initialized:
            return
        
        try:
            import onnxruntime as ort
            
            # Get model path
            if self._model_path is None:
                self._model_path = self._download_model()
            
            print(f"[OpenPose] Loading model from {self._model_path}...")
            
            # Create ONNX Runtime session
            # Use CPU provider by default, GPU if available
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            available_providers = ort.get_available_providers()
            providers = [p for p in providers if p in available_providers]
            
            self._session = ort.InferenceSession(self._model_path, providers=providers)
            
            # Get input/output names
            self._input_name = self._session.get_inputs()[0].name
            self._output_names = [o.name for o in self._session.get_outputs()]
            
            self._initialized = True
            print(f"[OpenPose] Initialized with providers: {providers}")
            
        except ImportError as e:
            raise RuntimeError(
                "onnxruntime is required for OpenPose. "
                "Install with: pip install onnxruntime"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenPose: {e}")
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for OpenPose input.
        
        Args:
            frame: BGR image (OpenCV format)
        
        Returns:
            Preprocessed numpy array ready for model input
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        resized = cv2.resize(rgb_frame, (self._input_width, self._input_height))
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Normalize with ImageNet mean/std
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (normalized - mean) / std
        
        # Transpose to CHW format and add batch dimension
        input_tensor = normalized.transpose(2, 0, 1)
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor.astype(np.float32)
    
    def _extract_keypoints_from_heatmaps(
        self, 
        heatmaps: np.ndarray, 
        frame_height: int, 
        frame_width: int
    ) -> dict:
        """
        Extract keypoints from heatmaps.
        
        Args:
            heatmaps: Heatmap output from model [1, num_keypoints, h, w]
            frame_height: Original frame height
            frame_width: Original frame width
        
        Returns:
            Dictionary of landmarks
        """
        landmarks = {}
        
        # Remove batch dimension
        heatmaps = np.squeeze(heatmaps)
        
        num_keypoints = min(len(OPENPOSE_KEYPOINTS), heatmaps.shape[0])
        heatmap_height, heatmap_width = heatmaps.shape[1:3]
        
        for idx in range(num_keypoints):
            heatmap = heatmaps[idx]
            
            # Find the position of maximum confidence
            max_pos = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            confidence = float(heatmap[max_pos])
            
            # Scale coordinates to original frame size
            x = int(max_pos[1] * frame_width / heatmap_width)
            y = int(max_pos[0] * frame_height / heatmap_height)
            
            name = OPENPOSE_KEYPOINTS[idx]
            landmarks[name] = Landmark(
                x=x,
                y=y,
                visibility=confidence,  # Use raw confidence
                name=name
            )
        
        return landmarks
    
    def process(self, frame: np.ndarray) -> PoseResult:
        """
        Process a frame using OpenPose.
        
        Args:
            frame: BGR image (OpenCV format)
        
        Returns:
            PoseResult with detected landmarks
        """
        if not self._initialized:
            raise RuntimeError("OpenPose not initialized. Call initialize() first.")
        
        # Get frame dimensions
        h, w = frame.shape[:2]
        
        # Preprocess the frame
        input_tensor = self._preprocess_frame(frame)
        
        # Run inference
        outputs = self._session.run(self._output_names, {self._input_name: input_tensor})
        
        # First output is heatmaps
        heatmaps = outputs[0]
        
        # Extract keypoints
        landmarks = self._extract_keypoints_from_heatmaps(heatmaps, h, w)
        
        # Check if any keypoints were detected with reasonable confidence
        if not landmarks:
            return PoseResult(
                landmarks={},
                raw_output=outputs,
                success=False,
                error_message="No pose detected"
            )
        
        max_confidence = max(lm.visibility for lm in landmarks.values())
        if max_confidence < 0.3:
            return PoseResult(
                landmarks={},
                raw_output=outputs,
                success=False,
                error_message="No pose detected with sufficient confidence"
            )
        
        return PoseResult(
            landmarks=landmarks,
            raw_output=outputs,
            success=True,
            error_message=None
        )
    
    def cleanup(self) -> None:
        """Release OpenPose resources."""
        self._session = None
        self._initialized = False
        print("[OpenPose] Cleaned up resources")
