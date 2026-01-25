"""
PoseNet Pose Estimator implementation.

This module wraps PoseNet using TensorFlow Lite for efficient inference.
PoseNet is a vision model that can detect human figures in images and videos.
"""

from typing import List

import cv2
import numpy as np

from .base import Landmark, PoseEstimator, PoseResult


# PoseNet keypoint definitions (same as MoveNet - 17 keypoints)
POSENET_KEYPOINTS = [
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


class PoseNetPoseEstimator(PoseEstimator):
    """
    Pose estimator using PoseNet via TensorFlow Lite.
    
    PoseNet is a vision model that can detect human figures in images and video.
    It estimates where key body joints are. This implementation uses TensorFlow Lite
    for efficient inference, which is well-suited for edge devices.
    
    Args:
        model_path: Path to a custom TFLite model file (optional)
        input_size: Input size for the model (default: 257)
    """
    
    # Default model URL for download
    MODEL_URL = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite"
    
    def __init__(
        self,
        model_path: str = None,
        input_size: int = 257,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._model_path = model_path
        self._input_size = input_size
        self._interpreter = None
        self._input_details = None
        self._output_details = None
    
    @property
    def name(self) -> str:
        return "PoseNet (TFLite)"
    
    @property
    def supported_landmarks(self) -> List[str]:
        return POSENET_KEYPOINTS.copy()
    
    @property
    def visibility_thresholds(self) -> dict:
        """
        PoseNet-specific visibility thresholds.
        
        PoseNet outputs lower confidence values than MediaPipe. These thresholds
        are calibrated for PoseNet's native output range.
        """
        return {
            "ear": 0.30,
            "hip": 0.10,
            "shoulder": 0.25,
        }
    

    def _download_model(self) -> str:
        """Download the default PoseNet model if not available locally."""
        import os
        import urllib.request
        
        cache_dir = os.path.expanduser("~/.cache/pose_estimators")
        os.makedirs(cache_dir, exist_ok=True)
        
        model_filename = "posenet_mobilenet_v1.tflite"
        model_path = os.path.join(cache_dir, model_filename)
        
        if not os.path.exists(model_path):
            print(f"[PoseNet] Downloading model to {model_path}...")
            urllib.request.urlretrieve(self.MODEL_URL, model_path)
            print("[PoseNet] Download complete")
        
        return model_path
    
    def initialize(self) -> None:
        """Initialize the PoseNet TFLite interpreter."""
        if self._initialized:
            return
        
        try:
            # Force CPU-only mode to avoid GPU compatibility issues
            import os
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            
            # Use TensorFlow Lite for inference
            import tensorflow as tf
            
            # Get model path
            if self._model_path is None:
                self._model_path = self._download_model()
            
            print(f"[PoseNet] Loading model from {self._model_path}... (CPU mode)")
            
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
            print(f"[PoseNet] Initialized (input size: {self._input_size}x{self._input_size})")
            
        except ImportError as e:
            raise RuntimeError(
                "TensorFlow is required for PoseNet. "
                "Install with: pip install tensorflow"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to initialize PoseNet: {e}")
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for PoseNet input.
        
        Args:
            frame: BGR image (OpenCV format)
        
        Returns:
            Preprocessed numpy array ready for model input
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        resized = cv2.resize(rgb_frame, (self._input_size, self._input_size))
        
        # Normalize to float32 in range [0, 1] or [-1, 1] depending on model
        input_data = np.expand_dims(resized.astype(np.float32), axis=0)
        
        # Check if model expects normalized input
        input_dtype = self._input_details[0]['dtype']
        if input_dtype == np.float32:
            input_data = (input_data - 127.5) / 127.5  # Normalize to [-1, 1]
        
        return input_data
    
    def process(self, frame: np.ndarray) -> PoseResult:
        """
        Process a frame using PoseNet.
        
        Args:
            frame: BGR image (OpenCV format)
        
        Returns:
            PoseResult with detected landmarks
        """
        if not self._initialized:
            raise RuntimeError("PoseNet not initialized. Call initialize() first.")
        
        # Get frame dimensions for coordinate scaling
        h, w = frame.shape[:2]
        
        # Preprocess the frame
        input_data = self._preprocess_frame(frame)
        
        # Set input tensor
        self._interpreter.set_tensor(self._input_details[0]['index'], input_data)
        
        # Run inference
        self._interpreter.invoke()
        
        # Get output tensors
        # PoseNet outputs: heatmaps and offsets
        heatmaps = self._interpreter.get_tensor(self._output_details[0]['index'])
        offsets = self._interpreter.get_tensor(self._output_details[1]['index'])
        
        # Parse keypoints from heatmaps and offsets
        landmarks = self._parse_output(heatmaps, offsets, h, w)
        
        # Check if any keypoints were detected with reasonable confidence
        if not landmarks:
            return PoseResult(
                landmarks={},
                raw_output={"heatmaps": heatmaps, "offsets": offsets},
                success=False,
                error_message="No pose detected"
            )
        
        max_confidence = max(lm.visibility for lm in landmarks.values())
        if max_confidence < 0.1:
            return PoseResult(
                landmarks={},
                raw_output={"heatmaps": heatmaps, "offsets": offsets},
                success=False,
                error_message="No pose detected with sufficient confidence"
            )
        
        return PoseResult(
            landmarks=landmarks,
            raw_output={"heatmaps": heatmaps, "offsets": offsets},
            success=True,
            error_message=None
        )
    
    def _parse_output(
        self, 
        heatmaps: np.ndarray, 
        offsets: np.ndarray,
        frame_height: int,
        frame_width: int
    ) -> dict:
        """
        Parse PoseNet output into landmarks.
        
        Args:
            heatmaps: Heatmap output from model
            offsets: Offset output from model
            frame_height: Original frame height
            frame_width: Original frame width
        
        Returns:
            Dictionary of landmarks
        """
        landmarks = {}
        
        # Squeeze batch dimension
        heatmaps = np.squeeze(heatmaps)
        offsets = np.squeeze(offsets)
        
        num_keypoints = len(POSENET_KEYPOINTS)
        heatmap_height, heatmap_width = heatmaps.shape[:2]
        
        for idx, name in enumerate(POSENET_KEYPOINTS):
            # Get heatmap for this keypoint
            heatmap = heatmaps[:, :, idx]
            
            # Find the position of maximum confidence
            max_pos = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            confidence = float(heatmap[max_pos])
            
            # Apply sigmoid to get probability
            confidence = 1 / (1 + np.exp(-confidence))
            
            # Get the offset for refinement
            y_offset = offsets[max_pos[0], max_pos[1], idx]
            x_offset = offsets[max_pos[0], max_pos[1], idx + num_keypoints]
            
            # Calculate final position
            y = (max_pos[0] / heatmap_height + y_offset / self._input_size) * frame_height
            x = (max_pos[1] / heatmap_width + x_offset / self._input_size) * frame_width
            
            landmarks[name] = Landmark(
                x=int(x),
                y=int(y),
                visibility=confidence,  # Use raw confidence
                name=name
            )
        
        return landmarks
    
    def cleanup(self) -> None:
        """Release PoseNet resources."""
        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self._initialized = False
        print("[PoseNet] Cleaned up resources")
