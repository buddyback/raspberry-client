"""
OpenPose Pose Estimator implementation using Lightweight OpenPose PyTorch.

This module wraps the Lightweight OpenPose model (PyTorch format) for efficient
pose estimation.

Lightweight OpenPose is based on:
- Paper: "Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose"
- GitHub: https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch

Requirements:
    - torch (pip install torch)
"""

import os
import sys
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
    Pose estimator using Lightweight OpenPose via PyTorch.
    
    This implementation uses the Lightweight OpenPose model which is optimized
    for real-time inference on CPU. It detects 18 body keypoints.
    
    Args:
        checkpoint_path: Path to custom checkpoint file (optional, defaults to local checkpoint)
        height_size: Input height for the model (default: 256)
        use_cpu: Force CPU usage even if CUDA is available (default: True)
    """
    
    # Default checkpoint path relative to project root
    DEFAULT_CHECKPOINT = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "openpose", "checkpoint", "checkpoint_iter_370000.pth"
    )
    
    def __init__(
        self,
        checkpoint_path: str = None,
        height_size: int = 256,
        use_cpu: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._checkpoint_path = checkpoint_path or self.DEFAULT_CHECKPOINT
        self._height_size = height_size
        self._use_cpu = use_cpu
        self._net = None
        self._stride = 8
        self._upsample_ratio = 4
    
    @property
    def name(self) -> str:
        return "OpenPose (Lightweight PyTorch)"
    
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
    
    def _add_openpose_to_path(self):
        """Add the openpose directory to Python path for imports."""
        openpose_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "openpose"
        )
        if openpose_dir not in sys.path:
            sys.path.insert(0, openpose_dir)
    
    def initialize(self) -> None:
        """Initialize the OpenPose PyTorch model."""
        if self._initialized:
            return
        
        try:
            import torch
            
            # Add openpose directory to path for imports
            self._add_openpose_to_path()
            
            from models.with_mobilenet import PoseEstimationWithMobileNet
            from modules.load_state import load_state
            
            # Check if checkpoint exists
            if not os.path.exists(self._checkpoint_path):
                raise FileNotFoundError(
                    f"OpenPose checkpoint not found at {self._checkpoint_path}. "
                    "Please ensure the checkpoint file exists."
                )
            
            print(f"[OpenPose] Loading model from {self._checkpoint_path}...")
            
            # Create and load model
            self._net = PoseEstimationWithMobileNet()
            checkpoint = torch.load(self._checkpoint_path, map_location='cpu')
            load_state(self._net, checkpoint)
            
            # Set to evaluation mode
            self._net = self._net.eval()
            
            # Move to appropriate device
            if not self._use_cpu and torch.cuda.is_available():
                self._net = self._net.cuda()
                self._device = 'cuda'
                print("[OpenPose] Using CUDA")
            else:
                self._device = 'cpu'
                print("[OpenPose] Using CPU")
            
            self._initialized = True
            print(f"[OpenPose] Initialized successfully")
            
        except ImportError as e:
            raise RuntimeError(
                "PyTorch is required for OpenPose. "
                "Install with: pip install torch"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenPose: {e}")
    
    def _normalize(self, img, img_mean, img_scale):
        """Normalize image for network input."""
        img = np.array(img, dtype=np.float32)
        img = (img - img_mean) * img_scale
        return img
    
    def _pad_width(self, img, stride, pad_value, min_dims):
        """Pad image to be divisible by stride."""
        import math
        h, w, _ = img.shape
        h = min(min_dims[0], h)
        min_dims[0] = math.ceil(min_dims[0] / float(stride)) * stride
        min_dims[1] = max(min_dims[1], w)
        min_dims[1] = math.ceil(min_dims[1] / float(stride)) * stride
        pad = []
        pad.append(int(math.floor((min_dims[0] - h) / 2.0)))
        pad.append(int(math.floor((min_dims[1] - w) / 2.0)))
        pad.append(int(min_dims[0] - h - pad[0]))
        pad.append(int(min_dims[1] - w - pad[1]))
        padded_img = cv2.copyMakeBorder(img, pad[0], pad[2], pad[1], pad[3],
                                        cv2.BORDER_CONSTANT, value=pad_value)
        return padded_img, pad
    
    def _infer_fast(self, img, net_input_height_size):
        """
        Run fast inference on a single image.
        
        Based on the demo.py implementation.
        """
        import torch
        
        height, width, _ = img.shape
        scale = net_input_height_size / height
        
        scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        
        img_mean = np.array([128, 128, 128], np.float32)
        img_scale = np.float32(1/256)
        scaled_img = self._normalize(scaled_img, img_mean, img_scale)
        
        min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
        padded_img, pad = self._pad_width(scaled_img, self._stride, (0, 0, 0), min_dims)
        
        tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
        if self._device == 'cuda':
            tensor_img = tensor_img.cuda()
        
        with torch.no_grad():
            stages_output = self._net(tensor_img)
        
        stage2_heatmaps = stages_output[-2]
        heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=self._upsample_ratio, fy=self._upsample_ratio, 
                              interpolation=cv2.INTER_CUBIC)
        
        stage2_pafs = stages_output[-1]
        pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
        pafs = cv2.resize(pafs, (0, 0), fx=self._upsample_ratio, fy=self._upsample_ratio, 
                          interpolation=cv2.INTER_CUBIC)
        
        return heatmaps, pafs, scale, pad
    
    def _extract_keypoints_and_poses(self, heatmaps, pafs, scale, pad, frame_height, frame_width):
        """
        Extract keypoints and group them into poses.
        
        Returns landmarks for the primary (most confident) detected pose.
        """
        # Import modules from openpose directory
        from modules.keypoints import extract_keypoints, group_keypoints
        from modules.pose import Pose
        
        num_keypoints = Pose.num_kpts  # 18
        
        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):
            total_keypoints_num += extract_keypoints(
                heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num
            )
        
        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        
        # Transform keypoints back to original frame coordinates
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * self._stride / self._upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * self._stride / self._upsample_ratio - pad[0]) / scale
        
        if len(pose_entries) == 0:
            return {}
        
        # Get the most confident pose
        best_pose_idx = 0
        best_confidence = 0
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            confidence = pose_entries[n][18]  # Score is at index 18
            if confidence > best_confidence:
                best_confidence = confidence
                best_pose_idx = n
        
        pose_entry = pose_entries[best_pose_idx]
        
        # Extract landmarks from the best pose
        landmarks = {}
        for kpt_idx in range(num_keypoints):
            kpt_id = pose_entry[kpt_idx]
            if kpt_id == -1.0:
                # Keypoint not found - add with 0 confidence
                name = OPENPOSE_KEYPOINTS[kpt_idx]
                landmarks[name] = Landmark(
                    x=0,
                    y=0,
                    visibility=0.0,
                    name=name
                )
            else:
                kpt_id = int(kpt_id)
                x = int(all_keypoints[kpt_id, 0])
                y = int(all_keypoints[kpt_id, 1])
                confidence = float(all_keypoints[kpt_id, 2])
                
                # Clamp coordinates to frame bounds
                x = max(0, min(x, frame_width - 1))
                y = max(0, min(y, frame_height - 1))
                
                name = OPENPOSE_KEYPOINTS[kpt_idx]
                landmarks[name] = Landmark(
                    x=x,
                    y=y,
                    visibility=confidence,
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
        
        # Run inference
        heatmaps, pafs, scale, pad = self._infer_fast(frame, self._height_size)
        
        # Extract keypoints
        landmarks = self._extract_keypoints_and_poses(heatmaps, pafs, scale, pad, h, w)
        
        # Check if any keypoints were detected with reasonable confidence
        if not landmarks:
            return PoseResult(
                landmarks={},
                raw_output=(heatmaps, pafs),
                success=False,
                error_message="No pose detected"
            )
        
        # Check if we have enough confident keypoints
        confident_keypoints = sum(1 for lm in landmarks.values() if lm.visibility > 0.3)
        if confident_keypoints < 3:
            return PoseResult(
                landmarks={},
                raw_output=(heatmaps, pafs),
                success=False,
                error_message="No pose detected with sufficient confidence"
            )
        
        return PoseResult(
            landmarks=landmarks,
            raw_output=(heatmaps, pafs),
            success=True,
            error_message=None
        )
    
    def cleanup(self) -> None:
        """Release OpenPose resources."""
        self._net = None
        self._initialized = False
        print("[OpenPose] Cleaned up resources")
