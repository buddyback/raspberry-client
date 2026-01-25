"""
OpenPose Pose Estimator implementation using ONNX Runtime.

This module wraps the Lightweight OpenPose model (ONNX format) for efficient
pose estimation on ARM devices like Raspberry Pi.

Lightweight OpenPose is based on:
- Paper: "Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose"
- GitHub: https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch

Requirements:
    - onnxruntime (pip install onnxruntime)
    
To create the ONNX model, run export_to_onnx.py on an x86 machine with PyTorch.
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

# Body part connections for pose grouping
BODY_PARTS_KPT_IDS = [
    [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
    [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 16], [5, 17]
]
BODY_PARTS_PAF_IDS = (
    [12, 13], [20, 21], [14, 15], [16, 17], [22, 23], [24, 25], [0, 1], [2, 3], [4, 5],
    [6, 7], [8, 9], [10, 11], [28, 29], [30, 31], [34, 35], [32, 33], [36, 37], [18, 19], [26, 27]
)


class OpenPosePoseEstimator(PoseEstimator):
    """
    Pose estimator using Lightweight OpenPose via ONNX Runtime.
    
    This implementation uses the Lightweight OpenPose model converted to ONNX
    format for efficient inference on ARM devices like Raspberry Pi.
    
    Args:
        onnx_path: Path to ONNX model file (optional, defaults to local model)
        height_size: Input height for the model (default: 256)
    """
    
    # Default ONNX model path relative to project root
    DEFAULT_ONNX_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "openpose", "checkpoint", "openpose_lightweight.onnx"
    )
    
    # Fallback to PyTorch checkpoint path for error messages
    DEFAULT_CHECKPOINT = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "openpose", "checkpoint", "checkpoint_iter_370000.pth"
    )
    
    def __init__(
        self,
        onnx_path: str = None,
        height_size: int = 256,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._onnx_path = onnx_path or self.DEFAULT_ONNX_PATH
        self._height_size = height_size
        self._session = None
        self._stride = 8
        self._upsample_ratio = 4
    
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
            "hip": 0.10,
            "shoulder": 0.20,
        }
    
    def initialize(self) -> None:
        """Initialize the OpenPose ONNX Runtime session."""
        if self._initialized:
            return
        
        try:
            import onnxruntime as ort
            
            # Check if ONNX model exists
            if not os.path.exists(self._onnx_path):
                raise FileNotFoundError(
                    f"OpenPose ONNX model not found at {self._onnx_path}. "
                    f"Please run 'python openpose/export_to_onnx.py' on an x86 machine "
                    f"to convert the PyTorch model to ONNX format, then copy the "
                    f"resulting .onnx file to the Raspberry Pi."
                )
            
            print(f"[OpenPose] Loading ONNX model from {self._onnx_path}...")
            
            # Create ONNX Runtime session with CPU provider
            self._session = ort.InferenceSession(
                self._onnx_path,
                providers=['CPUExecutionProvider']
            )
            
            # Get input/output names
            self._input_name = self._session.get_inputs()[0].name
            self._output_names = [o.name for o in self._session.get_outputs()]
            
            self._initialized = True
            print(f"[OpenPose] Initialized successfully (ONNX Runtime)")
            
        except ImportError as e:
            raise RuntimeError(
                "ONNX Runtime is required for OpenPose. "
                "Install with: pip install onnxruntime"
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
        Run fast inference on a single image using ONNX Runtime.
        """
        height, width, _ = img.shape
        scale = net_input_height_size / height
        
        scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        
        img_mean = np.array([128, 128, 128], np.float32)
        img_scale = np.float32(1/256)
        scaled_img = self._normalize(scaled_img, img_mean, img_scale)
        
        min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
        padded_img, pad = self._pad_width(scaled_img, self._stride, (0, 0, 0), min_dims)
        
        # Prepare input for ONNX Runtime: (batch, channels, height, width)
        tensor_img = padded_img.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        
        # Run inference
        outputs = self._session.run(self._output_names, {self._input_name: tensor_img})
        
        # Process outputs - the model outputs heatmaps and PAFs
        # Output order matches torch model: stage2_heatmaps, stage2_pafs (last two outputs)
        if len(outputs) == 2:
            stage2_heatmaps = outputs[0]
            stage2_pafs = outputs[1]
        else:
            # If model has multiple stages, take the last heatmaps and pafs
            stage2_heatmaps = outputs[-2]
            stage2_pafs = outputs[-1]
        
        heatmaps = np.transpose(np.squeeze(stage2_heatmaps), (1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=self._upsample_ratio, fy=self._upsample_ratio, 
                              interpolation=cv2.INTER_CUBIC)
        
        pafs = np.transpose(np.squeeze(stage2_pafs), (1, 2, 0))
        pafs = cv2.resize(pafs, (0, 0), fx=self._upsample_ratio, fy=self._upsample_ratio, 
                          interpolation=cv2.INTER_CUBIC)
        
        return heatmaps, pafs, scale, pad
    
    def _extract_keypoints(self, heatmap, all_keypoints, total_keypoint_num):
        """Extract keypoints from a heatmap."""
        from operator import itemgetter
        import math
        
        heatmap[heatmap < 0.1] = 0
        heatmap_with_borders = np.pad(heatmap, [(2, 2), (2, 2)], mode='constant')
        heatmap_center = heatmap_with_borders[1:heatmap_with_borders.shape[0]-1, 1:heatmap_with_borders.shape[1]-1]
        heatmap_left = heatmap_with_borders[1:heatmap_with_borders.shape[0]-1, 2:heatmap_with_borders.shape[1]]
        heatmap_right = heatmap_with_borders[1:heatmap_with_borders.shape[0]-1, 0:heatmap_with_borders.shape[1]-2]
        heatmap_up = heatmap_with_borders[2:heatmap_with_borders.shape[0], 1:heatmap_with_borders.shape[1]-1]
        heatmap_down = heatmap_with_borders[0:heatmap_with_borders.shape[0]-2, 1:heatmap_with_borders.shape[1]-1]

        heatmap_peaks = (heatmap_center > heatmap_left) &\
                        (heatmap_center > heatmap_right) &\
                        (heatmap_center > heatmap_up) &\
                        (heatmap_center > heatmap_down)
        heatmap_peaks = heatmap_peaks[1:heatmap_center.shape[0]-1, 1:heatmap_center.shape[1]-1]
        keypoints = list(zip(np.nonzero(heatmap_peaks)[1], np.nonzero(heatmap_peaks)[0]))
        keypoints = sorted(keypoints, key=itemgetter(0))

        suppressed = np.zeros(len(keypoints), np.uint8)
        keypoints_with_score_and_id = []
        keypoint_num = 0
        for i in range(len(keypoints)):
            if suppressed[i]:
                continue
            for j in range(i+1, len(keypoints)):
                if math.sqrt((keypoints[i][0] - keypoints[j][0]) ** 2 +
                             (keypoints[i][1] - keypoints[j][1]) ** 2) < 6:
                    suppressed[j] = 1
            keypoint_with_score_and_id = (keypoints[i][0], keypoints[i][1], heatmap[keypoints[i][1], keypoints[i][0]],
                                          total_keypoint_num + keypoint_num)
            keypoints_with_score_and_id.append(keypoint_with_score_and_id)
            keypoint_num += 1
        all_keypoints.append(keypoints_with_score_and_id)
        return keypoint_num
    
    def _connections_nms(self, a_idx, b_idx, affinity_scores):
        """Non-maximum suppression for connections."""
        order = affinity_scores.argsort()[::-1]
        affinity_scores = affinity_scores[order]
        a_idx = a_idx[order]
        b_idx = b_idx[order]
        idx = []
        has_kpt_a = set()
        has_kpt_b = set()
        for t, (i, j) in enumerate(zip(a_idx, b_idx)):
            if i not in has_kpt_a and j not in has_kpt_b:
                idx.append(t)
                has_kpt_a.add(i)
                has_kpt_b.add(j)
        idx = np.asarray(idx, dtype=np.int32)
        return a_idx[idx], b_idx[idx], affinity_scores[idx]
    
    def _group_keypoints(self, all_keypoints_by_type, pafs, pose_entry_size=20, min_paf_score=0.05):
        """Group keypoints into poses using PAFs."""
        pose_entries = []
        all_keypoints = np.array([item for sublist in all_keypoints_by_type for item in sublist])
        points_per_limb = 10
        grid = np.arange(points_per_limb, dtype=np.float32).reshape(1, -1, 1)
        all_keypoints_by_type = [np.array(keypoints, np.float32) for keypoints in all_keypoints_by_type]
        
        for part_id in range(len(BODY_PARTS_PAF_IDS)):
            part_pafs = pafs[:, :, BODY_PARTS_PAF_IDS[part_id]]
            kpts_a = all_keypoints_by_type[BODY_PARTS_KPT_IDS[part_id][0]]
            kpts_b = all_keypoints_by_type[BODY_PARTS_KPT_IDS[part_id][1]]
            n = len(kpts_a)
            m = len(kpts_b)
            if n == 0 or m == 0:
                continue

            a = kpts_a[:, :2]
            a = np.broadcast_to(a[None], (m, n, 2))
            b = kpts_b[:, :2]
            vec_raw = (b[:, None, :] - a).reshape(-1, 1, 2)

            steps = (1 / (points_per_limb - 1) * vec_raw)
            points = steps * grid + a.reshape(-1, 1, 2)
            points = points.round().astype(dtype=np.int32)
            x = points[..., 0].ravel()
            y = points[..., 1].ravel()

            # Clamp to valid range
            x = np.clip(x, 0, pafs.shape[1] - 1)
            y = np.clip(y, 0, pafs.shape[0] - 1)

            field = part_pafs[y, x].reshape(-1, points_per_limb, 2)
            vec_norm = np.linalg.norm(vec_raw, ord=2, axis=-1, keepdims=True)
            vec = vec_raw / (vec_norm + 1e-6)
            affinity_scores = (field * vec).sum(-1).reshape(-1, points_per_limb)
            valid_affinity_scores = affinity_scores > min_paf_score
            valid_num = valid_affinity_scores.sum(1)
            affinity_scores = (affinity_scores * valid_affinity_scores).sum(1) / (valid_num + 1e-6)
            success_ratio = valid_num / points_per_limb

            valid_limbs = np.where(np.logical_and(affinity_scores > 0, success_ratio > 0.8))[0]
            if len(valid_limbs) == 0:
                continue
            b_idx, a_idx = np.divmod(valid_limbs, n)
            affinity_scores = affinity_scores[valid_limbs]

            a_idx, b_idx, affinity_scores = self._connections_nms(a_idx, b_idx, affinity_scores)
            connections = list(zip(kpts_a[a_idx, 3].astype(np.int32),
                                   kpts_b[b_idx, 3].astype(np.int32),
                                   affinity_scores))
            if len(connections) == 0:
                continue

            if part_id == 0:
                pose_entries = [np.ones(pose_entry_size) * -1 for _ in range(len(connections))]
                for i in range(len(connections)):
                    pose_entries[i][BODY_PARTS_KPT_IDS[0][0]] = connections[i][0]
                    pose_entries[i][BODY_PARTS_KPT_IDS[0][1]] = connections[i][1]
                    pose_entries[i][-1] = 2
                    pose_entries[i][-2] = np.sum(all_keypoints[connections[i][0:2], 2]) + connections[i][2]
            elif part_id == 17 or part_id == 18:
                kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
                kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
                for i in range(len(connections)):
                    for j in range(len(pose_entries)):
                        if pose_entries[j][kpt_a_id] == connections[i][0] and pose_entries[j][kpt_b_id] == -1:
                            pose_entries[j][kpt_b_id] = connections[i][1]
                        elif pose_entries[j][kpt_b_id] == connections[i][1] and pose_entries[j][kpt_a_id] == -1:
                            pose_entries[j][kpt_a_id] = connections[i][0]
                continue
            else:
                kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
                kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
                for i in range(len(connections)):
                    num = 0
                    for j in range(len(pose_entries)):
                        if pose_entries[j][kpt_a_id] == connections[i][0]:
                            pose_entries[j][kpt_b_id] = connections[i][1]
                            num += 1
                            pose_entries[j][-1] += 1
                            pose_entries[j][-2] += all_keypoints[connections[i][1], 2] + connections[i][2]
                    if num == 0:
                        pose_entry = np.ones(pose_entry_size) * -1
                        pose_entry[kpt_a_id] = connections[i][0]
                        pose_entry[kpt_b_id] = connections[i][1]
                        pose_entry[-1] = 2
                        pose_entry[-2] = np.sum(all_keypoints[connections[i][0:2], 2]) + connections[i][2]
                        pose_entries.append(pose_entry)

        filtered_entries = []
        for i in range(len(pose_entries)):
            if pose_entries[i][-1] < 3 or (pose_entries[i][-2] / pose_entries[i][-1] < 0.2):
                continue
            filtered_entries.append(pose_entries[i])
        pose_entries = np.asarray(filtered_entries)
        return pose_entries, all_keypoints
    
    def _extract_keypoints_and_poses(self, heatmaps, pafs, scale, pad, frame_height, frame_width):
        """
        Extract keypoints and group them into poses.
        
        Returns landmarks for the primary (most confident) detected pose.
        """
        num_keypoints = 18  # OpenPose COCO keypoints
        
        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):
            total_keypoints_num += self._extract_keypoints(
                heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num
            )
        
        pose_entries, all_keypoints = self._group_keypoints(all_keypoints_by_type, pafs)
        
        if len(all_keypoints) == 0:
            return {}
        
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
        Process a frame using OpenPose ONNX.
        
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
        self._session = None
        self._initialized = False
        print("[OpenPose] Cleaned up resources")
