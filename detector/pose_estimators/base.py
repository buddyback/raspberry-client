"""
Base classes for pose estimation.

This module defines the abstract interface that all pose estimators must implement,
along with standardized data structures for landmarks and results.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Landmark:
    """
    Represents a single keypoint/landmark detected by a pose estimator.
    
    Attributes:
        x: X coordinate in pixels
        y: Y coordinate in pixels
        visibility: Confidence score [0.0, 1.0] for this landmark
        name: Human-readable name of the landmark (e.g., 'left_shoulder')
    """
    x: int
    y: int
    visibility: float
    name: str


@dataclass
class PoseResult:
    """
    Standardized result from pose estimation.
    
    Attributes:
        landmarks: Dictionary mapping landmark names to Landmark objects
        raw_output: Original output from the underlying model (for debugging/advanced use)
        success: Whether pose detection was successful
        error_message: Optional error message if success is False
    """
    landmarks: Dict[str, Landmark] = field(default_factory=dict)
    raw_output: Any = None
    success: bool = False
    error_message: Optional[str] = None


# Standard landmark names used across all estimators
# These map to the landmarks needed by PostureAnalyzer
STANDARD_LANDMARKS = [
    "l_shoulder",
    "r_shoulder",
    "l_ear",
    "r_ear",
    "l_hip",
    "r_hip",
    "nose",
    "l_eye",
    "r_eye",
    "l_elbow",
    "r_elbow",
    "l_wrist",
    "r_wrist",
    "l_knee",
    "r_knee",
    "l_ankle",
    "r_ankle",
]


class PoseEstimator(ABC):
    """
    Abstract base class for all pose estimators.
    
    Implementations must provide methods to:
    - Initialize the underlying model
    - Process frames and return standardized landmarks
    - Clean up resources when done
    
    All implementations should return landmarks in a standardized format
    that can be consumed by PostureAnalyzer.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the estimator with optional configuration.
        
        Args:
            **kwargs: Model-specific configuration options
        """
        self._initialized = False
        self._config = kwargs
    
    @property
    def is_initialized(self) -> bool:
        """Check if the estimator has been initialized."""
        return self._initialized
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this estimator."""
        pass
    
    @property
    @abstractmethod
    def supported_landmarks(self) -> List[str]:
        """Return list of landmark names this estimator can detect."""
        pass
    
    @property
    def visibility_thresholds(self) -> Dict[str, float]:
        """
        Return model-specific visibility thresholds for webcam placement detection.
        
        These thresholds are optimized for each model since different models
        output confidence values in different ranges. MediaPipe outputs 0.9-1.0
        for visible keypoints, while MoveNet/PoseNet typically output 0.3-0.6.
        
        Returns:
            Dictionary with thresholds for 'ear', 'hip', and 'shoulder' detection.
            Default values are calibrated for MediaPipe.
        """
        return {
            "ear": 0.90,
            "hip": 0.75,
            "shoulder": 0.80,
        }
    
    @property
    def uses_reliable_visibility(self) -> bool:
        """
        Return True if this estimator's visibility values can reliably distinguish
        which side of the body faces the camera.
        
        MediaPipe reports ~1.0 visibility for all landmarks even when occluded,
        so visibility can't be used to determine webcam position.
        
        OpenPose/MoveNet report 0 or low visibility for occluded landmarks,
        so visibility CAN be used for side detection.
        
        Returns:
            True if visibility values reliably indicate occlusion, False otherwise.
        """
        return True  # Default: visibility is reliable (OpenPose, MoveNet, PoseNet)
    
    @property
    def uses_smoothing(self) -> bool:
        """
        Return True if this estimator's landmarks should be smoothed with a moving average.
        
        Some models (like MoveNet) have noisy/jittery outputs that benefit from smoothing.
        Others (like MediaPipe) already apply internal smoothing.
        
        Returns:
            True to apply moving average smoothing to landmarks, False otherwise.
        """
        return True  # Default: apply smoothing (most models benefit from it)
    
    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the underlying pose estimation model.
        
        This method should load model weights and prepare the estimator
        for processing frames. Called once before process() is used.
        
        Raises:
            RuntimeError: If initialization fails
        """
        pass
    
    @abstractmethod
    def process(self, frame: np.ndarray) -> PoseResult:
        """
        Process a single frame and return detected landmarks.
        
        Args:
            frame: BGR image as numpy array (OpenCV format)
        
        Returns:
            PoseResult with standardized landmarks
        
        Raises:
            RuntimeError: If estimator is not initialized
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """
        Release any resources held by the estimator.
        
        Called when the estimator is no longer needed.
        """
        pass
    
    def __enter__(self):
        """Context manager entry - initialize the estimator."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.cleanup()
        return False
