"""
Pose Estimators Package.

This package provides a unified interface for various pose estimation models:
- MediaPipe Pose (default, no additional dependencies)
- MoveNet (requires tensorflow, tensorflow-hub)
- PoseNet (requires tensorflow)


Usage:
    from detector.pose_estimators import create_estimator, EstimatorType
    
    # Create a MediaPipe estimator
    estimator = create_estimator(EstimatorType.MEDIAPIPE, model_complexity=2)
    
    # Or use string name
    estimator = create_estimator("movenet_lightning")
    
    # Initialize and use
    estimator.initialize()
    result = estimator.process(frame)
    if result.success:
        landmarks = result.landmarks
"""

from enum import Enum
from typing import Union

from .base import Landmark, PoseEstimator, PoseResult, STANDARD_LANDMARKS


class EstimatorType(Enum):
    """Available pose estimation models."""
    MEDIAPIPE = "mediapipe"
    MOVENET_LIGHTNING = "movenet_lightning"
    MOVENET_THUNDER = "movenet_thunder"
    POSENET = "posenet"

    OPENPOSE = "openpose"


def create_estimator(
    estimator_type: Union[EstimatorType, str],
    **kwargs
) -> PoseEstimator:
    """
    Factory function to create a pose estimator.
    
    Args:
        estimator_type: Type of estimator to create (EstimatorType enum or string)
        **kwargs: Additional arguments passed to the estimator constructor
    
    Returns:
        PoseEstimator instance (not yet initialized)
    
    Raises:
        ValueError: If the estimator type is unknown
        ImportError: If required dependencies are not installed
    
    Examples:
        # Using enum
        estimator = create_estimator(EstimatorType.MEDIAPIPE, model_complexity=2)
        
        # Using string
        estimator = create_estimator("movenet_lightning")
    """
    # Convert string to enum if needed
    if isinstance(estimator_type, str):
        try:
            estimator_type = EstimatorType(estimator_type.lower())
        except ValueError:
            valid_types = [e.value for e in EstimatorType]
            raise ValueError(
                f"Unknown estimator type: '{estimator_type}'. "
                f"Valid options are: {valid_types}"
            )
    
    # Create the appropriate estimator
    if estimator_type == EstimatorType.MEDIAPIPE:
        from .mediapipe_estimator import MediaPipePoseEstimator
        return MediaPipePoseEstimator(**kwargs)
    
    elif estimator_type in (EstimatorType.MOVENET_LIGHTNING, EstimatorType.MOVENET_THUNDER):
        from .movenet_estimator import MoveNetPoseEstimator
        variant = "lightning" if estimator_type == EstimatorType.MOVENET_LIGHTNING else "thunder"
        return MoveNetPoseEstimator(variant=variant, **kwargs)
    
    elif estimator_type == EstimatorType.POSENET:
        from .posenet_estimator import PoseNetPoseEstimator
        return PoseNetPoseEstimator(**kwargs)
    

    
    elif estimator_type == EstimatorType.OPENPOSE:
        from .openpose_estimator import OpenPosePoseEstimator
        return OpenPosePoseEstimator(**kwargs)
    

    
    else:
        raise ValueError(f"Unknown estimator type: {estimator_type}")


def list_available_estimators() -> list:
    """
    List all available estimator types.
    
    Returns:
        List of (name, description) tuples for each available estimator
    """
    return [
        ("mediapipe", "Google MediaPipe Pose - 33 landmarks, no extra dependencies"),
        ("movenet_lightning", "MoveNet Lightning - 17 landmarks, fast, requires TensorFlow"),
        ("movenet_thunder", "MoveNet Thunder - 17 landmarks, accurate, requires TensorFlow"),
        ("posenet", "PoseNet - 17 landmarks, TFLite, requires TensorFlow"),

        ("openpose", "OpenPose - 18 landmarks, PyTorch Lightweight, requires torch"),
    ]


__all__ = [
    "EstimatorType",
    "PoseEstimator",
    "PoseResult",
    "Landmark",
    "STANDARD_LANDMARKS",
    "create_estimator",
    "list_available_estimators",
]
