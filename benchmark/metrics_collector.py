"""
Metrics collector for real-time performance measurement.

This module provides the MetricsCollector class for tracking FPS, inference time,
CPU usage, and memory consumption during pose estimation benchmarks.
"""

import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import psutil


@dataclass
class MetricStats:
    """Statistical summary for a single metric."""
    min: float
    max: float
    avg: float
    std: float
    samples: int

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "min": round(self.min, 2),
            "max": round(self.max, 2),
            "avg": round(self.avg, 2),
            "std": round(self.std, 2),
            "samples": self.samples,
        }


@dataclass
class FrameMetrics:
    """Metrics captured for a single frame."""
    inference_time_ms: float
    fps: float
    cpu_percent: float
    memory_mb: float
    detection_success: bool


class MetricsCollector:
    """
    Collects real-time performance metrics during pose estimation.
    
    Tracks FPS, inference time, CPU usage, and memory for each frame,
    then provides aggregate statistics.
    
    Usage:
        collector = MetricsCollector()
        collector.start()
        
        for frame in frames:
            collector.begin_frame()
            result = estimator.process(frame)
            collector.end_frame(success=result.success)
        
        stats = collector.get_statistics()
    """

    def __init__(self, warmup_frames: int = 10):
        """
        Initialize the metrics collector.
        
        Args:
            warmup_frames: Number of initial frames to discard (warmup period)
        """
        self.warmup_frames = warmup_frames
        self._metrics: List[FrameMetrics] = []
        self._frame_count = 0
        self._start_time: Optional[float] = None
        self._frame_start_time: Optional[float] = None
        self._last_frame_time: Optional[float] = None
        self._process = psutil.Process()
        self._in_warmup = True

    def start(self) -> None:
        """Start the benchmark timer."""
        self._start_time = time.perf_counter()
        self._last_frame_time = self._start_time
        self._frame_count = 0
        self._metrics = []
        self._in_warmup = True

    def begin_frame(self) -> None:
        """Mark the beginning of frame processing."""
        self._frame_start_time = time.perf_counter()

    def end_frame(self, success: bool = True) -> FrameMetrics:
        """
        Mark the end of frame processing and record metrics.
        
        Args:
            success: Whether pose detection was successful for this frame
            
        Returns:
            FrameMetrics for this frame
        """
        end_time = time.perf_counter()
        
        # Calculate inference time
        inference_time_ms = (end_time - self._frame_start_time) * 1000
        
        # Calculate FPS from time since last frame
        if self._last_frame_time is not None:
            frame_duration = end_time - self._last_frame_time
            fps = 1.0 / frame_duration if frame_duration > 0 else 0
        else:
            fps = 0
        
        self._last_frame_time = end_time
        
        # Get CPU and memory usage
        cpu_percent = self._process.cpu_percent()
        memory_info = self._process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        
        metrics = FrameMetrics(
            inference_time_ms=inference_time_ms,
            fps=fps,
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            detection_success=success,
        )
        
        self._frame_count += 1
        
        # Skip warmup frames
        if self._frame_count > self.warmup_frames:
            self._in_warmup = False
            self._metrics.append(metrics)
        
        return metrics

    @property
    def is_warming_up(self) -> bool:
        """Check if still in warmup period."""
        return self._in_warmup

    @property
    def frame_count(self) -> int:
        """Total frames processed (including warmup)."""
        return self._frame_count

    @property
    def recorded_frame_count(self) -> int:
        """Frames recorded after warmup."""
        return len(self._metrics)

    def get_elapsed_time(self) -> float:
        """Get elapsed time since start in seconds."""
        if self._start_time is None:
            return 0
        return time.perf_counter() - self._start_time

    def _calculate_stats(self, values: List[float]) -> MetricStats:
        """Calculate min, max, avg, std for a list of values."""
        if not values:
            return MetricStats(min=0, max=0, avg=0, std=0, samples=0)
        
        return MetricStats(
            min=min(values),
            max=max(values),
            avg=statistics.mean(values),
            std=statistics.stdev(values) if len(values) > 1 else 0,
            samples=len(values),
        )

    def get_statistics(self) -> Dict:
        """
        Calculate aggregate statistics for all recorded metrics.
        
        Returns:
            Dictionary with statistics for each metric type
        """
        if not self._metrics:
            return {
                "fps": MetricStats(0, 0, 0, 0, 0).to_dict(),
                "inference_time_ms": MetricStats(0, 0, 0, 0, 0).to_dict(),
                "cpu_percent": MetricStats(0, 0, 0, 0, 0).to_dict(),
                "memory_mb": MetricStats(0, 0, 0, 0, 0).to_dict(),
                "total_frames": 0,
                "successful_detections": 0,
                "detection_rate": 0,
            }
        
        fps_values = [m.fps for m in self._metrics]
        inference_values = [m.inference_time_ms for m in self._metrics]
        cpu_values = [m.cpu_percent for m in self._metrics]
        memory_values = [m.memory_mb for m in self._metrics]
        
        successful = sum(1 for m in self._metrics if m.detection_success)
        total = len(self._metrics)
        
        return {
            "fps": self._calculate_stats(fps_values).to_dict(),
            "inference_time_ms": self._calculate_stats(inference_values).to_dict(),
            "cpu_percent": self._calculate_stats(cpu_values).to_dict(),
            "memory_mb": self._calculate_stats(memory_values).to_dict(),
            "total_frames": total,
            "successful_detections": successful,
            "detection_rate": round(successful / total * 100, 1) if total > 0 else 0,
        }

    def get_current_metrics(self) -> Optional[FrameMetrics]:
        """Get the most recent frame metrics."""
        return self._metrics[-1] if self._metrics else None

    def reset(self) -> None:
        """Reset all collected metrics."""
        self._metrics = []
        self._frame_count = 0
        self._start_time = None
        self._frame_start_time = None
        self._last_frame_time = None
        self._in_warmup = True
