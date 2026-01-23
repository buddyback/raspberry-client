#!/usr/bin/env python3
"""
Benchmark runner CLI for pose estimator performance testing.

Usage:
    # Single model benchmark
    python -m benchmark.benchmark_runner --estimator mediapipe --duration 30
    
    # All models benchmark
    python -m benchmark.benchmark_runner --all --duration 30
    
    # Using video file instead of camera
    python -m benchmark.benchmark_runner --estimator movenet_lightning --video test.mp4
"""

import argparse
import os
import sys
import time

# Fix protobuf compatibility issue
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import cv2

from benchmark.metrics_collector import MetricsCollector
from benchmark.results_exporter import ResultsExporter
from detector.pose_estimators import create_estimator

# Available estimators
AVAILABLE_ESTIMATORS = [
    "mediapipe",
    "movenet_lightning",
    "movenet_thunder",
    "posenet",
    "openpose",
]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark pose estimators for performance comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m benchmark.benchmark_runner --estimator mediapipe --duration 30
  python -m benchmark.benchmark_runner --all --duration 60 --output-dir ./results
  python -m benchmark.benchmark_runner --estimator posenet --video sample.mp4
        """,
    )
    
    # Estimator selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--estimator",
        type=str,
        choices=AVAILABLE_ESTIMATORS,
        help="Specific pose estimator to benchmark",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Benchmark all available estimators",
    )
    
    # Duration and warmup
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Duration of benchmark in seconds (default: 60)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup frames to discard (default: 10)",
    )
    
    # Video source
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index to use (default: 0)",
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Path to video file (overrides --camera)",
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory to save results (default: benchmark_results)",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Skip exporting results to files",
    )
    
    # Display
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--show-video",
        action="store_true",
        help="Display video feed during benchmark (slower)",
    )
    
    return parser.parse_args()


def open_video_source(args):
    """Open video capture from camera or file."""
    if args.video:
        if not os.path.exists(args.video):
            print(f"Error: Video file not found: {args.video}")
            sys.exit(1)
        cap = cv2.VideoCapture(args.video)
        source_name = args.video
    else:
        cap = cv2.VideoCapture(args.camera)
        source_name = f"camera:{args.camera}"
    
    if not cap.isOpened():
        print(f"Error: Could not open video source: {source_name}")
        sys.exit(1)
    
    return cap, source_name


def run_benchmark(
    estimator_name: str,
    cap: cv2.VideoCapture,
    duration: int,
    warmup: int,
    quiet: bool = False,
    show_video: bool = False,
) -> dict:
    """
    Run benchmark for a single estimator.
    
    Args:
        estimator_name: Name of estimator to benchmark
        cap: OpenCV video capture
        duration: Duration in seconds
        warmup: Warmup frames to discard
        quiet: Suppress progress output
        show_video: Display video feed
        
    Returns:
        Statistics dictionary
    """
    if not quiet:
        print(f"\n🚀 Starting benchmark: {estimator_name}")
        print(f"   Duration: {duration}s, Warmup: {warmup} frames")
    
    # Create and initialize estimator
    try:
        estimator_kwargs = {}
        if estimator_name == "mediapipe":
            estimator_kwargs["model_complexity"] = 2
        
        estimator = create_estimator(estimator_name, **estimator_kwargs)
        estimator.initialize()
    except Exception as e:
        print(f"   ❌ Failed to initialize {estimator_name}: {e}")
        return None
    
    # Initialize metrics collector
    collector = MetricsCollector(warmup_frames=warmup)
    
    try:
        collector.start()
        start_time = time.time()
        
        # Reset video to beginning if using file
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while True:
            elapsed = time.time() - start_time
            
            # Check duration
            if elapsed >= duration:
                break
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                # If video file ended, loop back
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break
            
            # Process frame
            collector.begin_frame()
            result = estimator.process(frame)
            metrics = collector.end_frame(success=result.success)
            
            # Progress output
            if not quiet and not collector.is_warming_up:
                if collector.recorded_frame_count % 30 == 0:
                    print(f"   ⏱️  {elapsed:.0f}s / {duration}s | "
                          f"FPS: {metrics.fps:.1f} | "
                          f"CPU: {metrics.cpu_percent:.0f}% | "
                          f"Mem: {metrics.memory_mb:.0f}MB")
            
            # Optional video display
            if show_video:
                cv2.imshow(f"Benchmark: {estimator_name}", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Get final statistics
        stats = collector.get_statistics()
        
        if not quiet:
            print(f"   ✅ Completed: {stats['total_frames']} frames recorded")
        
        return stats
        
    except KeyboardInterrupt:
        if not quiet:
            print("\n   ⚠️  Benchmark interrupted by user")
        return collector.get_statistics()
    
    finally:
        estimator.cleanup()
        if show_video:
            cv2.destroyAllWindows()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Determine which estimators to benchmark
    if args.all:
        estimators = AVAILABLE_ESTIMATORS
    else:
        estimators = [args.estimator]
    
    # Open video source
    cap, source_name = open_video_source(args)
    
    print("=" * 60)
    print("POSE ESTIMATOR BENCHMARK")
    print("=" * 60)
    print(f"Video source: {source_name}")
    print(f"Estimators: {', '.join(estimators)}")
    print(f"Duration per model: {args.duration}s")
    print("=" * 60)
    
    # Run benchmarks
    all_results = {}
    
    try:
        for estimator_name in estimators:
            result = run_benchmark(
                estimator_name=estimator_name,
                cap=cap,
                duration=args.duration,
                warmup=args.warmup,
                quiet=args.quiet,
                show_video=args.show_video,
            )
            
            if result is not None:
                all_results[estimator_name] = result
    
    finally:
        cap.release()
    
    if not all_results:
        print("\n❌ No benchmarks completed successfully")
        sys.exit(1)
    
    # Export results
    exporter = ResultsExporter(output_dir=args.output_dir)
    
    # Print summary
    exporter.print_summary(all_results)
    
    # Export to files
    if not args.no_export:
        config = {
            "duration_seconds": args.duration,
            "warmup_frames": args.warmup,
            "video_source": source_name,
            "estimators": estimators,
        }
        
        files = exporter.export_all(all_results, config)
        
        print(f"\n📁 Results saved to:")
        print(f"   JSON: {files['json']}")
        print(f"   CSV:  {files['csv']}")


if __name__ == "__main__":
    main()
