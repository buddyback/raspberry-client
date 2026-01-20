"""
Results exporter for benchmark data.

This module provides the ResultsExporter class for saving benchmark results
in JSON and CSV formats with system metadata.
"""

import csv
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil


def get_system_info() -> Dict[str, Any]:
    """Gather system information for benchmark metadata."""
    cpu_info = "Unknown"
    try:
        # Try to get CPU model name
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("model name") or line.startswith("Model"):
                    cpu_info = line.split(":")[1].strip()
                    break
    except (FileNotFoundError, PermissionError):
        cpu_info = platform.processor() or "Unknown"
    
    memory = psutil.virtual_memory()
    
    return {
        "platform": platform.system().lower(),
        "platform_release": platform.release(),
        "python_version": sys.version.split()[0],
        "cpu": cpu_info,
        "cpu_count": psutil.cpu_count(logical=True),
        "ram_total_gb": round(memory.total / (1024**3), 1),
    }


class ResultsExporter:
    """
    Exports benchmark results to various formats.
    
    Supports JSON and CSV export with comprehensive metadata
    about the system and benchmark configuration.
    """

    def __init__(self, output_dir: str = "benchmark_results"):
        """
        Initialize the results exporter.
        
        Args:
            output_dir: Directory to save result files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_json(
        self,
        results: Dict[str, Dict],
        config: Optional[Dict] = None,
        filename: Optional[str] = None,
    ) -> Path:
        """
        Export results to a JSON file.
        
        Args:
            results: Dictionary mapping estimator names to their statistics
            config: Benchmark configuration (duration, warmup, etc.)
            filename: Optional custom filename (without extension)
            
        Returns:
            Path to the created JSON file
        """
        timestamp = datetime.now()
        
        if filename is None:
            filename = f"benchmark_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        output_data = {
            "metadata": {
                "timestamp": timestamp.isoformat(),
                "system": get_system_info(),
                "config": config or {},
            },
            "results": results,
        }
        
        filepath = self.output_dir / f"{filename}.json"
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        return filepath

    def export_csv(
        self,
        results: Dict[str, Dict],
        filename: Optional[str] = None,
    ) -> Path:
        """
        Export results to a CSV file for spreadsheet analysis.
        
        Args:
            results: Dictionary mapping estimator names to their statistics
            filename: Optional custom filename (without extension)
            
        Returns:
            Path to the created CSV file
        """
        timestamp = datetime.now()
        
        if filename is None:
            filename = f"benchmark_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        filepath = self.output_dir / f"{filename}.csv"
        
        # Flatten the nested structure for CSV
        rows: List[Dict] = []
        for estimator_name, stats in results.items():
            row = {"estimator": estimator_name}
            
            # Add each metric's statistics
            for metric_name, metric_stats in stats.items():
                if isinstance(metric_stats, dict):
                    for stat_name, value in metric_stats.items():
                        row[f"{metric_name}_{stat_name}"] = value
                else:
                    row[metric_name] = metric_stats
            
            rows.append(row)
        
        if rows:
            fieldnames = rows[0].keys()
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        
        return filepath

    def print_summary(self, results: Dict[str, Dict]) -> None:
        """
        Print a formatted summary of benchmark results to console.
        
        Args:
            results: Dictionary mapping estimator names to their statistics
        """
        print("\n" + "=" * 70)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 70)
        
        for estimator_name, stats in results.items():
            print(f"\n📊 {estimator_name.upper()}")
            print("-" * 40)
            
            fps = stats.get("fps", {})
            print(f"  FPS:            {fps.get('avg', 0):.1f} avg "
                  f"(min: {fps.get('min', 0):.1f}, max: {fps.get('max', 0):.1f})")
            
            inference = stats.get("inference_time_ms", {})
            print(f"  Inference:      {inference.get('avg', 0):.1f}ms avg "
                  f"(min: {inference.get('min', 0):.1f}, max: {inference.get('max', 0):.1f})")
            
            cpu = stats.get("cpu_percent", {})
            print(f"  CPU Usage:      {cpu.get('avg', 0):.1f}% avg "
                  f"(min: {cpu.get('min', 0):.1f}%, max: {cpu.get('max', 0):.1f}%)")
            
            memory = stats.get("memory_mb", {})
            print(f"  Memory:         {memory.get('avg', 0):.1f}MB avg "
                  f"(min: {memory.get('min', 0):.1f}, max: {memory.get('max', 0):.1f})")
            
            total = stats.get("total_frames", 0)
            success = stats.get("successful_detections", 0)
            rate = stats.get("detection_rate", 0)
            print(f"  Detection Rate: {rate}% ({success}/{total} frames)")
        
        print("\n" + "=" * 70)

    def export_all(
        self,
        results: Dict[str, Dict],
        config: Optional[Dict] = None,
    ) -> Dict[str, Path]:
        """
        Export results to both JSON and CSV formats.
        
        Args:
            results: Dictionary mapping estimator names to their statistics
            config: Benchmark configuration
            
        Returns:
            Dictionary with 'json' and 'csv' keys mapping to file paths
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"benchmark_{timestamp}"
        
        return {
            "json": self.export_json(results, config, base_filename),
            "csv": self.export_csv(results, base_filename),
        }
