#!/usr/bin/env python3
"""
Benchmark results plotter.

Generates comparison charts from benchmark JSON files.

Usage:
    python -m benchmark.plotter
    python -m benchmark.plotter --input benchmark_results/
    python -m benchmark.plotter --output comparison.png
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_all_results(results_dir: Path) -> Dict[str, Dict]:
    """Load and merge all benchmark results from JSON files."""
    all_results = {}
    
    for json_file in sorted(results_dir.glob("*.json")):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            
            results = data.get("results", {})
            for estimator_name, stats in results.items():
                # Keep the latest result for each estimator
                all_results[estimator_name] = stats
                
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load {json_file}: {e}")
    
    return all_results


def create_comparison_plots(results: Dict[str, Dict], output_path: Path) -> None:
    """Create comparison bar charts for all metrics."""
    
    if not results:
        print("No results to plot!")
        return
    
    estimators = list(results.keys())
    n_estimators = len(estimators)
    
    # Extract metrics
    inference_times = [results[e]["inference_time_ms"]["avg"] for e in estimators]
    theoretical_fps = [1000 / t if t > 0 else 0 for t in inference_times]
    memory_usage = [results[e]["memory_mb"]["avg"] for e in estimators]
    cpu_usage = [results[e]["cpu_percent"]["avg"] for e in estimators]
    detection_rates = [results[e]["detection_rate"] for e in estimators]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Pose Estimator Benchmark Comparison", fontsize=16, fontweight="bold")
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_estimators))
    
    # 1. Inference Time (lower is better)
    ax1 = axes[0, 0]
    bars1 = ax1.bar(estimators, inference_times, color=colors)
    ax1.set_ylabel("Inference Time (ms)")
    ax1.set_title("⏱️ Inference Time (lower is better)")
    ax1.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars1, inference_times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Theoretical Max FPS (higher is better)
    ax2 = axes[0, 1]
    bars2 = ax2.bar(estimators, theoretical_fps, color=colors)
    ax2.set_ylabel("Max FPS")
    ax2.set_title("🚀 Max FPS (higher is better)")
    ax2.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars2, theoretical_fps):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.0f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Memory Usage (lower is better)
    ax3 = axes[0, 2]
    bars3 = ax3.bar(estimators, memory_usage, color=colors)
    ax3.set_ylabel("Memory (MB)")
    ax3.set_title("💾 Memory Usage (lower is better)")
    ax3.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars3, memory_usage):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'{val:.0f}', ha='center', va='bottom', fontsize=9)
    
    # 4. CPU Usage (lower is better for efficiency)
    ax4 = axes[1, 0]
    bars4 = ax4.bar(estimators, cpu_usage, color=colors)
    ax4.set_ylabel("CPU Usage (%)")
    ax4.set_title("⚡ CPU Usage (lower is better)")
    ax4.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars4, cpu_usage):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.0f}%', ha='center', va='bottom', fontsize=9)
    
    # 5. Detection Rate (higher is better)
    ax5 = axes[1, 1]
    bars5 = ax5.bar(estimators, detection_rates, color=colors)
    ax5.set_ylabel("Detection Rate (%)")
    ax5.set_title("🎯 Detection Rate (higher is better)")
    ax5.set_ylim(0, 110)
    ax5.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars5, detection_rates):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 6. Summary Table
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    table_data = []
    headers = ["Model", "Infer(ms)", "Max FPS", "Mem(MB)", "CPU%", "Det%"]
    
    for i, e in enumerate(estimators):
        table_data.append([
            e,
            f"{inference_times[i]:.1f}",
            f"{theoretical_fps[i]:.0f}",
            f"{memory_usage[i]:.0f}",
            f"{cpu_usage[i]:.0f}",
            f"{detection_rates[i]:.0f}",
        ])
    
    table = ax6.table(
        cellText=table_data,
        colLabels=headers,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax6.set_title("📊 Summary Table", fontsize=12, fontweight="bold", pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved to: {output_path}")
    
    # Also show the plot
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument(
        "--input",
        type=str,
        default="benchmark_results",
        help="Directory containing benchmark JSON files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results/comparison.png",
        help="Output path for the comparison plot",
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.input)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return 1
    
    print(f"📂 Loading results from: {results_dir}")
    results = load_all_results(results_dir)
    
    if not results:
        print("No benchmark results found!")
        return 1
    
    print(f"📊 Found {len(results)} estimators: {', '.join(results.keys())}")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    create_comparison_plots(results, output_path)
    
    return 0


if __name__ == "__main__":
    exit(main())
