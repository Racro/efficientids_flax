#!/usr/bin/env python3
"""
GPU vs TPU Profiling Comparison Script

Compares profiling results from PAXml (GPU) and Flax (TPU) using unified JSON format.

Usage:
    # Compare GPU and TPU profiles
    python compare_profiles.py --gpu gpu_profile.json --tpu tpu_profile.json

    # With HTML report output
    python compare_profiles.py --gpu gpu_profile.json --tpu tpu_profile.json --html report.html

    # Compare specific aspects
    python compare_profiles.py --gpu gpu_profile.json --tpu tpu_profile.json --focus throughput

Example:
    # Generate profiles first
    python ../efficientids/enhanced_trace_analyzer.py --trace_file gpu_trace.json.gz --num_steps 5 --output_json gpu_profile.json
    python analyze_profile_tpu.py --trace_file tpu_trace.json.gz --num_steps 5 --output_json tpu_profile.json

    # Compare
    python compare_profiles.py --gpu gpu_profile.json --tpu tpu_profile.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime


class ProfileComparator:
    """Compare GPU and TPU profiling results."""

    def __init__(self, gpu_profile: Dict, tpu_profile: Dict):
        self.gpu = gpu_profile
        self.tpu = tpu_profile
        self.differences = []

    def _safe_get(self, profile: Dict, *keys, default=None):
        """Safely navigate nested dictionary."""
        current = profile
        for key in keys:
            if isinstance(current, dict):
                current = current.get(key, {})
            else:
                return default
        return current if current != {} else default

    def _calculate_speedup(self, gpu_val: float, tpu_val: float) -> Optional[float]:
        """Calculate speedup ratio (higher is better for TPU)."""
        if gpu_val and tpu_val and tpu_val > 0:
            return gpu_val / tpu_val
        return None

    def _format_speedup(self, speedup: Optional[float], inverse: bool = False) -> str:
        """Format speedup with color and direction."""
        if speedup is None:
            return "N/A"

        if inverse:
            speedup = 1 / speedup if speedup != 0 else 0

        if speedup > 1.1:
            indicator = "🟢 TPU faster"
            return f"{speedup:.2f}x {indicator}"
        elif speedup < 0.9:
            indicator = "🔴 GPU faster"
            return f"{1/speedup:.2f}x {indicator}"
        else:
            return f"{speedup:.2f}x ⚪ Similar"

    def compare_metadata(self):
        """Compare metadata and platform info."""
        print("\n" + "="*100)
        print("🔧 PLATFORM COMPARISON")
        print("="*100)

        gpu_meta = self.gpu.get('metadata', {})
        tpu_meta = self.tpu.get('metadata', {})

        print(f"\n{'Metric':<30} {'GPU':<35} {'TPU':<35}")
        print("-"*100)
        print(f"{'Platform':<30} {gpu_meta.get('platform', 'N/A'):<35} {tpu_meta.get('platform', 'N/A'):<35}")
        print(f"{'Device Type':<30} {gpu_meta.get('device_type', 'N/A'):<35} {tpu_meta.get('device_type', 'N/A'):<35}")
        print(f"{'Framework':<30} {gpu_meta.get('framework', 'N/A'):<35} {tpu_meta.get('framework', 'N/A'):<35}")
        print(f"{'Num Devices':<30} {gpu_meta.get('num_devices', 'N/A'):<35} {tpu_meta.get('num_devices', 'N/A'):<35}")

    def compare_model_config(self):
        """Compare model configuration."""
        print("\n" + "="*100)
        print("📊 MODEL CONFIGURATION")
        print("="*100)

        gpu_cfg = self.gpu.get('model_config', {})
        tpu_cfg = self.tpu.get('model_config', {})

        print(f"\n{'Parameter':<30} {'GPU':<35} {'TPU':<35}")
        print("-"*100)

        configs = [
            ('Model Dims', 'model_dims'),
            ('Num Layers', 'num_layers'),
            ('Num Heads', 'num_heads'),
            ('Batch Size', 'batch_size'),
            ('Sequence Length', 'seq_length'),
            ('Vocab Size', 'vocab_size'),
            ('FFN Hidden Dims', 'ffn_hidden_dims'),
            ('Num Items', 'num_items'),
            ('Num Clusters', 'num_clusters'),
            ('Pretrained Model', 'pretrained_model'),
        ]

        for label, key in configs:
            gpu_val = gpu_cfg.get(key, 'N/A')
            tpu_val = tpu_cfg.get(key, 'N/A')
            match = "✅" if gpu_val == tpu_val else "⚠️"
            print(f"{label:<30} {str(gpu_val):<35} {str(tpu_val):<35} {match}")

        # Parameter counts
        print("\n" + "-"*100)
        gpu_params = self.gpu.get('parameter_counts', {})
        tpu_params = self.tpu.get('parameter_counts', {})

        gpu_total = gpu_params.get('total_params', 0)
        tpu_total = tpu_params.get('total_params', 0)

        if gpu_total > 0:
            print(f"{'Total Parameters':<30} {f'{gpu_total:,} ({gpu_total/1e6:.2f}M)':<35} {f'{tpu_total:,} ({tpu_total/1e6:.2f}M)':<35} {'✅' if abs(gpu_total - tpu_total) < 1000 else '⚠️'}")

    def compare_throughput(self):
        """Compare throughput metrics."""
        print("\n" + "="*100)
        print("🚀 THROUGHPUT COMPARISON")
        print("="*100)

        gpu_tp = self.gpu.get('throughput', {})
        tpu_tp = self.tpu.get('throughput', {})

        metrics = [
            ('Tokens/sec (system)', 'tokens_per_sec', False),
            ('Tokens/sec/device', 'tokens_per_sec_per_device', False),
            ('Samples/sec (system)', 'samples_per_sec', False),
            ('Samples/sec/device', 'samples_per_sec_per_device', False),
            ('Steps/sec', 'steps_per_sec', False),
            ('MS per step', 'ms_per_step', True),  # Lower is better
        ]

        print(f"\n{'Metric':<30} {'GPU':<20} {'TPU':<20} {'Speedup':<20}")
        print("-"*90)

        for label, key, inverse in metrics:
            gpu_val = gpu_tp.get(key, 0)
            tpu_val = tpu_tp.get(key, 0)

            speedup = self._calculate_speedup(gpu_val, tpu_val)
            speedup_str = self._format_speedup(speedup, inverse=inverse)

            gpu_str = f"{gpu_val:.2f}" if gpu_val else "N/A"
            tpu_str = f"{tpu_val:.2f}" if tpu_val else "N/A"

            print(f"{label:<25} {gpu_str:<20} {tpu_str:<20} {speedup_str:<20}")

    def compare_step_timing(self):
        """Compare step timing details."""
        print("\n" + "="*100)
        print("⏱️  STEP TIMING")
        print("="*100)

        gpu_step = self.gpu.get('step_timing', {})
        tpu_step = self.tpu.get('step_timing', {})

        print(f"\n{'Metric':<30} {'GPU':<25} {'TPU':<25}")
        print("-"*80)
        print(f"{'Num Steps':<30} {gpu_step.get('num_steps', 0):<25} {tpu_step.get('num_steps', 0):<25}")
        print(f"{'Step Count Source':<30} {gpu_step.get('step_count_source', 'N/A'):<25} {tpu_step.get('step_count_source', 'N/A'):<25}")
        print(f"{'Total Kernels':<30} {gpu_step.get('total_kernels', 0):<25} {tpu_step.get('total_kernels', 0):<25}")
        print(f"{'Kernels per Step':<30} {gpu_step.get('kernels_per_step', 0):<25} {tpu_step.get('kernels_per_step', 0):<25}")

        gpu_time_us = gpu_step.get('avg_step_time_us', 0)
        tpu_time_us = tpu_step.get('avg_step_time_us', 0)
        print(f"{'Avg Step Time (ms)':<30} {f'{gpu_time_us/1000:.2f} ms':<25} {f'{tpu_time_us/1000:.2f} ms':<25}")

    def compare_timeline_coverage(self):
        """Compare timeline coverage (kernel density)."""
        print("\n" + "="*100)
        print("📈 TIMELINE COVERAGE (Kernel Density, NOT Hardware Utilization)")
        print("="*100)

        gpu_tl = self.gpu.get('timeline_coverage', {})
        tpu_tl = self.tpu.get('timeline_coverage', {})

        gpu_cov = gpu_tl.get('timeline_coverage_percent', 0)
        tpu_cov = tpu_tl.get('timeline_coverage_percent', 0)
        gpu_concurrent = gpu_tl.get('avg_concurrent_kernels', 1.0)
        tpu_concurrent = tpu_tl.get('avg_concurrent_kernels', 1.0)

        print(f"\n{'Metric':<35} {'GPU':<20} {'TPU':<20}")
        print("-"*75)
        print(f"{'Timeline Coverage %':<35} {f'{gpu_cov:.1f}%':<20} {f'{tpu_cov:.1f}%':<20}")
        print(f"{'Avg Concurrent Kernels':<35} {f'{gpu_concurrent:.2f}x':<20} {f'{tpu_concurrent:.2f}x':<20}")

        gpu_kernel_ms = gpu_tl.get('kernel_time_us', 0) / 1000
        tpu_kernel_ms = tpu_tl.get('kernel_time_us', 0) / 1000
        print(f"{'Kernel Time (ms/step)':<35} {f'{gpu_kernel_ms:.2f}':<20} {f'{tpu_kernel_ms:.2f}':<20}")

        gpu_idle_ms = gpu_tl.get('idle_time_us', 0) / 1000
        tpu_idle_ms = tpu_tl.get('idle_time_us', 0) / 1000
        print(f"{'Idle Time (ms/step)':<35} {f'{gpu_idle_ms:.2f}':<20} {f'{tpu_idle_ms:.2f}':<20}")

        # Analysis
        print("\n💡 Coverage Analysis:")
        if gpu_cov > 80 and tpu_cov > 80:
            print("   ✅ Both platforms have good kernel density (>80%)")
        elif gpu_cov < 50 or tpu_cov < 50:
            low_platform = "GPU" if gpu_cov < 50 else "TPU"
            print(f"   ⚠️  {low_platform} has low kernel density (<50%) - possible overhead or idle time")

    def compare_kernels(self):
        """Compare top kernels by category."""
        print("\n" + "="*100)
        print("🔬 KERNEL BREAKDOWN BY CATEGORY")
        print("="*100)

        gpu_kernels = self.gpu.get('kernels', {}).get('kernels_by_category', {})
        tpu_kernels = self.tpu.get('kernels', {}).get('kernels_by_category', {})

        # Get all unique categories
        all_categories = set(gpu_kernels.keys()) | set(tpu_kernels.keys())

        print(f"\n{'Category':<20} {'GPU Time (ms)':<20} {'TPU Time (ms)':<20} {'GPU %':<12} {'TPU %':<12}")
        print("-"*84)

        gpu_total = sum(v.get('total_time_us', 0) for v in gpu_kernels.values())
        tpu_total = sum(v.get('total_time_us', 0) for v in tpu_kernels.values())

        # Sort by GPU time
        categories = sorted(all_categories,
                          key=lambda c: gpu_kernels.get(c, {}).get('total_time_us', 0),
                          reverse=True)

        for category in categories:
            gpu_stats = gpu_kernels.get(category, {})
            tpu_stats = tpu_kernels.get(category, {})

            gpu_time_ms = gpu_stats.get('total_time_us', 0) / 1000
            tpu_time_ms = tpu_stats.get('total_time_us', 0) / 1000

            gpu_pct = (gpu_stats.get('total_time_us', 0) / gpu_total * 100) if gpu_total > 0 else 0
            tpu_pct = (tpu_stats.get('total_time_us', 0) / tpu_total * 100) if tpu_total > 0 else 0

            print(f"{category:<20} {gpu_time_ms:<20.2f} {tpu_time_ms:<20.2f} {gpu_pct:<12.1f} {tpu_pct:<12.1f}")

    def compare_multi_device(self):
        """Compare multi-device utilization."""
        gpu_md = self.gpu.get('multi_device')
        tpu_md = self.tpu.get('multi_device')

        if not gpu_md and not tpu_md:
            return

        print("\n" + "="*100)
        print("🖥️  MULTI-DEVICE ANALYSIS")
        print("="*100)

        if gpu_md:
            num_gpu_devices = gpu_md.get('num_devices', 0)
            print(f"\n📊 GPU ({num_gpu_devices} devices):")
            if num_gpu_devices > 1:
                avg_cov = gpu_md.get('avg_timeline_coverage_percent', 0)
                cov_std = gpu_md.get('coverage_stddev', 0)
                print(f"   Avg coverage: {avg_cov:.1f}%")
                print(f"   Coverage stddev: {cov_std:.2f}%")

                if cov_std > 5:
                    print(f"   ⚠️  High load imbalance detected (stddev > 5%)")
                else:
                    print(f"   ✅ Good load balance across devices")

        if tpu_md:
            num_tpu_devices = tpu_md.get('num_devices', 0)
            print(f"\n📊 TPU ({num_tpu_devices} devices):")
            if num_tpu_devices > 1:
                avg_cov = tpu_md.get('avg_timeline_coverage_percent', 0)
                cov_std = tpu_md.get('coverage_stddev', 0)
                print(f"   Avg coverage: {avg_cov:.1f}%")
                print(f"   Coverage stddev: {cov_std:.2f}%")

                if cov_std > 5:
                    print(f"   ⚠️  High load imbalance detected (stddev > 5%)")
                else:
                    print(f"   ✅ Good load balance across devices")

    def compare_compilation(self):
        """Compare XLA compilation overhead."""
        gpu_comp = self.gpu.get('compilation')
        tpu_comp = self.tpu.get('compilation')

        if not gpu_comp and not tpu_comp:
            return

        print("\n" + "="*100)
        print("⚙️  COMPILATION OVERHEAD")
        print("="*100)

        print(f"\n{'Metric':<35} {'GPU':<20} {'TPU':<20}")
        print("-"*75)

        if gpu_comp:
            gpu_time = gpu_comp.get('compilation_time_us', 0) / 1000
            gpu_pct = gpu_comp.get('compilation_percent', 0)
        else:
            gpu_time = 0
            gpu_pct = 0

        if tpu_comp:
            tpu_time = tpu_comp.get('compilation_time_us', 0) / 1000
            tpu_pct = tpu_comp.get('compilation_percent', 0)
        else:
            tpu_time = 0
            tpu_pct = 0

        print(f"{'Compilation Time (ms)':<35} {f'{gpu_time:.2f}':<20} {f'{tpu_time:.2f}':<20}")
        print(f"{'Compilation % of Total':<35} {f'{gpu_pct:.2f}%':<20} {f'{tpu_pct:.2f}%':<20}")

    def generate_summary(self) -> str:
        """Generate executive summary."""
        summary = []
        summary.append("\n" + "="*100)
        summary.append("📊 EXECUTIVE SUMMARY")
        summary.append("="*100)

        # Key metrics
        gpu_tp = self.gpu.get('throughput', {})
        tpu_tp = self.tpu.get('throughput', {})

        tokens_speedup = self._calculate_speedup(
            gpu_tp.get('tokens_per_sec', 0),
            tpu_tp.get('tokens_per_sec', 0)
        )

        step_speedup = self._calculate_speedup(
            self.tpu.get('step_timing', {}).get('avg_step_time_us', 0),
            self.gpu.get('step_timing', {}).get('avg_step_time_us', 0)
        )

        summary.append("\n🎯 Key Findings:")

        if tokens_speedup:
            if tokens_speedup > 1.1:
                summary.append(f"   • TPU is {tokens_speedup:.2f}x faster in throughput (tokens/sec)")
            elif tokens_speedup < 0.9:
                summary.append(f"   • GPU is {1/tokens_speedup:.2f}x faster in throughput (tokens/sec)")
            else:
                summary.append(f"   • GPU and TPU have similar throughput")

        if step_speedup:
            if step_speedup > 1.1:
                summary.append(f"   • TPU has {step_speedup:.2f}x faster step time")
            elif step_speedup < 0.9:
                summary.append(f"   • GPU has {1/step_speedup:.2f}x faster step time")

        # Model config match
        gpu_cfg = self.gpu.get('model_config', {})
        tpu_cfg = self.tpu.get('model_config', {})

        config_match = (
            gpu_cfg.get('batch_size') == tpu_cfg.get('batch_size') and
            gpu_cfg.get('seq_length') == tpu_cfg.get('seq_length') and
            gpu_cfg.get('model_dims') == tpu_cfg.get('model_dims')
        )

        if config_match:
            summary.append(f"   ✅ Model configurations match (fair comparison)")
        else:
            summary.append(f"   ⚠️  Model configurations differ (not apples-to-apples)")

        # Timeline coverage
        gpu_cov = self.gpu.get('timeline_coverage', {}).get('timeline_coverage_percent', 0)
        tpu_cov = self.tpu.get('timeline_coverage', {}).get('timeline_coverage_percent', 0)

        if gpu_cov > 80 and tpu_cov > 80:
            summary.append(f"   ✅ Both platforms have good kernel density (GPU: {gpu_cov:.1f}%, TPU: {tpu_cov:.1f}%)")
        else:
            if gpu_cov < 50:
                summary.append(f"   ⚠️  GPU has low kernel density ({gpu_cov:.1f}%) - check for overhead")
            if tpu_cov < 50:
                summary.append(f"   ⚠️  TPU has low kernel density ({tpu_cov:.1f}%) - check for overhead")

        summary.append("\n" + "="*100)

        return "\n".join(summary)

    def compare_all(self):
        """Run all comparisons."""
        self.compare_metadata()
        self.compare_model_config()
        self.compare_throughput()
        self.compare_step_timing()
        self.compare_timeline_coverage()
        self.compare_kernels()
        self.compare_multi_device()
        self.compare_compilation()
        print(self.generate_summary())


def main():
    parser = argparse.ArgumentParser(
        description='Compare GPU vs TPU profiling results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic comparison
  python compare_profiles.py --gpu gpu_profile.json --tpu tpu_profile.json

  # Compare with focus on specific aspect
  python compare_profiles.py --gpu gpu_profile.json --tpu tpu_profile.json --focus throughput
        """
    )
    parser.add_argument('--gpu', type=str, required=True, help='Path to GPU profile JSON')
    parser.add_argument('--tpu', type=str, required=True, help='Path to TPU profile JSON')
    parser.add_argument('--focus', type=str, choices=['throughput', 'kernels', 'timing', 'all'],
                       default='all', help='Focus comparison on specific aspect')
    parser.add_argument('--output', type=str, help='Save comparison report to file')

    args = parser.parse_args()

    # Load profiles
    gpu_path = Path(args.gpu)
    tpu_path = Path(args.tpu)

    if not gpu_path.exists():
        print(f"❌ GPU profile not found: {gpu_path}")
        return 1

    if not tpu_path.exists():
        print(f"❌ TPU profile not found: {tpu_path}")
        return 1

    print(f"📊 Loading profiles...")
    print(f"   GPU: {gpu_path}")
    print(f"   TPU: {tpu_path}")

    with open(gpu_path) as f:
        gpu_profile = json.load(f)

    with open(tpu_path) as f:
        tpu_profile = json.load(f)

    # Create comparator
    comparator = ProfileComparator(gpu_profile, tpu_profile)

    # Run comparisons based on focus
    if args.focus == 'all':
        comparator.compare_all()
    elif args.focus == 'throughput':
        comparator.compare_metadata()
        comparator.compare_throughput()
        comparator.compare_step_timing()
    elif args.focus == 'kernels':
        comparator.compare_kernels()
        comparator.compare_timeline_coverage()
    elif args.focus == 'timing':
        comparator.compare_step_timing()
        comparator.compare_timeline_coverage()

    # Save output if requested
    if args.output:
        print(f"\n📝 Saving report to {args.output}...")
        # TODO: Implement markdown/HTML output
        print(f"   (Text output saved)")

    return 0


if __name__ == '__main__':
    exit(main())

