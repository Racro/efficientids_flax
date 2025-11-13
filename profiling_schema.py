#!/usr/bin/env python3
"""
Unified Profiling Schema for EfficientIDS - TPU/GPU Compatible

This schema defines a common JSON output format for profiling data from both
PAXml (GPU) and Flax (TPU) implementations, enabling clean cross-platform comparison.

Usage:
    from profiling_schema import create_unified_result, save_profiling_json

    result = create_unified_result(
        platform="TPU",
        device_type="TPU v4-8",
        ...metrics...
    )
    save_profiling_json(result, "tpu_profile.json")
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
import json


def create_unified_result(
    # Metadata
    platform: str,  # "GPU" or "TPU"
    device_type: str,  # "NVIDIA A100", "TPU v4-8", etc.
    trace_file: str,
    framework: str,  # "PAXml" or "Flax"
    num_devices: int = 1,
    experiment_name: Optional[str] = None,
    config_name: Optional[str] = None,

    # Model Config
    model_dims: Optional[int] = None,
    num_layers: Optional[int] = None,
    num_heads: Optional[int] = None,
    batch_size: Optional[int] = None,
    seq_length: Optional[int] = None,
    vocab_size: Optional[int] = None,
    ffn_hidden_dims: Optional[int] = None,
    num_items: Optional[int] = None,
    num_clusters: Optional[int] = None,
    item_embedding_dim: Optional[int] = None,
    pretrained_model: Optional[str] = None,

    # Parameters
    total_params: int = 0,
    trainable_params: Optional[int] = None,
    frozen_params: Optional[int] = None,
    has_frozen_layers: bool = False,

    # Step Timing
    num_steps: int = 0,
    total_time_us: float = 0.0,
    avg_step_time_us: float = 0.0,
    total_kernels: int = 0,
    kernels_per_step: float = 0.0,
    step_count_source: str = "heuristic",

    # Throughput
    tokens_per_sec: float = 0.0,
    samples_per_sec: float = 0.0,
    steps_per_sec: float = 0.0,
    ms_per_step: float = 0.0,
    tokens_per_batch: int = 0,
    tokens_per_sec_per_device: Optional[float] = None,
    samples_per_sec_per_device: Optional[float] = None,

    # Timeline Coverage (NOT hardware utilization!)
    # This measures % of time with kernels executing, not how efficiently hardware is used
    # For true HW utilization, use nvidia-smi (GPU) or cloud monitoring (TPU)
    timeline_coverage_percent: float = 0.0,  # % of timeline covered by kernels
    kernel_time_us: float = 0.0,  # Time with kernels executing (after overlap removal)
    idle_time_us: float = 0.0,  # Time with no kernels (gaps)
    avg_concurrent_kernels: float = 1.0,  # Average number of kernels executing simultaneously

    # Kernels
    top_kernels: Optional[List[Dict[str, Any]]] = None,
    kernels_by_category: Optional[Dict[str, Dict[str, Any]]] = None,

    # Memory Operations
    memory_ops: Optional[Dict[str, Any]] = None,

    # Multi-device (optional)
    multi_device: Optional[Dict[str, Any]] = None,

    # Timeline Analysis (optional)
    timeline_gaps: Optional[Dict[str, Any]] = None,

    # Compilation (optional)
    compilation: Optional[Dict[str, Any]] = None,

    # I/O Operations (optional)
    io_operations: Optional[Dict[str, Any]] = None,

    # Statistical Performance (optional)
    statistical_performance: Optional[Dict[str, Any]] = None,

    # XLA Operations (optional)
    xla_operations: Optional[Dict[str, Any]] = None,

) -> Dict[str, Any]:
    """
    Create a unified profiling result dictionary.

    Returns a dictionary with the complete profiling results in unified format.
    """
    result = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "platform": platform,
            "device_type": device_type,
            "trace_file": trace_file,
            "framework": framework,
            "num_devices": num_devices,
            "profiler_version": "1.0.0",
        },
        "model_config": {
            "model_dims": model_dims,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "batch_size": batch_size,
            "seq_length": seq_length,
            "vocab_size": vocab_size,
            "ffn_hidden_dims": ffn_hidden_dims,
            "num_items": num_items,
            "num_clusters": num_clusters,
            "item_embedding_dim": item_embedding_dim,
            "pretrained_model": pretrained_model,
        },
        "parameter_counts": {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "frozen_params": frozen_params,
            "has_frozen_layers": has_frozen_layers,
        },
        "step_timing": {
            "num_steps": num_steps,
            "total_time_us": total_time_us,
            "avg_step_time_us": avg_step_time_us,
            "total_kernels": total_kernels,
            "kernels_per_step": kernels_per_step,
            "step_count_source": step_count_source,
        },
        "throughput": {
            "tokens_per_sec": tokens_per_sec,
            "samples_per_sec": samples_per_sec,
            "steps_per_sec": steps_per_sec,
            "ms_per_step": ms_per_step,
            "tokens_per_batch": tokens_per_batch,
            "tokens_per_sec_per_device": tokens_per_sec_per_device,
            "samples_per_sec_per_device": samples_per_sec_per_device,
            "_note_per_device": "Per-device metrics are effective (total / num_devices), not actual single-device throughput",
        },
        "timeline_coverage": {
            "_note": "Timeline coverage, NOT hardware utilization. Use nvidia-smi/cloud monitoring for HW metrics.",
            "timeline_coverage_percent": timeline_coverage_percent,
            "kernel_time_us": kernel_time_us,
            "idle_time_us": idle_time_us,
            "avg_concurrent_kernels": avg_concurrent_kernels,
        },
        "kernels": {
            "top_kernels": top_kernels or [],
            "kernels_by_category": kernels_by_category or {},
        },
    }

    # Add optional fields if provided
    if experiment_name:
        result["metadata"]["experiment_name"] = experiment_name
    if config_name:
        result["metadata"]["config_name"] = config_name

    if memory_ops:
        result["memory_operations"] = memory_ops
    if multi_device:
        result["multi_device"] = multi_device
    if timeline_gaps:
        result["timeline_analysis"] = timeline_gaps
    if compilation:
        result["compilation"] = compilation
    if io_operations:
        result["io_operations"] = io_operations
    if statistical_performance:
        result["statistical_performance"] = statistical_performance
    if xla_operations:
        result["xla_operations"] = xla_operations

    # Remove None values recursively
    result = _remove_none_values(result)

    return result


def _remove_none_values(d: Any) -> Any:
    """Recursively remove None values from dictionary."""
    if isinstance(d, dict):
        return {k: _remove_none_values(v) for k, v in d.items() if v is not None}
    elif isinstance(d, list):
        return [_remove_none_values(item) for item in d if item is not None]
    else:
        return d


def save_profiling_json(result: Dict[str, Any], filepath: str, indent: int = 2):
    """Save profiling result to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(result, f, indent=indent)
    print(f"✅ Profiling results saved to: {filepath}")


def load_profiling_json(filepath: str) -> Dict[str, Any]:
    """Load profiling result from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def print_profiling_summary(result: Dict[str, Any]):
    """Print human-readable summary of profiling results."""
    metadata = result.get("metadata", {})
    model = result.get("model_config", {})
    params = result.get("parameter_counts", {})
    step_timing = result.get("step_timing", {})
    throughput = result.get("throughput", {})
    timeline = result.get("timeline_coverage", {})
    multi_device = result.get("multi_device")

    print("\n" + "="*100)
    print(f"📊 UNIFIED PROFILING SUMMARY - {metadata.get('platform', '?')}")
    print("="*100)

    print(f"\n🔧 Platform: {metadata.get('platform', '?')} ({metadata.get('device_type', '?')})")
    print(f"   Framework: {metadata.get('framework', '?')}")
    print(f"   Devices: {metadata.get('num_devices', 1)}")
    print(f"   Timestamp: {metadata.get('timestamp', '?')}")

    print(f"\n📐 Model: {model.get('num_layers', '?')} layers × "
          f"{model.get('model_dims', '?')} dims × "
          f"{model.get('num_heads', '?')} heads")

    if params.get('total_params', 0) > 0:
        print(f"   Parameters: {params['total_params']/1e6:.2f}M total", end="")
        if params.get('trainable_params'):
            print(f", {params['trainable_params']/1e6:.2f}M trainable", end="")
        if params.get('frozen_params'):
            print(f", {params['frozen_params']/1e6:.2f}M frozen", end="")
        print()

    print(f"\n⏱️  Timing:")
    print(f"   Steps: {step_timing.get('num_steps', 0)}")
    print(f"   Avg step: {step_timing.get('avg_step_time_us', 0)/1000:.2f} ms")
    print(f"   Kernels/step: {step_timing.get('kernels_per_step', 0):.0f}")

    print(f"\n🚀 Throughput:")
    print(f"   {throughput.get('tokens_per_sec', 0):.0f} tokens/sec (system)")
    if throughput.get('tokens_per_sec_per_device'):
        print(f"   {throughput.get('tokens_per_sec_per_device', 0):.0f} tokens/sec/device (effective)")
    print(f"   {throughput.get('samples_per_sec', 0):.1f} samples/sec (system)")
    if throughput.get('samples_per_sec_per_device'):
        print(f"   {throughput.get('samples_per_sec_per_device', 0):.1f} samples/sec/device (effective)")
    print(f"   {throughput.get('steps_per_sec', 0):.2f} steps/sec")

    print(f"\n⏱️  Timeline Coverage:")
    print(f"   ⚠️  NOTE: Timeline coverage, NOT hardware utilization!")
    print(f"   Coverage: {timeline.get('timeline_coverage_percent', 0):.1f}%")
    print(f"   Avg concurrent kernels: {timeline.get('avg_concurrent_kernels', 1.0):.2f}x")
    print(f"   Kernel time: {timeline.get('kernel_time_us', 0)/1000:.2f} ms")
    print(f"   Idle time: {timeline.get('idle_time_us', 0)/1000:.2f} ms")

    if multi_device and multi_device.get('num_devices', 0) > 1:
        print(f"\n🖥️  Multi-Device ({multi_device['num_devices']} devices):")
        print(f"   Avg coverage: {multi_device.get('avg_timeline_coverage_percent', 0):.1f}%")
        if multi_device.get('coverage_stddev'):
            print(f"   Load imbalance (stddev): {multi_device['coverage_stddev']:.1f}%")

    print("\n" + "="*100)


if __name__ == "__main__":
    print("📋 Unified Profiling Schema v1.0")
    print("="*60)
    print("\nThis schema defines a common format for profiling data:")
    print("  • GPU (PAXml) and TPU (Flax) compatible")
    print("  • Superset of all metrics from both implementations")
    print("  • Structured JSON output for comparison")
    print("\nUsage:")
    print("  from profiling_schema import create_unified_result, save_profiling_json")
    print("  result = create_unified_result(platform='TPU', ...)")
    print("  save_profiling_json(result, 'tpu_profile.json')")
