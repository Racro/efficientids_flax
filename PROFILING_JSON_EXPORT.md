# Profiling JSON Export - GPU/TPU Comparison Guide

## Overview

The profiling analyzers now support exporting results to a unified JSON format, enabling clean comparison between GPU (PAXml) and TPU (Flax) profiling runs.

## Quick Start

### On TPU System (efficientids_flax):

```bash
# Run profiling
python train_efficientids.py \
  --config tpu_optimized \
  --enable_profiling \
  --profiler_start_step 150 \
  --profiler_num_steps 5

# Find trace file
TRACE=$(find checkpoints/*/profiler_traces -name "*.trace.json.gz" | head -1)

# Analyze and export to JSON
python analyze_profile_tpu.py \
  --trace_file "$TRACE" \
  --num_steps 5 \
  --output_json gemma_tpu_profile.json
```

### On GPU System (efficientids):

```bash
# Run profiling (profiling auto-enabled if PROFILER_CAPTURE_STEP is set)
python run_with_logging.py --config Gemma2BConfig128 --mode train --gpu 0

# Find trace file
TRACE=$(find logs_Gemma2BConfig128_*/profiler_traces -name "*.trace.json.gz" | head -1)

# Analyze and export to JSON
python enhanced_trace_analyzer.py \
  --trace_file "$TRACE" \
  --output_json gemma_gpu_profile.json
```

### On Your Laptop (comparison):

```bash
# Copy JSON files from both systems
scp tpu-system:efficientids_flax/gemma_tpu_profile.json .
scp gpu-system:efficientids/gemma_gpu_profile.json .

# Run comparison
python compare_profiles.py \
  --tpu gemma_tpu_profile.json \
  --gpu gemma_gpu_profile.json \
  --output comparison_report.html
```

## Unified JSON Schema

The JSON output follows a unified schema with these main sections:

```json
{
  "metadata": {
    "timestamp": "ISO 8601",
    "platform": "TPU" or "GPU",
    "device_type": "TPU v4-8", "NVIDIA A100", etc.,
    "framework": "Flax" or "PAXml",
    "num_devices": 1
  },
  "model_config": {
    "model_dims": 2048,
    "num_layers": 18,
    "batch_size": 16,
    "seq_length": 128,
    "pretrained_model": "google/gemma-2b"
  },
  "parameter_counts": {
    "total_params": 2500000000,
    "trainable_params": 3500000,  // PAXml only
    "frozen_params": 2496500000   // PAXml only
  },
  "step_timing": {
    "num_steps": 5,
    "avg_step_time_us": 82600,
    "total_kernels": 1200,
    "kernels_per_step": 240
  },
  "throughput": {
    "tokens_per_sec": 25000,
    "samples_per_sec": 12.1,
    "steps_per_sec": 12.1,
    "ms_per_step": 82.6
  },
  "compute_utilization": {
    "compute_util_percent": 72.1,
    "kernel_time_us": 59500,
    "overhead_time_us": 23100
  },
  "kernels": {
    "top_kernels": [...],
    "kernels_by_category": {...}
  },
  "multi_device": {          // Optional (TPU multi-core, multi-GPU)
    "num_devices": 8,
    "device_stats": [...],
    "avg_utilization_percent": 71.5
  },
  "timeline_analysis": {...}, // Optional (TPU-specific)
  "compilation": {...},        // Optional
  "io_operations": {...}       // Optional
}
```

## Metrics Comparison

### ✅ Directly Comparable Metrics

These metrics use identical calculations and can be directly compared:

- `throughput.tokens_per_sec`
- `throughput.samples_per_sec`
- `throughput.steps_per_sec`
- `step_timing.avg_step_time_us`
- `step_timing.kernels_per_step`
- `compute_utilization.compute_util_percent`
- `model_config.*` (all fields)

### ⚠️ Platform-Specific Metrics

These metrics are only available on specific platforms:

**TPU-only (Flax):**
- `multi_device` (if using TPU pods)
- `timeline_analysis.gap_time_percent`
- Detailed XLA operation breakdown

**GPU-only (PAXml):**
- `parameter_counts.trainable_params`
- `parameter_counts.frozen_params`
- Layer freezing detection

## Files

- `profiling_schema.py` - Unified schema definition and helper functions
- `analyze_profile_tpu.py` - TPU profiler analyzer (updated with JSON export)
- `enhanced_trace_analyzer.py` - GPU profiler analyzer (to be updated)
- `compare_profiles.py` - Comparison script (to be created)

## Notes

1. **Profiler code unchanged**: The profiler itself (`train/profiler.py`) doesn't need modification - it just captures traces as before.

2. **Backward compatible**: Console output still works exactly as before. JSON export is optional via `--output_json`.

3. **Step count critical**: For accurate comparison, always specify `--num_steps` explicitly to match the profiling configuration.

4. **Schema version**: Current version is 1.0.0. Future updates will maintain backward compatibility.

## Troubleshooting

### "profiling_schema.py not found"

Make sure `profiling_schema.py` is in the same directory as the analyzer:

```bash
# For TPU system
ls efficientids_flax/profiling_schema.py

# For GPU system
ls efficientids/profiling_schema.py
```

### Missing metrics in JSON

Some optional sections (like `multi_device`) are only included if the data is available. Single-device runs won't have multi-device stats.

### Step count mismatch

If step counts differ between GPU and TPU runs, the comparison script will warn you. Always profile the same number of steps on both platforms.
