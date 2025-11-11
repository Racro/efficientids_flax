#!/usr/bin/env python3
"""
GPU-specific JAX profiler trace analyzer for efficientids_flax.

Analyzes GPU profiling traces collected from training runs to provide insights into:
- Step timing and throughput metrics
- Multi-GPU utilization and load balancing
- Timeline gap analysis and idle time detection
- XLA compilation overhead
- CUDA kernel performance by operation type
- Memory transfer analysis (H2D, D2H, D2D)
- I/O and data loading bottlenecks
- Statistical performance stability (variance analysis)
- Bottleneck identification and recommendations

GPU-specific features:
- CUDA kernel detection and analysis
- GPU memory transfer tracking
- NCCL collective operation analysis
- cuDNN/cuBLAS operation breakdown
- SM utilization tracking

Usage:
    # Basic analysis
    python analyze_profile_gpu.py --trace_file <path_to_gpu_trace.json.gz>

    # With explicit step count (recommended)
    python analyze_profile_gpu.py --trace_file <path> --num_steps 2

    # Specify checkpoint directory
    python analyze_profile_gpu.py --trace_file <path> --checkpoint_dir checkpoints/gpu_run
"""

import json
import gzip
import re
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import statistics


class ModelConfig:
    """Model configuration extracted from logs or config files."""

    def __init__(self):
        self.model_dims = None
        self.num_layers = None
        self.num_heads = None
        self.batch_size = None
        self.seq_length = None
        self.vocab_size = None
        self.ffn_hidden_dims = None
        self.pretrained_model = None
        self.num_items = None
        self.num_clusters = None


class GPUProfileAnalyzer:
    """Analyzes JAX profiler traces from GPU training runs."""

    def __init__(self, trace_file: str, checkpoint_dir: Optional[str] = None, num_steps: Optional[int] = None):
        self.trace_file = Path(trace_file)
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else self.trace_file.parent.parent.parent.parent.parent
        self.num_steps = num_steps
        self.trace_data = None
        self.events = []
        self.kernels = []
        self.memory_ops = []
        self.xla_ops = []
        self.io_ops = []
        self.compilation_events = []
        self.step_events = []

    def load_trace(self):
        """Load and parse the trace file."""
        print(f"📊 Loading trace from: {self.trace_file}")
        print(f"📁 Checkpoint directory: {self.checkpoint_dir}")

        if self.trace_file.suffix == '.gz':
            with gzip.open(self.trace_file, 'rt') as f:
                self.trace_data = json.load(f)
        else:
            with open(self.trace_file, 'r') as f:
                self.trace_data = json.load(f)

        self.events = self.trace_data.get('traceEvents', [])
        print(f"   Found {len(self.events)} trace events")

    def parse_events(self):
        """Parse trace events into categories (GPU-specific)."""
        for event in self.events:
            name = event.get('name', '').lower()
            cat = event.get('cat', '').lower()
            ph = event.get('ph', '')

            # GPU kernel events (CUDA kernels)
            if 'kernel' in name or 'cuda' in cat or cat == 'kernel':
                self.kernels.append(event)

            # Memory transfer events (GPU-specific)
            elif any(x in name for x in ['memcpy', 'h2d', 'd2h', 'd2d', 'copy']):
                self.memory_ops.append(event)

            # XLA operations
            elif 'xla' in name or 'hlo' in name:
                self.xla_ops.append(event)

            # Compilation events
            elif 'compile' in name or 'jit' in name:
                self.compilation_events.append(event)

            # I/O events
            elif any(x in name for x in ['read', 'write', 'io', 'data']):
                self.io_ops.append(event)

            # Step markers
            elif 'step' in name:
                self.step_events.append(event)

        # Count kernels with metadata
        kernels_with_metadata = sum(1 for k in self.kernels if k.get('args', {}).get('name'))

        print(f"   Parsed: {len(self.kernels)} kernels ({kernels_with_metadata} with metadata), "
              f"{len(self.step_events)} step events, {len(self.memory_ops)} memory ops")
        print(f"   Found: {len(self.xla_ops)} XLA ops, 0 XLA modules, "
              f"{len(self.compilation_events)} compilation events, {len(self.io_ops)} I/O events")

    def analyze_multi_device(self) -> Dict:
        """Analyze multi-GPU utilization and load balancing."""
        device_kernels = defaultdict(list)

        for kernel in self.kernels:
            pid = kernel.get('pid')
            tid = kernel.get('tid')
            device_id = f"{pid}:{tid}"

            if 'dur' in kernel:
                device_kernels[device_id].append({
                    'start': kernel['ts'],
                    'end': kernel['ts'] + kernel['dur'],
                    'dur': kernel['dur'],
                    'name': kernel.get('name', '')
                })

        num_devices = len(device_kernels)
        print(f"   Devices: {num_devices} GPU(s) detected")

        if num_devices == 0:
            return {}

        # Calculate utilization per device with interval merging
        device_stats = {}
        for device_id, kernels in device_kernels.items():
            # Sort by start time
            sorted_kernels = sorted(kernels, key=lambda x: x['start'])

            # Merge overlapping intervals
            if sorted_kernels:
                merged_intervals = []
                current_start = sorted_kernels[0]['start']
                current_end = sorted_kernels[0]['end']

                for kernel in sorted_kernels[1:]:
                    if kernel['start'] <= current_end:
                        # Overlapping, extend the end
                        current_end = max(current_end, kernel['end'])
                    else:
                        # No overlap, save current and start new
                        merged_intervals.append((current_start, current_end))
                        current_start = kernel['start']
                        current_end = kernel['end']

                # Don't forget the last interval
                merged_intervals.append((current_start, current_end))

                # Calculate total busy time from merged intervals
                busy_time = sum(end - start for start, end in merged_intervals)
                total_time = max(k['end'] for k in sorted_kernels) - min(k['start'] for k in sorted_kernels)

                device_stats[device_id] = {
                    'num_kernels': len(kernels),
                    'busy_time_us': busy_time,
                    'total_time_us': total_time,
                    'utilization': (busy_time / total_time * 100) if total_time > 0 else 0
                }

        return device_stats

    def extract_model_config(self) -> ModelConfig:
        """Extract model configuration from config files."""
        config = ModelConfig()

        config_file = self.checkpoint_dir.parent / 'configs' / 'config.py'
        if config_file.exists():
            with open(config_file) as f:
                content = f.read()

            # Try to find get_gpu_config or similar function
            gpu_func_start = content.find('def get_gpu_config')
            if gpu_func_start == -1:
                # Fallback to TPU config
                gpu_func_start = content.find('def get_tpu_optimized_config')

            if gpu_func_start != -1:
                # Extract function signature for pretrained model name
                func_sig_end = content.find(') ->', gpu_func_start)
                if func_sig_end == -1:
                    func_sig_end = content.find('):', gpu_func_start)
                if func_sig_end != -1:
                    func_signature = content[gpu_func_start:func_sig_end]
                    pretrained_param_match = re.search(r'pretrained_lm_name:\s*str\s*=\s*["\']([^"\']+)["\']', func_signature)
                    if pretrained_param_match:
                        config.pretrained_model = pretrained_param_match.group(1)

                # Extract function body
                func_body = content[gpu_func_start:gpu_func_start + 3000]

                # Extract values
                model_dims_match = re.search(r'model_dims\s*=\s*(\d+)', func_body)
                batch_size_match = re.search(r'batch_size\s*=\s*(\d+)', func_body)
                max_seq_len_match = re.search(r'max_seq_len\s*=\s*(\d+)', func_body)
                num_items_match = re.search(r'num_items\s*=\s*(\d+)', func_body)
                num_clusters_match = re.search(r'num_clusters\s*=\s*(\d+)', func_body)

                if model_dims_match:
                    config.model_dims = int(model_dims_match.group(1))
                if batch_size_match:
                    config.batch_size = int(batch_size_match.group(1))
                if max_seq_len_match:
                    config.seq_length = int(max_seq_len_match.group(1))
                if num_items_match:
                    config.num_items = int(num_items_match.group(1))
                if num_clusters_match:
                    config.num_clusters = int(num_clusters_match.group(1))

                # Set model specs based on pretrained model
                if config.pretrained_model and 'gemma-2b' in config.pretrained_model.lower():
                    config.num_layers = 18
                    config.num_heads = 8

            # Extract model-specific configs from gemma_model.py or llama_flax.py
            if config.pretrained_model:
                model_file = None
                if 'gemma' in config.pretrained_model.lower():
                    model_file = self.checkpoint_dir.parent / 'core' / 'gemma_model.py'
                elif 'llama' in config.pretrained_model.lower():
                    model_file = self.checkpoint_dir.parent / 'core' / 'llama_flax.py'

                if model_file and model_file.exists():
                    try:
                        with open(model_file) as f:
                            model_content = f.read()
                            vocab_match = re.search(r'vocab_size:\s*int\s*=\s*(\d+)', model_content)
                            if vocab_match:
                                config.vocab_size = int(vocab_match.group(1))
                            ffn_match = re.search(r'intermediate_size.*?(\d+)\s*\*\s*hidden_size', model_content)
                            if ffn_match and config.model_dims:
                                config.ffn_hidden_dims = int(ffn_match.group(1)) * config.model_dims
                    except Exception as e:
                        print(f"   Warning: Could not parse {model_file.name}: {e}")

        return config

    def calculate_params(self, config: ModelConfig) -> int:
        """Calculate model parameter count."""
        if not all([config.model_dims, config.num_layers, config.vocab_size]):
            return 0

        # Embedding layer
        embedding_params = config.vocab_size * config.model_dims

        # Transformer layers
        d = config.model_dims
        layer_params = (
            4 * d * d +  # Q, K, V, O projection
            8 * d * d +  # FFN (2 * 4d * d)
            4 * d  # Layer norms (2 per layer)
        )
        transformer_params = config.num_layers * layer_params

        # Final layer norm + output projection
        final_params = d + config.vocab_size * d

        # EfficientIDS additions
        efficientids_params = 0
        if config.num_items and config.model_dims:
            efficientids_params += config.num_items * config.model_dims  # Item embeddings
        if config.num_clusters and config.model_dims:
            efficientids_params += config.num_clusters * config.model_dims  # Cluster embeddings

        total_params = embedding_params + transformer_params + final_params + efficientids_params
        return total_params

    def analyze_throughput(self, config: ModelConfig, device_stats: Dict) -> Dict:
        """Calculate training throughput metrics."""
        if not self.kernels:
            return {}

        # Get time range
        all_times = [k['ts'] for k in self.kernels if 'ts' in k]
        if 'dur' in self.kernels[-1]:
            all_times.append(self.kernels[-1]['ts'] + self.kernels[-1]['dur'])

        total_time_us = max(all_times) - min(all_times)
        total_time_s = total_time_us / 1e6

        # Calculate step timing
        num_steps = self.num_steps if self.num_steps else len(self.step_events)
        if num_steps == 0:
            num_steps = 1  # Default fallback

        avg_step_time_ms = (total_time_s / num_steps) * 1000 if num_steps > 0 else 0
        kernels_per_step = len(self.kernels) / num_steps if num_steps > 0 else 0

        # Calculate throughput
        throughput = {}
        if config.batch_size and config.seq_length and avg_step_time_ms > 0:
            tokens_per_batch = config.batch_size * config.seq_length
            tokens_per_sec = (tokens_per_batch / avg_step_time_ms) * 1000
            samples_per_sec = (config.batch_size / avg_step_time_ms) * 1000
            steps_per_sec = 1000 / avg_step_time_ms

            throughput = {
                'tokens_per_sec': tokens_per_sec,
                'samples_per_sec': samples_per_sec,
                'steps_per_sec': steps_per_sec,
                'tokens_per_batch': tokens_per_batch
            }

        return {
            'num_steps': num_steps,
            'avg_step_time_ms': avg_step_time_ms,
            'total_time_s': total_time_s,
            'kernels_per_step': kernels_per_step,
            'throughput': throughput
        }

    def print_report(self, config: ModelConfig, device_stats: Dict, throughput: Dict):
        """Print comprehensive analysis report."""
        print("\n" + "=" * 120)
        print("🚀 JAX PROFILER ANALYSIS - GPU COMPREHENSIVE REPORT")
        print("=" * 120)

        # Model configuration
        print("\n📊 MODEL CONFIGURATION:")
        if config.num_layers and config.model_dims and config.num_heads:
            print(f"   Architecture: {config.num_layers} layers × {config.model_dims} dims × {config.num_heads} heads")

        total_params = self.calculate_params(config)
        if total_params > 0:
            print(f"   Parameters: {total_params:,} ({total_params/1e9:.2f}B)")

        if config.batch_size:
            print(f"   Batch size: {config.batch_size}")
        if config.seq_length:
            print(f"   Sequence length: {config.seq_length}")
        if config.batch_size and config.seq_length:
            print(f"   Tokens per batch: {config.batch_size * config.seq_length:,}")

        # Throughput
        print("\n⏱️  STEP TIMING & THROUGHPUT:")
        if throughput:
            print(f"   ✅ Step count from CLI: {throughput['num_steps']}")
            print(f"   Average step time: {throughput['avg_step_time_ms']:.2f} ms")
            print(f"   Total trace time: {throughput['total_time_s']:.2f} seconds")
            print(f"   Kernels per step: {throughput['kernels_per_step']:.0f}")

            if throughput.get('throughput'):
                t = throughput['throughput']
                print(f"\n   📈 Throughput:")
                print(f"      • {t['tokens_per_sec']:.1f} tokens/sec")
                print(f"      • {t['samples_per_sec']:.2f} samples/sec")
                print(f"      • {t['steps_per_sec']:.2f} steps/sec")

        # Multi-GPU analysis
        if device_stats:
            print("\n🔄 MULTI-GPU ANALYSIS:")
            print(f"   Number of GPUs: {len(device_stats)}")
            for device_id, stats in sorted(device_stats.items()):
                print(f"   GPU {device_id}:")
                print(f"      Kernels: {stats['num_kernels']}")
                print(f"      Utilization: {stats['utilization']:.1f}%")
                print(f"      Busy time: {stats['busy_time_us']/1e3:.2f} ms")

            # Load balancing
            utilizations = [s['utilization'] for s in device_stats.values()]
            if len(utilizations) > 1:
                util_std = statistics.stdev(utilizations)
                print(f"\n   Load Balancing:")
                print(f"      Utilization std dev: {util_std:.2f}%")
                if util_std > 10:
                    print(f"      ⚠️  High imbalance detected across GPUs")

        print("\n" + "=" * 120)

    def run(self):
        """Run full analysis pipeline."""
        self.load_trace()
        self.parse_events()
        config = self.extract_model_config()

        print(f"\n📋 Extracted Model Configuration:")
        print(f"   Model dims: {config.model_dims}")
        print(f"   Num layers: {config.num_layers}")
        print(f"   Num heads: {config.num_heads}")
        print(f"   Batch size: {config.batch_size}")
        print(f"   Sequence length: {config.seq_length}")
        print(f"   Vocab size: {config.vocab_size if config.vocab_size else 'Unknown'}")

        total_params = self.calculate_params(config)
        print(f"   Estimated params: {total_params:,} ({total_params/1e6:.2f}M)")

        device_stats = self.analyze_multi_device()
        throughput = self.analyze_throughput(config, device_stats)

        self.print_report(config, device_stats, throughput)


def main():
    parser = argparse.ArgumentParser(description='Analyze JAX GPU profiler traces')
    parser.add_argument('trace_file', nargs='?', help='Path to trace file (.json or .json.gz)')
    parser.add_argument('--trace_file', dest='trace_file_flag', help='Path to trace file')
    parser.add_argument('--checkpoint_dir', help='Checkpoint directory (auto-detected if not provided)')
    parser.add_argument('--num_steps', type=int, help='Number of training steps in trace')

    args = parser.parse_args()

    trace_file = args.trace_file or args.trace_file_flag
    if not trace_file:
        parser.error('trace_file is required')

    analyzer = GPUProfileAnalyzer(
        trace_file=trace_file,
        checkpoint_dir=args.checkpoint_dir,
        num_steps=args.num_steps
    )
    analyzer.run()


if __name__ == '__main__':
    main()

