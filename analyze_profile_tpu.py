#!/usr/bin/env python3
"""
TPU-specific JAX profiler trace analyzer for efficientids_flax.

Analyzes TPU profiling traces collected from training runs to provide insights into:
- Step timing and throughput metrics
- Multi-device TPU utilization and load balancing
- Timeline gap analysis and idle time detection
- XLA compilation overhead
- TPU kernel performance by operation type
- Memory transfer analysis (H2D, D2H, D2D)
- I/O and data loading bottlenecks
- Statistical performance stability (variance analysis)
- Bottleneck identification and recommendations

TPU-specific features:
- TPU System::Execute event detection
- Multi-core TPU profiling support (v2, v3, v4, v5, v6)
- Automatic TPU device count and utilization tracking
- Load imbalance detection across TPU cores
- HLO operation analysis (fusion, collective-permute, etc.)
- TPU memory transfer tracking
- Performance variance and outlier detection

Usage:
    # Basic analysis
    python analyze_profile_tpu.py --trace_file checkpoints/tpu_gemma/profiler_traces/plugins/profile/2025_11_11_09_06_52/t1v-n-bf49aeb0-w-0.trace.json.gz

    # With explicit step count (recommended)
    python analyze_profile_tpu.py --trace_file <path> --num_steps 2

    # Specify checkpoint directory
    python analyze_profile_tpu.py --trace_file <path> --checkpoint_dir checkpoints/tpu_gemma
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
        self.pretrained_model = None  # e.g., "google/gemma-2b"
        self.num_items = None  # For EfficientIDS
        self.num_clusters = None

    def is_valid(self) -> bool:
        """Check if we have minimum required config."""
        return all([self.model_dims, self.num_layers, self.batch_size, self.seq_length])

    def calculate_params(self) -> int:
        """Calculate approximate total parameters."""
        if not self.is_valid():
            return 0

        total_params = 0

        # Calculate transformer params from architecture (works for both pretrained and from-scratch)
        d = self.model_dims
        n = self.num_layers
        vocab = self.vocab_size or 50000
        ffn_dim = self.ffn_hidden_dims or (4 * d)

        # Transformer: Embeddings + Layers
        embed_params = vocab * d
        per_layer = (4 * d * d) + (2 * d * ffn_dim) + (4 * d)  # Attn + FFN + LN
        total_params = embed_params + (n * per_layer)

        # Add EfficientIDS-specific layers (item embeddings, cluster embeddings, projections)
        if self.num_items and self.pretrained_model:
            item_embed_dim = 384
            total_params += self.num_items * item_embed_dim  # Item embeddings
            if self.num_clusters:
                total_params += self.num_clusters * d  # Cluster embeddings
            total_params += item_embed_dim * d  # Projection layer

        return total_params

    def calculate_tokens_per_batch(self) -> int:
        """Calculate total tokens per batch."""
        return self.batch_size * self.seq_length if self.batch_size and self.seq_length else 0


class ProfileAnalyzer:
    """Analyzer for JAX profiling traces."""

    def __init__(self, trace_file: str, checkpoint_dir: Optional[str] = None, num_steps: Optional[int] = None):
        """
        Load trace and extract model config.

        Args:
            trace_file: Path to trace.json.gz
            checkpoint_dir: Optional checkpoint directory (auto-detected if not provided)
            num_steps: Optional number of steps captured (overrides heuristic)
        """
        self.trace_file = trace_file
        self.num_steps_override = num_steps

        # Auto-detect checkpoint directory from trace path
        if checkpoint_dir:
            self.checkpoint_dir = Path(checkpoint_dir)
        else:
            trace_path = Path(trace_file)
            # Go up to find checkpoints directory
            for parent in trace_path.parents:
                if parent.name == 'checkpoints' or (parent / 'config.json').exists():
                    self.checkpoint_dir = parent
                    break
            else:
                # Fallback to parent of profiler_traces
                self.checkpoint_dir = trace_path.parent.parent.parent.parent

        print(f"📊 Loading trace from: {trace_file}")
        print(f"📁 Checkpoint directory: {self.checkpoint_dir}")

        # Load trace
        with gzip.open(trace_file, 'rt') as f:
            self.trace_data = json.load(f)

        self.events = self.trace_data.get('traceEvents', [])
        print(f"   Found {len(self.events)} trace events")

        # Parse events
        self._parse_events()

        # Extract model config
        self.config = self._extract_model_config()

    def _parse_events(self):
        """Parse events by type."""
        self.kernel_events = []
        self.op_events = []
        self.memory_events = []
        self.step_events = []
        self.xla_ops = []
        self.xla_modules = []
        self.compilation_events = []
        self.io_events = []

        # Track device information
        self.devices = {}  # pid -> device name

        kernels_with_metadata = 0
        step_annotations = set()

        for event in self.events:
            name = event.get('name', '')
            cat = event.get('cat', '')
            ph = event.get('ph', '')
            args = event.get('args', {})
            pid = event.get('pid', 0)
            tid = event.get('tid', 0)

            # Track device metadata
            if ph == 'M' and name == 'process_name':
                device_name = args.get('name', '')
                if device_name:
                    self.devices[pid] = device_name

            if ph == 'X':  # Complete events
                # Check if this is a TPU device event (actual compute operations)
                device_name = self.devices.get(pid, '')
                is_tpu_device = 'TPU:' in device_name and '/device:' in device_name

                # TPU HLO operations on device (fusion, copy, etc.)
                if is_tpu_device:
                    # These are the actual TPU compute kernels
                    if not any(x in name.lower() for x in ['process_name', 'thread_name', 'sort_index']):
                        self.kernel_events.append(event)
                        if 'name' in args or 'hlo_op' in args:
                            kernels_with_metadata += 1
                # Host-side operations
                elif 'tpu::system::execute' in name.lower() or ('tfrt' in name.lower() and 'execute' in name.lower()):
                    # Host-side TPU execution coordination (not the actual kernels)
                    self.xla_ops.append(event)
                # Categorize XLA operations
                elif cat == 'XLA Ops' or cat == 'Async XLA Ops':
                    self.xla_ops.append(event)
                elif cat == 'XLA Modules':
                    self.xla_modules.append(event)
                elif 'compile' in name.lower() or 'xla::compile' in name.lower():
                    self.compilation_events.append(event)
                # Data loading / I/O
                elif any(x in name.lower() for x in ['input_pipeline', 'data_iterator', 'prefetch', 'dataset']):
                    self.io_events.append(event)
                # GPU Kernels
                elif 'kernel' in name.lower() or 'cutlass' in name.lower() or '::Kernel' in name:
                    self.kernel_events.append(event)
                    if 'name' in args or 'hlo_op' in args:
                        kernels_with_metadata += 1
                # Memory operations
                elif 'memory' in name.lower() or 'alloc' in name.lower() or 'memcpy' in name.lower() or 'h2d' in name.lower() or 'd2h' in name.lower() or 'd2d' in name.lower():
                    self.memory_events.append(event)
                # XLA/TPU operations
                elif 'xlalinearize' in name.lower() or 'loadprogram' in name.lower():
                    self.xla_ops.append(event)
                else:
                    self.op_events.append(event)

                # Check for step annotations
                if name.lower() in ['train', 'trainstep', 'train_step'] or 'step' in cat.lower():
                    self.step_events.append(event)
                    if 'step' in args:
                        step_annotations.add(args['step'])
                    elif 'step_num' in args:
                        step_annotations.add(args['step_num'])

        self.detected_step_count = len(step_annotations) if step_annotations else None
        num_devices = len([d for d in self.devices.values() if 'TPU' in d or 'GPU' in d])

        print(f"   Parsed: {len(self.kernel_events)} kernels ({kernels_with_metadata} with metadata), "
              f"{len(self.step_events)} step events, {len(self.memory_events)} memory ops")
        print(f"   Found: {len(self.xla_ops)} XLA ops, {len(self.xla_modules)} XLA modules, "
              f"{len(self.compilation_events)} compilation events, {len(self.io_events)} I/O events")
        print(f"   Devices: {num_devices} devices detected")
        if self.detected_step_count:
            print(f"   Detected {self.detected_step_count} unique training steps from trace annotations")

    def _extract_model_config(self) -> ModelConfig:
        """Extract model configuration from checkpoint directory."""
        config = ModelConfig()

        # Try to find config.json
        config_file = self.checkpoint_dir / 'config.json'
        if config_file.exists():
            try:
                with open(config_file) as f:
                    cfg = json.load(f)
                    config.model_dims = cfg.get('embed_dim') or cfg.get('hidden_size') or cfg.get('d_model')
                    config.num_layers = cfg.get('num_layers') or cfg.get('num_hidden_layers')
                    config.num_heads = cfg.get('num_heads') or cfg.get('num_attention_heads')
                    config.vocab_size = cfg.get('vocab_size')
                    config.ffn_hidden_dims = cfg.get('intermediate_size') or cfg.get('ffn_dim')
            except Exception as e:
                print(f"   Warning: Could not parse config.json: {e}")

        # Try to extract from configs/config.py (efficientids_flax style)
        config_py = self.checkpoint_dir.parent / 'configs' / 'config.py'
        if config_py.exists():
            try:
                with open(config_py) as f:
                    content = f.read()

                # Try to find get_tpu_optimized_config function (most likely for TPU profiling)
                tpu_func_start = content.find('def get_tpu_optimized_config')
                if tpu_func_start != -1:
                    # First get function signature to extract default value for pretrained_lm_name
                    func_sig_end = content.find(') ->', tpu_func_start)
                    if func_sig_end == -1:
                        func_sig_end = content.find('):', tpu_func_start)
                    if func_sig_end != -1:
                        func_signature = content[tpu_func_start:func_sig_end]
                        pretrained_param_match = re.search(r'pretrained_lm_name:\s*str\s*=\s*["\']([^"\']+)["\']', func_signature)
                        if pretrained_param_match:
                            config.pretrained_model = pretrained_param_match.group(1)

                    # Extract function body (next 3000 chars should cover it)
                    func_body = content[tpu_func_start:tpu_func_start + 3000]

                    # Extract values from the function
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

                    # Gemma 2B specs
                    config.num_layers = 18
                    config.num_heads = 8

                # Extract Gemma config from core/gemma_model.py for accurate params
                gemma_model_py = self.checkpoint_dir.parent / 'core' / 'gemma_model.py'
                if gemma_model_py.exists() and config.pretrained_model:
                    try:
                        with open(gemma_model_py) as f:
                            gemma_content = f.read()
                            # Look for GemmaConfig class definition
                            vocab_match = re.search(r'vocab_size:\s*int\s*=\s*(\d+)', gemma_content)
                            if vocab_match:
                                config.vocab_size = int(vocab_match.group(1))
                            # Extract FFN dimension
                            ffn_match = re.search(r'intermediate_size.*?(\d+)\s*\*\s*hidden_size', gemma_content)
                            if ffn_match:
                                config.ffn_hidden_dims = int(ffn_match.group(1)) * config.model_dims
                    except Exception as e:
                        print(f"   Warning: Could not parse gemma_model.py: {e}")

                # If TPU config not found, try gemma config
                if not config.model_dims:
                    gemma_func_start = content.find('def get_gemma_config')
                    if gemma_func_start != -1:
                        func_body = content[gemma_func_start:gemma_func_start + 2000]

                        model_dims_match = re.search(r'model_dims\s*=\s*(\d+)', func_body)
                        batch_size_match = re.search(r'batch_size\s*=\s*(\d+)', func_body)
                        max_seq_len_match = re.search(r'max_seq_len\s*=\s*(\d+)', func_body)

                        if model_dims_match:
                            config.model_dims = int(model_dims_match.group(1))
                        if batch_size_match:
                            config.batch_size = int(batch_size_match.group(1))
                        if max_seq_len_match:
                            config.seq_length = int(max_seq_len_match.group(1))

                        config.num_layers = 18
                        config.num_heads = 8

            except Exception as e:
                print(f"   Warning: Could not parse configs/config.py: {e}")

        # Try to extract from training config/logs if available
        train_config = self.checkpoint_dir.parent / 'train_efficientids.py'
        if train_config.exists():
            try:
                with open(train_config) as f:
                    content = f.read()
                    # Look for batch size and sequence length if not found yet
                    if not config.batch_size:
                        batch_match = re.search(r'batch_size["\s:=]+(\d+)', content)
                        if batch_match:
                            config.batch_size = int(batch_match.group(1))
                    if not config.seq_length:
                        seq_match = re.search(r'(?:seq_len|max_seq_length)["\s:=]+(\d+)', content)
                        if seq_match:
                            config.seq_length = int(seq_match.group(1))
            except Exception as e:
                print(f"   Warning: Could not parse training config: {e}")

        print("\n📋 Extracted Model Configuration:")
        print(f"   Model dims: {config.model_dims or 'Unknown'}")
        print(f"   Num layers: {config.num_layers or 'Unknown'}")
        print(f"   Num heads: {config.num_heads or 'Unknown'}")
        print(f"   Batch size: {config.batch_size or 'Unknown'}")
        print(f"   Sequence length: {config.seq_length or 'Unknown'}")
        print(f"   Vocab size: {config.vocab_size or 'Unknown'}")

        if config.is_valid():
            params = config.calculate_params()
            print(f"   Estimated params: {params:,} ({params/1e6:.2f}M)")

        return config

    def _categorize_kernel(self, kernel_name: str, event: Optional[Dict] = None) -> str:
        """Categorize kernel by operation type."""
        name_lower = kernel_name.lower()

        # TPU HLO operations
        if 'fusion' in name_lower:
            return 'Fusion'
        elif 'copy-start' in name_lower or 'copy-done' in name_lower:
            return 'Copy'
        elif 'all-reduce' in name_lower or 'all-gather' in name_lower or 'reduce-scatter' in name_lower:
            return 'Collective'
        elif 'jit_train_step' in name_lower or 'jit_' in name_lower:
            return 'JIT Entry'
        elif 'convert' in name_lower:
            return 'Convert'
        elif 'reduce' in name_lower:
            return 'Reduce'
        elif 'clamp' in name_lower:
            return 'Clamp'
        elif 'bitcast' in name_lower:
            return 'Bitcast'

        # Check metadata first
        if event:
            args = event.get('args', {})
            full_name = args.get('name', '').lower()
            if full_name:
                if 'attention' in full_name or 'attn' in full_name:
                    return 'Attention'
                elif 'ffn' in full_name or 'mlp' in full_name:
                    return 'FFN/MLP'
                elif 'embed' in full_name:
                    return 'Embeddings'
                elif 'norm' in full_name:
                    return 'Normalization'

        # Fallback to kernel name patterns
        if any(x in name_lower for x in ['qkv', 'query', 'key', 'value', 'attention', 'attn']):
            return 'Attention'
        elif any(x in name_lower for x in ['ffn', 'mlp', 'feedforward']):
            return 'FFN/MLP'
        elif any(x in name_lower for x in ['embed', 'vocab', 'lm_head']):
            return 'Embeddings'
        elif any(x in name_lower for x in ['softmax', 'layernorm', 'rmsnorm']):
            return 'Normalization'
        elif any(x in name_lower for x in ['add', 'mul', 'gelu', 'relu', 'silu']):
            return 'Elementwise'
        elif 'gemm' in name_lower or 'dot' in name_lower:
            return 'GEMM'

        return 'Other'

    def analyze_step_times(self) -> Dict[str, Any]:
        """Analyze step timing from kernel patterns."""
        if not self.kernel_events:
            return {}

        kernel_timestamps = [(e.get('ts', 0), e.get('dur', 0)) for e in self.kernel_events]
        kernel_timestamps = [(ts, dur) for ts, dur in kernel_timestamps if ts > 0]
        kernel_timestamps.sort()

        first_ts = kernel_timestamps[0][0]
        last_ts, last_dur = kernel_timestamps[-1]
        total_time_us = (last_ts + last_dur) - first_ts

        total_kernels = len(self.kernel_events)

        # Determine step count
        if self.num_steps_override is not None:
            estimated_steps = self.num_steps_override
            step_source = "command-line"
        elif hasattr(self, 'detected_step_count') and self.detected_step_count is not None:
            estimated_steps = self.detected_step_count
            step_source = "trace"
        else:
            # Heuristic: ~200-300 kernels per step for transformer models
            estimated_steps = max(1, total_kernels // 250)
            step_source = "heuristic"

        avg_step_time_us = total_time_us / estimated_steps

        return {
            'num_steps': estimated_steps,
            'total_us': total_time_us,
            'avg_us': avg_step_time_us,
            'total_kernels': total_kernels,
            'kernels_per_step': total_kernels // estimated_steps if estimated_steps > 0 else 0,
            'step_source': step_source,
        }

    def analyze_kernels_by_operation(self) -> Dict[str, Any]:
        """Analyze kernels grouped by operation type."""
        print("\n🔍 Analyzing kernels by operation type...")

        op_kernels = defaultdict(lambda: {'time_us': 0, 'count': 0})

        for event in self.kernel_events:
            name = event.get('name', 'unknown')
            dur = event.get('dur', 0)
            category = self._categorize_kernel(name, event)

            op_kernels[category]['time_us'] += dur
            op_kernels[category]['count'] += 1

        return dict(op_kernels)

    def analyze_top_kernels(self, top_n: int = 20) -> List[Dict[str, Any]]:
        """Get top kernels with detailed info."""
        print(f"\n🔍 Analyzing top {top_n} kernels...")

        kernel_times = defaultdict(list)

        for event in self.kernel_events:
            name = event.get('name', 'unknown')
            dur = event.get('dur', 0)
            if dur > 0:
                kernel_times[name].append(dur)

        kernel_stats = []
        for name, times in kernel_times.items():
            total = sum(times)
            count = len(times)
            avg = total / count

            # Find an event for this kernel
            event = next((e for e in self.kernel_events if e.get('name') == name), None)
            category = self._categorize_kernel(name, event)

            kernel_stats.append({
                'name': name,
                'category': category,
                'total_us': total,
                'count': count,
                'avg_us': avg,
            })

        kernel_stats.sort(key=lambda x: x['total_us'], reverse=True)
        return kernel_stats[:top_n]

    def estimate_gpu_utilization(self, step_stats: Dict) -> Dict[str, float]:
        """Estimate GPU utilization metrics."""
        if not step_stats or not self.kernel_events:
            return {}

        total_kernel_time_us = sum(e.get('dur', 0) for e in self.kernel_events)
        total_trace_time_us = step_stats.get('total_us', 0)

        gpu_util_pct = (total_kernel_time_us / total_trace_time_us * 100) if total_trace_time_us > 0 else 0

        return {
            'gpu_compute_util_pct': gpu_util_pct,
            'kernel_time_us': total_kernel_time_us,
            'overhead_time_us': total_trace_time_us - total_kernel_time_us,
        }

    def calculate_throughput_metrics(self, step_stats: Dict) -> Dict[str, float]:
        """Calculate throughput in tokens/sec and other metrics."""
        if not step_stats or not self.config.is_valid():
            return {}

        avg_step_ms = step_stats['avg_us'] / 1000
        tokens_per_batch = self.config.calculate_tokens_per_batch()

        if avg_step_ms > 0 and tokens_per_batch > 0:
            tokens_per_sec = (tokens_per_batch / avg_step_ms) * 1000
            samples_per_sec = (self.config.batch_size / avg_step_ms) * 1000
        else:
            tokens_per_sec = 0
            samples_per_sec = 0

        return {
            'tokens_per_sec': tokens_per_sec,
            'samples_per_sec': samples_per_sec,
            'tokens_per_batch': tokens_per_batch,
            'ms_per_step': avg_step_ms,
            'steps_per_sec': 1000 / avg_step_ms if avg_step_ms > 0 else 0,
        }

    def analyze_memory_ops(self) -> Dict[str, Any]:
        """Analyze memory operations."""
        if not self.memory_events:
            return {}

        mem_times = defaultdict(list)
        mem_types = defaultdict(int)
        transfer_stats = {'h2d': [], 'd2h': [], 'd2d': []}

        for event in self.memory_events:
            name = event.get('name', 'unknown')
            dur = event.get('dur', 0)
            if dur > 0:
                mem_times[name].append(dur)

                name_lower = name.lower()
                if 'alloc' in name_lower:
                    mem_types['allocations'] += 1
                elif 'free' in name_lower:
                    mem_types['frees'] += 1
                elif 'h2d' in name_lower or 'hosttodedevice' in name_lower:
                    mem_types['host_to_device'] += 1
                    transfer_stats['h2d'].append(dur)
                elif 'd2h' in name_lower or 'devicetohost' in name_lower:
                    mem_types['device_to_host'] += 1
                    transfer_stats['d2h'].append(dur)
                elif 'd2d' in name_lower or 'devicetodevice' in name_lower:
                    mem_types['device_to_device'] += 1
                    transfer_stats['d2d'].append(dur)
                elif 'copy' in name_lower or 'memcpy' in name_lower:
                    mem_types['copies'] += 1
                else:
                    mem_types['other'] += 1

        total_mem_time = sum(sum(times) for times in mem_times.values())

        # Calculate transfer statistics
        transfer_summary = {}
        for key, durations in transfer_stats.items():
            if durations:
                transfer_summary[key] = {
                    'count': len(durations),
                    'total_us': sum(durations),
                    'avg_us': sum(durations) / len(durations),
                    'max_us': max(durations),
                }

        return {
            'total_time_us': total_mem_time,
            'num_ops': len(self.memory_events),
            'unique_ops': len(mem_times),
            'by_type': dict(mem_types),
            'transfers': transfer_summary,
        }

    def analyze_multi_device(self) -> Dict[str, Any]:
        """Analyze per-device utilization and load balancing."""
        print("\n🔍 Analyzing multi-device utilization...")

        if not self.devices:
            return {}

        # Get compute devices (exclude host)
        compute_devices = {pid: name for pid, name in self.devices.items()
                          if 'TPU' in name or 'GPU' in name}

        if not compute_devices:
            return {}

        device_stats = {}

        for pid, device_name in compute_devices.items():
            # Get all kernel events for this device
            device_kernels = [e for e in self.kernel_events if e.get('pid') == pid]

            if not device_kernels:
                continue

            total_kernel_time = sum(e.get('dur', 0) for e in device_kernels)

            # Find timeline span for this device
            timestamps = [(e.get('ts', 0), e.get('dur', 0)) for e in device_kernels]
            timestamps = [(ts, dur) for ts, dur in timestamps if ts > 0]

            if timestamps:
                timestamps.sort()
                first_ts = timestamps[0][0]
                last_ts, last_dur = timestamps[-1]
                total_time = (last_ts + last_dur) - first_ts

                # Calculate actual utilization accounting for overlaps
                # Sort by start time and merge overlapping intervals
                intervals = [(ts, ts + dur) for ts, dur in timestamps]
                intervals.sort()

                merged = []
                for start, end in intervals:
                    if merged and start <= merged[-1][1]:
                        # Overlapping - extend the last interval
                        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                    else:
                        merged.append((start, end))

                # Actual busy time is sum of merged intervals
                actual_busy_time = sum(end - start for start, end in merged)
                actual_util_pct = (actual_busy_time / total_time * 100) if total_time > 0 else 0

                device_stats[device_name] = {
                    'kernel_count': len(device_kernels),
                    'total_kernel_time_us': total_kernel_time,
                    'actual_busy_time_us': actual_busy_time,
                    'total_time_us': total_time,
                    'utilization_pct': actual_util_pct,
                    'parallelism': total_kernel_time / actual_busy_time if actual_busy_time > 0 else 1,
                }

        # Calculate load imbalance
        if len(device_stats) > 1:
            utils = [s['utilization_pct'] for s in device_stats.values()]
            kernel_counts = [s['kernel_count'] for s in device_stats.values()]

            load_imbalance = {
                'utilization_stddev': statistics.stdev(utils) if len(utils) > 1 else 0,
                'utilization_range': max(utils) - min(utils),
                'kernel_count_stddev': statistics.stdev(kernel_counts) if len(kernel_counts) > 1 else 0,
            }
        else:
            load_imbalance = {}

        return {
            'num_devices': len(device_stats),
            'device_stats': device_stats,
            'load_imbalance': load_imbalance,
        }

    def analyze_timeline_gaps(self) -> Dict[str, Any]:
        """Analyze kernel launch gaps and idle time."""
        print("\n🔍 Analyzing timeline gaps and idle time...")

        if not self.kernel_events:
            return {}

        # Sort kernels by start time
        kernel_timeline = []
        for event in self.kernel_events:
            ts = event.get('ts', 0)
            dur = event.get('dur', 0)
            if ts > 0:
                kernel_timeline.append((ts, ts + dur))

        kernel_timeline.sort()

        # Find gaps between consecutive kernels
        gaps = []
        for i in range(len(kernel_timeline) - 1):
            end_current = kernel_timeline[i][1]
            start_next = kernel_timeline[i + 1][0]
            gap = start_next - end_current
            if gap > 0:
                gaps.append(gap)

        if not gaps:
            return {}

        total_gap_time = sum(gaps)

        # Categorize gaps
        small_gaps = [g for g in gaps if g < 100]  # < 100 us
        medium_gaps = [g for g in gaps if 100 <= g < 1000]  # 100-1000 us
        large_gaps = [g for g in gaps if g >= 1000]  # >= 1 ms

        return {
            'total_gaps': len(gaps),
            'total_gap_time_us': total_gap_time,
            'avg_gap_us': total_gap_time / len(gaps),
            'max_gap_us': max(gaps),
            'min_gap_us': min(gaps),
            'gap_stddev_us': statistics.stdev(gaps) if len(gaps) > 1 else 0,
            'small_gaps': len(small_gaps),
            'medium_gaps': len(medium_gaps),
            'large_gaps': len(large_gaps),
            'large_gap_time_us': sum(large_gaps),
        }

    def analyze_compilation(self) -> Dict[str, Any]:
        """Analyze XLA compilation overhead."""
        print("\n🔍 Analyzing XLA compilation...")

        compilation_time = sum(e.get('dur', 0) for e in self.compilation_events)

        # Analyze XLA modules
        module_stats = defaultdict(list)
        for event in self.xla_modules:
            name = event.get('name', 'unknown')
            dur = event.get('dur', 0)
            if dur > 0:
                module_stats[name].append(dur)

        # Top modules by time
        top_modules = []
        for name, durations in module_stats.items():
            total = sum(durations)
            top_modules.append({
                'name': name,
                'total_us': total,
                'count': len(durations),
                'avg_us': total / len(durations),
            })
        top_modules.sort(key=lambda x: x['total_us'], reverse=True)

        return {
            'compilation_time_us': compilation_time,
            'num_compilation_events': len(self.compilation_events),
            'num_xla_modules': len(self.xla_modules),
            'num_xla_ops': len(self.xla_ops),
            'top_modules': top_modules[:10],
        }

    def analyze_xla_ops(self, top_n: int = 15) -> Dict[str, Any]:
        """Analyze XLA operations."""
        print(f"\n🔍 Analyzing XLA operations (top {top_n})...")

        if not self.xla_ops:
            return {}

        op_stats = defaultdict(list)
        for event in self.xla_ops:
            name = event.get('name', 'unknown')
            dur = event.get('dur', 0)
            if dur > 0:
                op_stats[name].append(dur)

        # Aggregate
        results = []
        for name, durations in op_stats.items():
            total = sum(durations)
            results.append({
                'name': name,
                'total_us': total,
                'count': len(durations),
                'avg_us': total / len(durations),
                'max_us': max(durations),
            })

        results.sort(key=lambda x: x['total_us'], reverse=True)
        return {'top_ops': results[:top_n]}

    def analyze_io_operations(self) -> Dict[str, Any]:
        """Analyze I/O and data loading operations."""
        print("\n🔍 Analyzing I/O and data loading...")

        if not self.io_events:
            return {}

        io_stats = defaultdict(list)
        for event in self.io_events:
            name = event.get('name', 'unknown')
            dur = event.get('dur', 0)
            if dur > 0:
                io_stats[name].append(dur)

        total_io_time = sum(sum(durations) for durations in io_stats.values())

        # Top I/O operations
        top_io = []
        for name, durations in io_stats.items():
            total = sum(durations)
            top_io.append({
                'name': name,
                'total_us': total,
                'count': len(durations),
                'avg_us': total / len(durations),
            })
        top_io.sort(key=lambda x: x['total_us'], reverse=True)

        return {
            'total_io_time_us': total_io_time,
            'num_io_ops': len(self.io_events),
            'unique_io_ops': len(io_stats),
            'top_io_ops': top_io[:10],
        }

    def analyze_statistical_performance(self) -> Dict[str, Any]:
        """Analyze kernel duration variance and stability."""
        print("\n🔍 Analyzing performance stability...")

        if not self.kernel_events:
            return {}

        # Group kernels by name
        kernel_durations = defaultdict(list)
        for event in self.kernel_events:
            name = event.get('name', 'unknown')
            dur = event.get('dur', 0)
            if dur > 0:
                kernel_durations[name].append(dur)

        # Calculate variance metrics
        high_variance_kernels = []
        for name, durations in kernel_durations.items():
            if len(durations) < 2:
                continue

            avg = sum(durations) / len(durations)
            stddev = statistics.stdev(durations)
            cv = (stddev / avg * 100) if avg > 0 else 0  # Coefficient of variation

            if cv > 10:  # High variance if CV > 10%
                high_variance_kernels.append({
                    'name': name,
                    'avg_us': avg,
                    'stddev_us': stddev,
                    'cv_percent': cv,
                    'count': len(durations),
                    'min_us': min(durations),
                    'max_us': max(durations),
                })

        high_variance_kernels.sort(key=lambda x: x['cv_percent'], reverse=True)

        # Overall kernel duration stats
        all_durations = [e.get('dur', 0) for e in self.kernel_events if e.get('dur', 0) > 0]

        return {
            'num_kernels_analyzed': len(kernel_durations),
            'high_variance_kernels': high_variance_kernels[:10],
            'overall_kernel_stats': {
                'avg_us': sum(all_durations) / len(all_durations) if all_durations else 0,
                'stddev_us': statistics.stdev(all_durations) if len(all_durations) > 1 else 0,
                'min_us': min(all_durations) if all_durations else 0,
                'max_us': max(all_durations) if all_durations else 0,
            },
        }

    def print_summary(self):
        """Print comprehensive summary."""
        print("\n" + "="*120)
        print("🚀 JAX PROFILER ANALYSIS - COMPREHENSIVE REPORT")
        print("="*120)

        # Step timing
        step_stats = self.analyze_step_times()

        # Model info
        if self.config.is_valid():
            print("\n📊 MODEL CONFIGURATION:")
            print(f"   Architecture: {self.config.num_layers} layers × {self.config.model_dims} dims × {self.config.num_heads or '?'} heads")
            params = self.config.calculate_params()
            print(f"   Parameters: {params:,} ({params/1e6:.2f}M)")
            print(f"   Batch size: {self.config.batch_size}")
            print(f"   Sequence length: {self.config.seq_length}")
            print(f"   Tokens per batch: {self.config.calculate_tokens_per_batch():,}")

        # Timing & Throughput
        if step_stats:
            print("\n⏱️  STEP TIMING & THROUGHPUT:")
            step_source = step_stats.get('step_source', 'unknown')
            if step_source == 'trace':
                print(f"   ✅ Step count from trace: {step_stats['num_steps']}")
            elif step_source == 'command-line':
                print(f"   ✅ Step count from CLI: {step_stats['num_steps']}")
            else:
                print(f"   ⚠️  Step count (estimated): {step_stats['num_steps']}")

            print(f"   Average step time: {step_stats['avg_us']/1000:.2f} ms")
            print(f"   Total trace time: {step_stats['total_us']/1e6:.2f} seconds")
            print(f"   Kernels per step: {step_stats['kernels_per_step']}")

            throughput = self.calculate_throughput_metrics(step_stats)
            if throughput:
                print("\n   📈 Throughput:")
                print(f"      • {throughput['tokens_per_sec']:.1f} tokens/sec")
                print(f"      • {throughput['samples_per_sec']:.2f} samples/sec")
                print(f"      • {throughput['steps_per_sec']:.2f} steps/sec")

        # Multi-device analysis
        device_stats = self.analyze_multi_device()
        if device_stats:
            print(f"\n🖥️  MULTI-DEVICE ANALYSIS ({device_stats['num_devices']} devices):")
            for device_name, stats in device_stats['device_stats'].items():
                print(f"\n   {device_name}:")
                print(f"      Utilization: {stats['utilization_pct']:.1f}%")
                print(f"      Kernel count: {stats['kernel_count']}")
                print(f"      Actual busy time: {stats['actual_busy_time_us']/1000:.2f} ms")
                print(f"      Total kernel time: {stats['total_kernel_time_us']/1000:.2f} ms")
                print(f"      Avg parallelism: {stats['parallelism']:.2f}x")

            if device_stats.get('load_imbalance'):
                imbalance = device_stats['load_imbalance']
                print(f"\n   Load Balance Metrics:")
                print(f"      Utilization stddev: {imbalance['utilization_stddev']:.2f}%")
                print(f"      Utilization range: {imbalance['utilization_range']:.2f}%")
                print(f"      Kernel count stddev: {imbalance['kernel_count_stddev']:.1f}")

        # GPU Utilization
        gpu_util = {}
        if step_stats:
            gpu_util = self.estimate_gpu_utilization(step_stats)
            if gpu_util:
                num_steps = step_stats.get('num_steps', 1)
                print("\n🎯 OVERALL DEVICE UTILIZATION:")
                print(f"   Compute utilization: {gpu_util['gpu_compute_util_pct']:.1f}%")
                print(f"   Kernel time: {gpu_util['kernel_time_us']/(1000*num_steps):.2f} ms/step")
                print(f"   Overhead time: {gpu_util['overhead_time_us']/(1000*num_steps):.2f} ms/step")

        # Timeline gaps
        gap_stats = self.analyze_timeline_gaps()
        if gap_stats:
            print("\n⏸️  TIMELINE GAP ANALYSIS:")
            print(f"   Total gaps: {gap_stats['total_gaps']}")
            print(f"   Total idle time: {gap_stats['total_gap_time_us']/1000:.2f} ms ({gap_stats['total_gap_time_us']/step_stats.get('total_us', 1)*100:.1f}% of trace)")
            print(f"   Average gap: {gap_stats['avg_gap_us']:.2f} μs")
            print(f"   Max gap: {gap_stats['max_gap_us']/1000:.2f} ms")
            print(f"   Gap distribution:")
            print(f"      Small (<100μs): {gap_stats['small_gaps']}")
            print(f"      Medium (100μs-1ms): {gap_stats['medium_gaps']}")
            print(f"      Large (≥1ms): {gap_stats['large_gaps']} ({gap_stats['large_gap_time_us']/1000:.2f} ms total)")

        # Compilation analysis
        comp_stats = self.analyze_compilation()
        if comp_stats.get('num_compilation_events', 0) > 0 or comp_stats.get('num_xla_modules', 0) > 0:
            print("\n⚙️  COMPILATION ANALYSIS:")
            if comp_stats['compilation_time_us'] > 0:
                print(f"   Compilation time: {comp_stats['compilation_time_us']/1000:.2f} ms")
            print(f"   XLA modules: {comp_stats['num_xla_modules']}")
            print(f"   XLA ops: {comp_stats['num_xla_ops']}")

            if comp_stats.get('top_modules'):
                print(f"\n   Top XLA Modules by time:")
                for mod in comp_stats['top_modules'][:5]:
                    print(f"      • {mod['name'][:70]}: {mod['total_us']/1000:.2f} ms")

        # XLA ops
        xla_ops = self.analyze_xla_ops(top_n=10)
        if xla_ops.get('top_ops'):
            print("\n🔧 TOP XLA OPERATIONS:")
            print(f"   {'Operation':<60} {'Time(ms)':<12} {'Count':<8} {'Avg(ms)':<10}")
            print(f"   {'-'*95}")
            for op in xla_ops['top_ops']:
                name = op['name'][:57] + '...' if len(op['name']) > 60 else op['name']
                print(f"   {name:<60} {op['total_us']/1000:<12.2f} {op['count']:<8} {op['avg_us']/1000:<10.3f}")

        # I/O analysis
        io_stats = self.analyze_io_operations()
        if io_stats:
            print("\n📥 I/O & DATA LOADING:")
            print(f"   Total I/O time: {io_stats['total_io_time_us']/1000:.2f} ms")
            print(f"   I/O operations: {io_stats['num_io_ops']}")

            if io_stats.get('top_io_ops'):
                print(f"\n   Top I/O operations:")
                for op in io_stats['top_io_ops'][:5]:
                    print(f"      • {op['name'][:70]}: {op['total_us']/1000:.2f} ms ({op['count']} calls)")

        # Operation breakdown
        op_kernels = self.analyze_kernels_by_operation()
        if op_kernels:
            num_steps = step_stats.get('num_steps', 1) if step_stats else 1
            print("\n🔬 KERNEL TIME BY OPERATION TYPE:")
            total_time = sum(v['time_us'] for v in op_kernels.values())

            sorted_ops = sorted(op_kernels.items(), key=lambda x: x[1]['time_us'], reverse=True)

            print(f"   {'Operation':<20} {'Time/step (ms)':<15} {'%':<8} {'Count':<8}")
            print(f"   {'-'*60}")
            for op_name, stats in sorted_ops:
                time_ms_per_step = (stats['time_us'] / 1000) / num_steps
                pct = (stats['time_us'] / total_time * 100) if total_time > 0 else 0
                count = stats['count']
                print(f"   {op_name:<20} {time_ms_per_step:<15.2f} {pct:<8.1f} {count:<8}")

        # Top kernels
        top_kernels = self.analyze_top_kernels(top_n=15)
        if top_kernels:
            num_steps = step_stats.get('num_steps', 1) if step_stats else 1
            print(f"\n🏆 TOP 15 KERNELS (across {num_steps} steps):")
            print(f"   {'Kernel':<50} {'Category':<15} {'Time(ms)':<12} {'Count':<7}")
            print(f"   {'-'*90}")
            for k in top_kernels:
                short_name = k['name'][:47] + '...' if len(k['name']) > 50 else k['name']
                time_ms = k['total_us'] / 1000
                print(f"   {short_name:<50} {k['category']:<15} {time_ms:<12.2f} {k['count']:<7}")

        # Memory operations
        mem_stats = self.analyze_memory_ops()
        if mem_stats:
            print("\n💾 MEMORY OPERATIONS:")
            print(f"   Total time: {mem_stats['total_time_us']/1000:.2f} ms")
            print(f"   Number of ops: {mem_stats['num_ops']}")
            if mem_stats.get('by_type'):
                print("   Breakdown:")
                for op_type, count in mem_stats['by_type'].items():
                    print(f"      • {op_type}: {count}")

            if mem_stats.get('transfers'):
                print("\n   Data Transfers:")
                for transfer_type, stats in mem_stats['transfers'].items():
                    print(f"      {transfer_type.upper()}: {stats['count']} transfers, "
                          f"{stats['total_us']/1000:.2f} ms total, {stats['avg_us']/1000:.2f} ms avg")

        # Statistical performance
        perf_stats = self.analyze_statistical_performance()
        if perf_stats:
            print("\n📊 PERFORMANCE STABILITY:")
            overall = perf_stats['overall_kernel_stats']
            print(f"   Overall kernel duration: {overall['avg_us']:.2f} μs (±{overall['stddev_us']:.2f} μs)")
            print(f"   Range: {overall['min_us']:.2f} - {overall['max_us']:.2f} μs")

            if perf_stats.get('high_variance_kernels'):
                print(f"\n   High Variance Kernels (CV > 10%):")
                print(f"   {'Kernel':<50} {'Avg(μs)':<12} {'StdDev':<12} {'CV%':<8} {'Count':<7}")
                print(f"   {'-'*95}")
                for k in perf_stats['high_variance_kernels'][:5]:
                    name = k['name'][:47] + '...' if len(k['name']) > 50 else k['name']
                    print(f"   {name:<50} {k['avg_us']:<12.2f} {k['stddev_us']:<12.2f} {k['cv_percent']:<8.1f} {k['count']:<7}")

        # Summary
        print("\n" + "="*120)
        print("✅ ANALYSIS COMPLETE")
        print("="*120)

        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        has_recommendations = False

        if step_stats and step_stats.get('step_source') == 'heuristic':
            print("   ⚠️  Step count was estimated. For accurate metrics, re-run with:")
            print("      --num_steps <actual_number_of_steps>")
            has_recommendations = True

        if device_stats and device_stats.get('load_imbalance'):
            imbalance = device_stats['load_imbalance']
            if imbalance.get('utilization_stddev', 0) > 5:
                print(f"   ⚠️  Load imbalance detected (stddev {imbalance['utilization_stddev']:.1f}%):")
                print("      - Check if data is evenly distributed across devices")
                print("      - Verify model parallelism configuration")
                has_recommendations = True

        if gap_stats and gap_stats.get('large_gaps', 0) > 10:
            print(f"   ⚠️  Found {gap_stats['large_gaps']} large gaps (≥1ms):")
            print("      - May indicate kernel launch overhead or synchronization bottlenecks")
            print("      - Consider using async execution or better pipelining")
            has_recommendations = True

        if gpu_util and gpu_util.get('gpu_compute_util_pct', 100) < 50:
            print(f"   • Low device utilization ({gpu_util['gpu_compute_util_pct']:.1f}%) - consider:")
            print(f"     - Increasing batch size")
            print(f"     - Reducing framework overhead")
            print(f"     - Checking for data loading bottlenecks")
            has_recommendations = True

        if perf_stats and perf_stats.get('high_variance_kernels'):
            print(f"   ℹ️  Found {len(perf_stats['high_variance_kernels'])} kernels with high variance")
            print("      - May indicate dynamic workloads or non-deterministic behavior")
            has_recommendations = True

        if not has_recommendations:
            print("   ✅ No major issues detected!")

        print("="*120)


def main():
    parser = argparse.ArgumentParser(
        description='JAX profiler trace analyzer for efficientids_flax',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python analyze_profile.py --trace_file checkpoints/tpu_gemma/profiler_traces/plugins/profile/*/trace.json.gz

  # With explicit step count
  python analyze_profile.py --trace_file trace.json.gz --num_steps 2

  # Specify checkpoint directory
  python analyze_profile.py --trace_file trace.json.gz --checkpoint_dir checkpoints/tpu_gemma
        """
    )
    parser.add_argument('--trace_file', type=str, required=True, help='Path to trace.json.gz')
    parser.add_argument('--checkpoint_dir', type=str, help='Checkpoint directory (auto-detected if not provided)')
    parser.add_argument('--num_steps', type=int, help='Number of steps captured (overrides auto-detection)')

    args = parser.parse_args()

    if not Path(args.trace_file).exists():
        print(f"❌ Error: Trace file not found: {args.trace_file}")
        return 1

    analyzer = ProfileAnalyzer(args.trace_file, args.checkpoint_dir, args.num_steps)
    analyzer.print_summary()

    return 0


if __name__ == '__main__':
    exit(main())

