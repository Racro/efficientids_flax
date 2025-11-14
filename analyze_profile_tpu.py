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
- Automatic step detection from jit_train_step events
- Automatic incomplete step detection and exclusion
- TPU System::Execute event detection
- Multi-core TPU profiling support (v2, v3, v4, v5, v6)
- Automatic TPU device count and utilization tracking
- Load imbalance detection across TPU cores
- HLO operation analysis (fusion, collective-permute, etc.)
- TPU memory transfer tracking
- Performance variance and outlier detection

Usage:
    # Basic analysis (auto-detects steps and model config)
    python analyze_profile_tpu.py --trace_file t1v-n-bf49aeb0-w-0.trace.json.gz

    # Specify checkpoint directory for model config
    python analyze_profile_tpu.py --trace_file <path> --checkpoint_dir checkpoints/tpu_gemma

    # Export to JSON for GPU/TPU comparison
    python analyze_profile_tpu.py --trace_file <path> --output_json tpu_profile.json
"""

import json
import gzip
import re
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import statistics

# Import unified profiling schema
try:
    from profiling_schema import create_unified_result, save_profiling_json, print_profiling_summary
    HAS_SCHEMA = True
except ImportError:
    HAS_SCHEMA = False
    print("⚠️  Warning: profiling_schema.py not found. JSON export disabled.")


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
                    item_embedding_dim_match = re.search(r'item_embedding_dim\s*=\s*(\d+)', func_body)

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
                    if item_embedding_dim_match:
                        config.item_embedding_dim = int(item_embedding_dim_match.group(1))

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

    def _detect_training_steps(self) -> List[Dict]:
        """Detect actual jit_train_step events and their timing.

        Returns list of step info dicts with start_us, end_us, duration_us.
        """
        # Find TPU devices
        tpu_pids = [pid for pid, name in self.devices.items() if 'TPU:' in name]
        if not tpu_pids:
            return []

        # Find all jit_train_step events
        all_steps = []
        for pid in tpu_pids:
            steps = [e for e in self.kernel_events
                    if 'jit_train_step' in e.get('name', '') and e.get('pid') == pid]
            all_steps.extend([(pid, e) for e in steps])

        if not all_steps:
            return []

        # Group by step (all devices execute simultaneously)
        all_steps.sort(key=lambda x: x[1]['ts'])
        step_groups = []
        current_group = []
        last_ts = 0

        for pid, e in all_steps:
            if current_group and (e['ts'] - last_ts) > 10000:  # >10ms = new step
                step_groups.append(current_group)
                current_group = []
            current_group.append((pid, e))
            last_ts = e['ts']

        if current_group:
            step_groups.append(current_group)

        # Calculate duration for each step
        step_info = []
        for step_num, group in enumerate(step_groups):
            min_start = min(e['ts'] for pid, e in group)
            max_end = max(e['ts'] + e['dur'] for pid, e in group)
            duration = max_end - min_start

            step_info.append({
                'step_num': step_num + 1,
                'start_us': min_start,
                'end_us': max_end,
                'duration_us': duration,
                'num_devices': len(group)
            })

        return step_info

    def analyze_step_times(self) -> Dict[str, Any]:
        """Analyze step timing from kernel patterns."""
        if not self.kernel_events:
            return {}

        # FIXED: Use max end time instead of last event by timestamp
        timestamps = [(e.get('ts', 0), e.get('dur', 0)) for e in self.kernel_events]
        timestamps = [(ts, dur) for ts, dur in timestamps if ts > 0]

        first_ts = min(ts for ts, dur in timestamps)
        last_end_ts = max(ts + dur for ts, dur in timestamps)
        total_time_us = last_end_ts - first_ts

        total_kernels = len(self.kernel_events)

        # Detect actual training steps from trace
        detected_steps = self._detect_training_steps()

        # Determine step count and handle incomplete steps
        if detected_steps:
            num_steps = len(detected_steps)

            # Check if last step is incomplete (significantly shorter than others)
            if num_steps >= 2:
                durations = [s['duration_us'] for s in detected_steps]
                last_duration = durations[-1]
                avg_earlier = sum(durations[:-1]) / len(durations[:-1])

                # If last step is <80% of average, it's incomplete
                if last_duration < 0.8 * avg_earlier:
                    print(f"   ⚠️  Last step is incomplete ({last_duration/1000:.2f}ms vs {avg_earlier/1000:.2f}ms avg)")
                    print(f"   Using first {num_steps-1} complete steps only")

                    complete_steps = detected_steps[:-1]
                    num_complete = len(complete_steps)

                    # Calculate time from first to last COMPLETE step
                    total_time_us = complete_steps[-1]['end_us'] - complete_steps[0]['start_us']
                    avg_step_time_us = total_time_us / num_complete

                    return {
                        'num_steps': num_complete,
                        'total_steps_in_trace': num_steps,
                        'incomplete_steps': 1,
                        'total_us': total_time_us,
                        'avg_us': avg_step_time_us,
                        'total_kernels': total_kernels,
                        'kernels_per_step': total_kernels // num_complete if num_complete > 0 else 0,
                        'step_source': 'trace-detected-complete-only',
                    }

            # All steps are complete
            avg_step_time_us = total_time_us / num_steps

            return {
                'num_steps': num_steps,
                'total_us': total_time_us,
                'avg_us': avg_step_time_us,
                'total_kernels': total_kernels,
                'kernels_per_step': total_kernels // num_steps if num_steps > 0 else 0,
                'step_source': 'trace-detected',
            }
        else:
            # No jit_train_step events found - fallback to heuristic
            print(f"   ⚠️  Warning: No jit_train_step events detected in trace")
            print(f"   Falling back to heuristic estimation")

            # Heuristic: ~200-300 kernels per step for transformer models
            estimated_steps = max(1, total_kernels // 250)
            avg_step_time_us = total_time_us / estimated_steps

            return {
                'num_steps': estimated_steps,
                'total_us': total_time_us,
                'avg_us': avg_step_time_us,
                'total_kernels': total_kernels,
                'kernels_per_step': total_kernels // estimated_steps if estimated_steps > 0 else 0,
                'step_source': 'heuristic',
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

    def estimate_timeline_coverage(self, step_stats: Dict, multi_device_stats: Optional[Dict] = None) -> Dict[str, float]:
        """Calculate timeline coverage metrics.

        IMPORTANT: This measures % of timeline covered by kernels, NOT hardware utilization.
        For true hardware utilization, use nvidia-smi (GPU) or cloud monitoring (TPU).

        Timeline coverage = % of trace time where at least one kernel is executing.
        - High (>80%): Dense kernel execution, minimal idle time
        - Low (<50%): Many gaps, launch overhead, or data loading bottlenecks

        For multi-device systems, uses per-device stats that account for kernel overlaps.
        """
        if not step_stats or not self.kernel_events:
            return {}

        total_kernel_time_us = sum(e.get('dur', 0) for e in self.kernel_events)
        total_trace_time_us = step_stats.get('total_us', 0)

        # Check if we have multi-device stats (more accurate for TPU/multi-GPU)
        if multi_device_stats and multi_device_stats.get('num_devices', 0) > 1:
            # Use average timeline coverage from multi-device analysis (accounts for overlaps)
            device_stats = multi_device_stats.get('device_stats', {})
            if device_stats:
                coverages = [stats['timeline_coverage_pct'] for stats in device_stats.values()]
                avg_coverage = sum(coverages) / len(coverages)

                # Calculate average per-device busy time from device stats
                avg_busy_time = sum(stats['actual_busy_time_us'] for stats in device_stats.values()) / len(device_stats)
                avg_device_timeline = sum(stats['total_time_us'] for stats in device_stats.values()) / len(device_stats)
                avg_idle_time = avg_device_timeline - avg_busy_time

                # Average concurrent kernels across devices
                concurrent_kernels = [stats['avg_concurrent_kernels'] for stats in device_stats.values()]
                avg_concurrent = sum(concurrent_kernels) / len(concurrent_kernels)

                return {
                    'timeline_coverage_pct': avg_coverage,
                    'kernel_time_us': avg_busy_time,
                    'idle_time_us': avg_idle_time,
                    'avg_concurrent_kernels': avg_concurrent,
                    'num_devices_detected': multi_device_stats['num_devices'],
                    'total_kernel_time_all_devices_us': total_kernel_time_us,
                    'calculation_method': 'multi_device_overlap_aware',
                    '_note': 'Timeline coverage, NOT hardware utilization',
                }

        # Single device or no multi-device stats: simple calculation
        num_devices = len([d for d in self.devices.values() if 'TPU' in d or 'GPU' in d])
        if num_devices == 0:
            num_devices = 1

        if num_devices == 1:
            # Single device: calculate with overlap removal for accuracy
            timestamps = [(e.get('ts', 0), e.get('dur', 0)) for e in self.kernel_events]
            timestamps = [(ts, dur) for ts, dur in timestamps if ts > 0]

            if timestamps:
                # Merge overlapping intervals
                intervals = [(ts, ts + dur) for ts, dur in timestamps]
                intervals.sort()

                merged = []
                for start, end in intervals:
                    if merged and start <= merged[-1][1]:
                        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                    else:
                        merged.append((start, end))

                actual_busy_time = sum(end - start for start, end in merged)
                coverage_pct = (actual_busy_time / total_trace_time_us * 100) if total_trace_time_us > 0 else 0
                idle_time_us = total_trace_time_us - actual_busy_time
                avg_concurrent = total_kernel_time_us / actual_busy_time if actual_busy_time > 0 else 1.0
            else:
                coverage_pct = 0
                actual_busy_time = 0
                idle_time_us = total_trace_time_us
                avg_concurrent = 1.0

            return {
                'timeline_coverage_pct': coverage_pct,
                'kernel_time_us': actual_busy_time,
                'idle_time_us': idle_time_us,
                'avg_concurrent_kernels': avg_concurrent,
                'num_devices_detected': num_devices,
                'calculation_method': 'single_device_overlap_aware',
                '_note': 'Timeline coverage, NOT hardware utilization',
            }
        else:
            # Multi-device fallback (if multi-device analysis failed)
            # Do global overlap removal and divide by num_devices
            timestamps = [(e.get('ts', 0), e.get('dur', 0)) for e in self.kernel_events]
            timestamps = [(ts, dur) for ts, dur in timestamps if ts > 0]

            if timestamps:
                # Merge overlapping intervals globally
                intervals = [(ts, ts + dur) for ts, dur in timestamps]
                intervals.sort()

                merged = []
                for start, end in intervals:
                    if merged and start <= merged[-1][1]:
                        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                    else:
                        merged.append((start, end))

                # Global busy time (across all devices)
                # This represents the timeline span where at least one device was busy
                global_busy_time = sum(end - start for start, end in merged)

                # Timeline coverage: what % of time had any device busy?
                coverage_pct = (global_busy_time / total_trace_time_us * 100) if total_trace_time_us > 0 else 0
                idle_time_us = total_trace_time_us - global_busy_time

                # Kernel concurrency: total work / busy time
                # This captures both intra-device and inter-device parallelism
                avg_concurrent = total_kernel_time_us / global_busy_time if global_busy_time > 0 else 1.0

                # For reporting: use global_busy_time as kernel_time_us
                # (This is the actual time with kernels executing, with overlaps removed)
                kernel_time_us = global_busy_time
            else:
                kernel_time_us = 0
                coverage_pct = 0
                idle_time_us = total_trace_time_us
                avg_concurrent = 1.0

            return {
                'timeline_coverage_pct': coverage_pct,
                'kernel_time_us': kernel_time_us,
                'idle_time_us': idle_time_us,
                'avg_concurrent_kernels': avg_concurrent,
                'num_devices_detected': num_devices,
                'total_kernel_time_all_devices_us': total_kernel_time_us,
                'calculation_method': 'multi_device_global_overlap_removal',
                '_note': 'Timeline coverage, NOT hardware utilization. Used global overlap removal.',
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

        # Calculate per-device metrics
        num_devices = len([d for d in self.devices.values() if 'TPU' in d or 'GPU' in d])
        if num_devices == 0:
            num_devices = 1

        tokens_per_sec_per_device = tokens_per_sec / num_devices if num_devices > 0 else 0
        samples_per_sec_per_device = samples_per_sec / num_devices if num_devices > 0 else 0

        return {
            'tokens_per_sec': tokens_per_sec,
            'samples_per_sec': samples_per_sec,
            'tokens_per_batch': tokens_per_batch,
            'ms_per_step': avg_step_ms,
            'steps_per_sec': 1000 / avg_step_ms if avg_step_ms > 0 else 0,
            'tokens_per_sec_per_device': tokens_per_sec_per_device,
            'samples_per_sec_per_device': samples_per_sec_per_device,
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

        # Group kernels by device pid
        kernels_by_pid = defaultdict(list)
        for event in self.kernel_events:
            pid = event.get('pid')
            if pid is not None:
                kernels_by_pid[pid].append(event)

        for pid, device_name in compute_devices.items():
            # Get all kernel events for this device
            device_kernels = kernels_by_pid.get(pid, [])

            if not device_kernels:
                # Try to match by device name if pid doesn't work
                continue

            total_kernel_time = sum(e.get('dur', 0) for e in device_kernels)

            # Find timeline span for this device
            timestamps = [(e.get('ts', 0), e.get('dur', 0)) for e in device_kernels]
            timestamps = [(ts, dur) for ts, dur in timestamps if ts > 0]

            if timestamps:
                # FIXED: Use max end time instead of last event by timestamp
                first_ts = min(ts for ts, dur in timestamps)
                last_end_ts = max(ts + dur for ts, dur in timestamps)
                total_time = last_end_ts - first_ts

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
                timeline_coverage_pct = (actual_busy_time / total_time * 100) if total_time > 0 else 0

                # Kernel concurrency: avg number of kernels running simultaneously
                # If total_kernel_time > actual_busy_time, kernels overlap
                avg_concurrent_kernels = total_kernel_time / actual_busy_time if actual_busy_time > 0 else 1

                device_stats[device_name] = {
                    'kernel_count': len(device_kernels),
                    'total_kernel_time_us': total_kernel_time,
                    'actual_busy_time_us': actual_busy_time,
                    'total_time_us': total_time,
                    'timeline_coverage_pct': timeline_coverage_pct,
                    'avg_concurrent_kernels': avg_concurrent_kernels,
                }

        # Calculate load imbalance
        if len(device_stats) > 1:
            coverages = [s['timeline_coverage_pct'] for s in device_stats.values()]
            kernel_counts = [s['kernel_count'] for s in device_stats.values()]

            load_imbalance = {
                'coverage_stddev': statistics.stdev(coverages) if len(coverages) > 1 else 0,
                'coverage_range': max(coverages) - min(coverages),
                'kernel_count_stddev': statistics.stdev(kernel_counts) if len(kernel_counts) > 1 else 0,
            }
        else:
            load_imbalance = {}

        # If we didn't populate any device_stats (pid matching failed), return partial info
        if not device_stats:
            print(f"   ⚠️  Warning: Could not match kernels to devices by PID")
            print(f"   Devices: {list(compute_devices.values())}")
            if kernels_by_pid:
                print(f"   Kernel PIDs: {list(kernels_by_pid.keys())}")
            return {
                'num_devices': len(compute_devices),
                'device_stats': {},
                'load_imbalance': {},
                '_warning': 'PID matching failed - using fallback calculation'
            }

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
            if step_source == 'trace-detected-complete-only':
                total_steps = step_stats.get('total_steps_in_trace', step_stats['num_steps'])
                incomplete = step_stats.get('incomplete_steps', 0)
                print(f"   ✅ Step count from trace: {step_stats['num_steps']} complete steps ({total_steps} total, {incomplete} incomplete)")
            elif step_source == 'trace-detected':
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
                num_devices = len([d for d in self.devices.values() if 'TPU' in d or 'GPU' in d]) or 1

                print("\n   📈 Throughput:")
                print(f"      • {throughput['tokens_per_sec']:.1f} tokens/sec (system)")
                print(f"      • {throughput['tokens_per_sec_per_device']:.1f} tokens/sec/device (effective)")
                print(f"      • {throughput['samples_per_sec']:.2f} samples/sec (system)")
                print(f"      • {throughput['samples_per_sec_per_device']:.2f} samples/sec/device (effective)")
                print(f"      • {throughput['steps_per_sec']:.2f} steps/sec")

                if num_devices > 1:
                    print(f"\n      Note: Per-device metrics assume {num_devices} devices working together")
                    print(f"            For model-sharded setups, this is effective throughput per device")

        # Multi-device analysis
        device_stats = self.analyze_multi_device()
        if device_stats:
            print(f"\n🖥️  MULTI-DEVICE ANALYSIS ({device_stats['num_devices']} devices):")
            for device_name, stats in device_stats['device_stats'].items():
                print(f"\n   {device_name}:")
                print(f"      Timeline coverage: {stats['timeline_coverage_pct']:.1f}%")
                print(f"      Kernel count: {stats['kernel_count']}")
                print(f"      Actual busy time: {stats['actual_busy_time_us']/1000:.2f} ms")
                print(f"      Total kernel time: {stats['total_kernel_time_us']/1000:.2f} ms")
                print(f"      Avg concurrent kernels: {stats['avg_concurrent_kernels']:.2f}x")

            if device_stats.get('load_imbalance'):
                imbalance = device_stats['load_imbalance']
                print(f"\n   Load Balance Metrics:")
                print(f"      Coverage stddev: {imbalance['coverage_stddev']:.2f}%")
                print(f"      Coverage range: {imbalance['coverage_range']:.2f}%")
                print(f"      Kernel count stddev: {imbalance['kernel_count_stddev']:.1f}")

        # Timeline Coverage (NOT hardware utilization!)
        timeline_metrics = {}
        if step_stats:
            timeline_metrics = self.estimate_timeline_coverage(step_stats, device_stats)
            if timeline_metrics:
                num_steps = step_stats.get('num_steps', 1)
                print("\n🎯 TIMELINE COVERAGE:")
                print(f"   ⚠️  NOTE: This is timeline coverage, NOT hardware utilization!")
                print(f"   Timeline coverage: {timeline_metrics['timeline_coverage_pct']:.1f}%")
                print(f"   Avg concurrent kernels: {timeline_metrics.get('avg_concurrent_kernels', 1.0):.2f}x")
                print(f"   Kernel time: {timeline_metrics['kernel_time_us']/(1000*num_steps):.2f} ms/step")
                print(f"   Idle time: {timeline_metrics['idle_time_us']/(1000*num_steps):.2f} ms/step")
                if 'calculation_method' in timeline_metrics:
                    print(f"   Method: {timeline_metrics['calculation_method']}")

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
            if imbalance.get('coverage_stddev', 0) > 5:
                print(f"   ⚠️  Load imbalance detected (coverage stddev {imbalance['coverage_stddev']:.1f}%):")
                print("      - Check if data is evenly distributed across devices")
                print("      - Verify model parallelism configuration")
                has_recommendations = True

        if gap_stats and gap_stats.get('large_gaps', 0) > 10:
            print(f"   ⚠️  Found {gap_stats['large_gaps']} large gaps (≥1ms):")
            print("      - May indicate kernel launch overhead or synchronization bottlenecks")
            print("      - Consider using async execution or better pipelining")
            has_recommendations = True

        if timeline_metrics and timeline_metrics.get('timeline_coverage_pct', 100) < 50:
            print(f"   • Low timeline coverage ({timeline_metrics['timeline_coverage_pct']:.1f}%) - consider:")
            print(f"     - Increasing batch size")
            print(f"     - Reducing kernel launch overhead")
            print(f"     - Checking for data loading bottlenecks")
            print(f"     - Note: This measures idle time, not hardware utilization")
            has_recommendations = True

        if perf_stats and perf_stats.get('high_variance_kernels'):
            print(f"   ℹ️  Found {len(perf_stats['high_variance_kernels'])} kernels with high variance")
            print("      - May indicate dynamic workloads or non-deterministic behavior")
            has_recommendations = True

        if not has_recommendations:
            print("   ✅ No major issues detected!")

        print("="*120)

    def generate_unified_result(self) -> Optional[Dict[str, Any]]:
        """Generate unified profiling result for JSON export."""
        if not HAS_SCHEMA:
            print("❌ Cannot generate JSON: profiling_schema not available")
            return None

        # Collect all metrics
        step_stats = self.analyze_step_times()
        throughput = self.calculate_throughput_metrics(step_stats) if step_stats else {}
        multi_device = self.analyze_multi_device()
        timeline_metrics = self.estimate_timeline_coverage(step_stats, multi_device) if step_stats else {}
        top_kernels = self.analyze_top_kernels(top_n=20)
        kernels_by_op = self.analyze_kernels_by_operation()
        mem_ops = self.analyze_memory_ops()
        timeline_gaps = self.analyze_timeline_gaps()
        compilation = self.analyze_compilation()
        xla_ops = self.analyze_xla_ops()
        io_ops = self.analyze_io_operations()
        perf_stats = self.analyze_statistical_performance()

        # Detect device type from trace
        device_type = "Unknown"
        for device_name in self.devices.values():
            if 'TPU' in device_name:
                # Extract TPU version if possible
                if '/device:TPU:' in device_name:
                    device_type = "TPU (multi-core)"
                else:
                    device_type = "TPU"
                break
            elif 'GPU' in device_name:
                device_type = "GPU"
                break

        # Determine number of TPU/GPU devices
        num_devices = len([d for d in self.devices.values() if 'TPU' in d or 'GPU' in d])

        # Prepare memory operations in unified format
        memory_unified = None
        if mem_ops:
            transfers = mem_ops.get('transfers', {})
            memory_unified = {
                "h2d_count": transfers.get('h2d', {}).get('count', 0),
                "h2d_total_time_us": transfers.get('h2d', {}).get('total_us', 0.0),
                "d2h_count": transfers.get('d2h', {}).get('count', 0),
                "d2h_total_time_us": transfers.get('d2h', {}).get('total_us', 0.0),
                "d2d_count": transfers.get('d2d', {}).get('count', 0),
                "d2d_total_time_us": transfers.get('d2d', {}).get('total_us', 0.0),
                "total_memory_ops": mem_ops.get('num_ops', 0),
            }

        # Prepare multi-device stats in unified format
        multi_device_unified = None
        if multi_device and multi_device.get('num_devices', 0) > 1:
            device_stats_list = []
            for idx, (device_name, stats) in enumerate(multi_device.get('device_stats', {}).items()):
                device_stats_list.append({
                    "device_name": device_name,
                    "device_id": idx,
                    "kernel_count": stats['kernel_count'],
                    "total_kernel_time_us": stats['total_kernel_time_us'],
                    "actual_busy_time_us": stats['actual_busy_time_us'],
                    "total_time_us": stats['total_time_us'],
                    "timeline_coverage_percent": stats['timeline_coverage_pct'],
                    "avg_concurrent_kernels": stats['avg_concurrent_kernels'],
                })

            multi_device_unified = {
                "num_devices": multi_device['num_devices'],
                "device_stats": device_stats_list,
                "coverage_stddev": multi_device.get('load_imbalance', {}).get('coverage_stddev'),
                "coverage_range": multi_device.get('load_imbalance', {}).get('coverage_range'),
                "kernel_count_stddev": multi_device.get('load_imbalance', {}).get('kernel_count_stddev'),
                "_note": "Timeline coverage per device, NOT hardware utilization",
            }

            # Calculate average coverage
            if device_stats_list:
                avg_coverage = sum(d['timeline_coverage_percent'] for d in device_stats_list) / len(device_stats_list)
                multi_device_unified["avg_timeline_coverage_percent"] = avg_coverage

        # Prepare timeline gaps in unified format
        timeline_unified = None
        if timeline_gaps:
            timeline_unified = {
                "total_gaps": timeline_gaps.get('total_gaps', 0),
                "total_gap_time_us": timeline_gaps.get('total_gap_time_us', 0.0),
                "avg_gap_us": timeline_gaps.get('avg_gap_us', 0.0),
                "max_gap_us": timeline_gaps.get('max_gap_us', 0.0),
                "gap_time_percent": (timeline_gaps.get('total_gap_time_us', 0.0) / step_stats.get('total_us', 1) * 100) if step_stats else 0.0,
                "gaps_under_10us": timeline_gaps.get('small_gaps', 0),
                "gaps_10_100us": timeline_gaps.get('medium_gaps', 0),
                "gaps_100_1000us": 0,  # Not tracked separately
                "gaps_over_1000us": timeline_gaps.get('large_gaps', 0),
            }

        # Prepare compilation stats
        compilation_unified = None
        if compilation:
            compilation_unified = {
                "compilation_time_us": compilation.get('compilation_time_us', 0.0),
                "num_xla_modules": compilation.get('num_xla_modules', 0),
                "num_compilation_events": compilation.get('num_compilation_events', 0),
                "compilation_percent": (compilation.get('compilation_time_us', 0.0) / step_stats.get('total_us', 1) * 100) if step_stats else 0.0,
            }

        # Prepare I/O operations
        io_unified = None
        if io_ops:
            io_unified = {
                "io_event_count": io_ops.get('num_io_ops', 0),
                "io_total_time_us": io_ops.get('total_io_time_us', 0.0),
                "io_percent": (io_ops.get('total_io_time_us', 0.0) / step_stats.get('total_us', 1) * 100) if step_stats else 0.0,
            }

        # Prepare statistical performance
        stats_unified = None
        if perf_stats:
            overall = perf_stats.get('overall_kernel_stats', {})
            stats_unified = {
                "step_time_variance": overall.get('stddev_us', 0.0) ** 2 if overall.get('stddev_us') else None,
                "kernel_time_variance": overall.get('stddev_us', 0.0) ** 2 if overall.get('stddev_us') else None,
                "num_outlier_steps": None,  # Not tracked
            }

        # Create unified result
        result = create_unified_result(
            # Metadata
            platform="TPU" if device_type.startswith("TPU") else "GPU",
            device_type=device_type,
            trace_file=str(self.trace_file),
            framework="Flax",
            num_devices=num_devices,

            # Model Config
            model_dims=self.config.model_dims,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            batch_size=self.config.batch_size,
            seq_length=self.config.seq_length,
            vocab_size=self.config.vocab_size,
            ffn_hidden_dims=self.config.ffn_hidden_dims,
            num_items=self.config.num_items,
            num_clusters=self.config.num_clusters,
            item_embedding_dim=self.config.item_embedding_dim,
            pretrained_model=self.config.pretrained_model,

            # Parameters (Flax doesn't track frozen params separately)
            total_params=self.config.calculate_params() if self.config.is_valid() else 0,

            # Step Timing
            num_steps=step_stats.get('num_steps', 0) if step_stats else 0,
            total_time_us=step_stats.get('total_us', 0.0) if step_stats else 0.0,
            avg_step_time_us=step_stats.get('avg_us', 0.0) if step_stats else 0.0,
            total_kernels=step_stats.get('total_kernels', 0) if step_stats else 0,
            kernels_per_step=step_stats.get('kernels_per_step', 0.0) if step_stats else 0.0,
            step_count_source=step_stats.get('step_source', 'heuristic') if step_stats else 'unknown',

            # Throughput
            tokens_per_sec=throughput.get('tokens_per_sec', 0.0),
            samples_per_sec=throughput.get('samples_per_sec', 0.0),
            steps_per_sec=throughput.get('steps_per_sec', 0.0),
            ms_per_step=throughput.get('ms_per_step', 0.0),
            tokens_per_batch=throughput.get('tokens_per_batch', 0),
            tokens_per_sec_per_device=throughput.get('tokens_per_sec_per_device'),
            samples_per_sec_per_device=throughput.get('samples_per_sec_per_device'),

            # Timeline Coverage (per-device average for multi-device)
            timeline_coverage_percent=timeline_metrics.get('timeline_coverage_pct', 0.0),
            kernel_time_us=timeline_metrics.get('kernel_time_us', 0.0),
            idle_time_us=timeline_metrics.get('idle_time_us', 0.0),
            avg_concurrent_kernels=timeline_metrics.get('avg_concurrent_kernels', 1.0),

            # Kernels
            top_kernels=top_kernels,
            kernels_by_category=kernels_by_op,

            # Optional sections
            memory_ops=memory_unified,
            multi_device=multi_device_unified,
            timeline_gaps=timeline_unified,
            compilation=compilation_unified,
            io_operations=io_unified,
            statistical_performance=stats_unified,
            xla_operations=xla_ops,
        )

        return result


def main():
    parser = argparse.ArgumentParser(
        description='JAX profiler trace analyzer for efficientids_flax',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis (auto-detects steps from trace)
  python analyze_profile.py --trace_file checkpoints/tpu_gemma/profiler_traces/plugins/profile/*/trace.json.gz

  # Specify checkpoint directory
  python analyze_profile.py --trace_file trace.json.gz --checkpoint_dir checkpoints/tpu_gemma

  # Export to JSON for GPU/TPU comparison
  python analyze_profile.py --trace_file trace.json.gz --output_json tpu_profile.json
        """
    )
    parser.add_argument('--trace_file', type=str, required=True, help='Path to trace.json.gz')
    parser.add_argument('--checkpoint_dir', type=str, help='Checkpoint directory (auto-detected if not provided)')
    parser.add_argument('--output_json', type=str, help='Save results to JSON file (unified format for comparison)')

    args = parser.parse_args()

    if not Path(args.trace_file).exists():
        print(f"❌ Error: Trace file not found: {args.trace_file}")
        return 1

    analyzer = ProfileAnalyzer(args.trace_file, args.checkpoint_dir, num_steps=None)

    # Always print console summary
    analyzer.print_summary()

    # Optionally export to JSON
    if args.output_json:
        if not HAS_SCHEMA:
            print(f"\n❌ Error: Cannot export JSON - profiling_schema.py not found")
            print("   Make sure profiling_schema.py is in the same directory")
            return 1

        print(f"\n📊 Generating unified JSON output...")
        result = analyzer.generate_unified_result()
        if result:
            save_profiling_json(result, args.output_json)
            print(f"\n💡 You can now compare this profile with GPU results using:")
            print(f"   python compare_profiles.py --tpu {args.output_json} --gpu <gpu_profile.json>")
        else:
            print(f"❌ Failed to generate JSON output")
            return 1

    return 0


if __name__ == '__main__':
    exit(main())
