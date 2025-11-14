"""
JAX Profiler for EfficientIDS Flax - TPU & GPU Compatible

Usage in training loop:
    profiler = Profiler(num_steps=5, log_dir='./profiler_traces')

    for step in range(num_steps):
        profiler.begin_step(step)

        # Training step here
        state, metrics = train_step(state, batch)

        profiler.end_step()

        # Trigger profiling at specific step
        if step == 100:
            profiler.capture_async()

View traces:
    - TPU: Upload to https://ui.perfetto.dev/
    - GPU: Use TensorBoard or Perfetto
"""

import os
import time
import atexit
import logging
import threading
import jax.profiler
from typing import Optional

logger = logging.getLogger(__name__)


class Profiler:
    """
    JAX profiler wrapper for capturing performance traces.

    Compatible with both TPU and GPU backends.
    """

    def __init__(
        self,
        num_steps: int = 5,
        log_dir: str = './profiler_traces',
        auto_enable_at_step: int | None = None,
        flop_counter: Optional[any] = None,
    ):
        """
        Initialize profiler.

        Args:
            num_steps: Number of consecutive steps to profile
            log_dir: Directory to save profiler traces
            auto_enable_at_step: Automatically start profiling at this step (optional)
            flop_counter: Optional FLOPCounter instance for MFU tracking
        """
        self.num_steps = num_steps
        self.log_dir = log_dir
        self.auto_enable_at_step = auto_enable_at_step
        self.flop_counter = flop_counter

        # State tracking
        self._enabled = False
        self._active = False
        self._start_step = None
        self._steps_captured = 0
        self._step_times = []

        # FLOP tracking
        self._tokens_processed = []
        self._step_start_times = []

        # Create output directory
        os.makedirs(log_dir, exist_ok=True)

        # Register cleanup
        atexit.register(self._cleanup)

        logger.info(f"Profiler initialized: {num_steps} steps → {log_dir}")
        if auto_enable_at_step is not None:
            logger.info(f"  Auto-enable at step {auto_enable_at_step}")
        if flop_counter is not None:
            logger.info(f"  FLOP tracking: ENABLED (MFU calculation available)")

    def begin_step(self, step: int):
        """
        Call at the beginning of each training step.

        Args:
            step: Current training step number
        """
        # Auto-enable if configured
        if self.auto_enable_at_step is not None and step == self.auto_enable_at_step:
            self.capture_async()

        # Check if we should profile this step
        if self._enabled:
            if self._start_step is None:
                self._start_step = step

            # Start profiling if within capture window
            if step >= self._start_step and step < self._start_step + self.num_steps:
                if not self._active:
                    self._start_profiling(step)

        # Record step start time for performance tracking
        self._step_start_time = time.time()

    def end_step(self, num_tokens: Optional[int] = None):
        """
        Call at the end of each training step.

        Args:
            num_tokens: Number of tokens processed in this step (for FLOP tracking)
        """
        # Track step duration
        if hasattr(self, '_step_start_time'):
            duration = time.time() - self._step_start_time
            self._step_times.append(duration)

            # Track tokens for FLOP calculation
            if num_tokens is not None and self._active:
                self._tokens_processed.append(num_tokens)
                self._step_start_times.append(self._step_start_time)

        # Increment counter if profiling
        if self._active:
            self._steps_captured += 1

            # Stop if we've captured enough steps
            if self._steps_captured >= self.num_steps:
                self._stop_profiling()

    def capture_async(self):
        """
        Enable profiling for the next N steps.

        Call this at any training step to trigger profiling.
        """
        if self._active:
            logger.warning("Profiler already active, ignoring capture_async()")
            return

        self._enabled = True
        self._start_step = None
        self._steps_captured = 0
        logger.info(f"📊 Profiler enabled for next {self.num_steps} steps")

    def _start_profiling(self, step: int):
        """Internal: Start JAX profiler (PAXml-style)."""
        logger.info(f"🚀 Starting profiler at step {step} for {self.num_steps} steps")
        logger.info(f"   Trace directory: {self.log_dir}")

        try:
            jax.profiler.start_trace(self.log_dir)
            self._active = True
            logger.info("   ✓ JAX profiler started")
        except Exception as e:
            logger.error(f"   ✗ Failed to start profiler: {e}")
            self._active = False
            self._enabled = False

    def _stop_profiling(self):
        """Internal: Stop JAX profiler (PAXml-style with exception handling)."""
        if not self._active:
            return

        logger.info(f"⏸️  Stopping profiler after {self._steps_captured} steps")

        try:
            jax.profiler.stop_trace()
            logger.info(f"   ✓ Trace saved to: {self.log_dir}/")
            logger.info(f"   📁 View at: https://ui.perfetto.dev/")

            # Export FLOP statistics if available
            if self.flop_counter is not None and self._tokens_processed:
                self._export_flop_stats()

        except Exception as e:
            logger.error(f"   ✗ Error stopping profiler: {e}")
            logger.info(f"   ⚠️  Traces may still be saved despite error")
        finally:
            self._active = False
            self._enabled = False

    def _export_flop_stats(self):
        """Export FLOP statistics to JSON file."""
        if not self._tokens_processed or self.flop_counter is None:
            return

        try:
            import json

            # Calculate average throughput
            total_tokens = sum(self._tokens_processed)
            total_time = sum(self._step_times[-len(self._tokens_processed):])
            tokens_per_sec = total_tokens / total_time if total_time > 0 else 0

            # Calculate MFU
            mfu = self.flop_counter.compute_mfu(tokens_per_sec)
            achieved_tflops = self.flop_counter.compute_achieved_flops(tokens_per_sec)

            # Create stats dict
            stats = {
                'profiling_steps': self._steps_captured,
                'total_tokens': total_tokens,
                'total_time_sec': total_time,
                'tokens_per_sec': tokens_per_sec,
                'model_params': self.flop_counter.num_params,
                'flops_per_token': self.flop_counter.estimate_flops_per_token(),
                'achieved_tflops': achieved_tflops,
                'peak_tflops': self.flop_counter.peak_flops,
                'mfu_percent': mfu,
                'per_step_tokens': self._tokens_processed,
                'per_step_times': self._step_times[-len(self._tokens_processed):],
            }

            # Save to JSON
            output_file = os.path.join(self.log_dir, 'flop_stats.json')
            with open(output_file, 'w') as f:
                json.dump(stats, f, indent=2)

            logger.info(f"   📊 FLOP Statistics:")
            logger.info(f"      Throughput: {tokens_per_sec:.2f} tokens/sec")
            logger.info(f"      Achieved: {achieved_tflops:.2f} TFLOP/s")
            logger.info(f"      Peak: {self.flop_counter.peak_flops:.2f} TFLOP/s")
            logger.info(f"      MFU: {mfu:.2f}%")
            logger.info(f"   💾 Saved to: {output_file}")

        except Exception as e:
            logger.error(f"   ✗ Failed to export FLOP stats: {e}")

    def _cleanup(self):
        """Cleanup on program exit."""
        if self._active:
            logger.warning("⚠️  Profiler still active on exit, stopping...")
            self._stop_profiling()

    @property
    def avg_step_time(self) -> float:
        """Average step duration in seconds."""
        if not self._step_times:
            return 0.0
        return sum(self._step_times) / len(self._step_times)

    @property
    def is_profiling(self) -> bool:
        """True if currently capturing a trace."""
        return self._active


# Convenience context manager for one-time profiling
class ProfileContext:
    """
    Context manager for profiling a code block.

    Usage:
        with ProfileContext('./traces'):
            # Code to profile
            train_step(state, batch)
    """

    def __init__(self, log_dir: str = './profiler_traces'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def __enter__(self):
        logger.info(f"Starting profiler: {self.log_dir}")
        jax.profiler.start_trace(self.log_dir)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        jax.profiler.stop_trace()
        logger.info(f"Trace saved: {self.log_dir}/")
        logger.info(f"View at: https://ui.perfetto.dev/")
        return False
