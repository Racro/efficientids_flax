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
import jax.profiler

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
    ):
        """
        Initialize profiler.

        Args:
            num_steps: Number of consecutive steps to profile
            log_dir: Directory to save profiler traces
            auto_enable_at_step: Automatically start profiling at this step (optional)
        """
        self.num_steps = num_steps
        self.log_dir = log_dir
        self.auto_enable_at_step = auto_enable_at_step

        # State tracking
        self._enabled = False
        self._active = False
        self._start_step = None
        self._steps_captured = 0
        self._step_times = []

        # Create output directory
        os.makedirs(log_dir, exist_ok=True)

        # Register cleanup
        atexit.register(self._cleanup)

        logger.info(f"Profiler initialized: {num_steps} steps → {log_dir}")
        if auto_enable_at_step is not None:
            logger.info(f"  Auto-enable at step {auto_enable_at_step}")

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

    def end_step(self):
        """Call at the end of each training step."""
        # Track step duration
        if hasattr(self, '_step_start_time'):
            duration = time.time() - self._step_start_time
            self._step_times.append(duration)

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
        """Internal: Start JAX profiler."""
        logger.info(f"🚀 Starting profiler at step {step} for {self.num_steps} steps")
        logger.info(f"   Trace directory: {self.log_dir}")

        try:
            # Start JAX profiler (works on both TPU and GPU)
            jax.profiler.start_trace(self.log_dir)
            self._active = True
            logger.info("   ✓ JAX profiler started")
        except Exception as e:
            logger.error(f"   ✗ Failed to start profiler: {e}")
            self._active = False
            self._enabled = False

    def _stop_profiling(self):
        """Internal: Stop JAX profiler."""
        if not self._active:
            return

        logger.info(f"⏸️  Stopping profiler after {self._steps_captured} steps")

        try:
            jax.profiler.stop_trace()
            logger.info(f"   ✓ Trace saved to: {self.log_dir}/")
            logger.info(f"   📁 View at: https://ui.perfetto.dev/")
        except Exception as e:
            logger.error(f"   ✗ Error stopping profiler: {e}")
        finally:
            self._active = False
            self._enabled = False

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
