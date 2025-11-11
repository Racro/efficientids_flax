# JAX Profiling for EfficientIDS Flax

Cross-platform profiling for TPU and GPU training.

## Quick Start

### Auto-profile (default: starts at step 150):

```bash
python train_efficientids.py \
  --config tpu_optimized \
  --enable_profiling
```

This will automatically profile steps 150-155 (after warmup/compilation).

### Custom profiling range:

```bash
python train_efficientids.py \
  --config tpu_optimized \
  --enable_profiling \
  --profiler_start_step 100 \
  --profiler_num_steps 10
```

This will profile steps 100-110.

### Manual trigger (programmatic):

```bash
# Enable profiling but don't auto-start
python train_efficientids.py \
  --config tpu_optimized \
  --enable_profiling
```

Then trigger manually by calling `trainer.profiler.capture_async()` at any step.

## Command-Line Arguments

- `--enable_profiling`: Enable profiler (default: False)
- `--profiler_start_step N`: Auto-start at step N (default: 150 - after warmup)
- `--profiler_num_steps N`: Profile N consecutive steps (default: 5)

**Why step 150?** First ~50 steps involve JIT compilation and warmup. Step 150 captures steady-state performance.

## Viewing Traces

### TPU

1. Traces saved to: `<checkpoint_dir>/profiler_traces/`
2. Upload to: https://ui.perfetto.dev/
3. Drag and drop the trace file

### GPU

**Option 1: Perfetto (recommended)**
- Same as TPU - upload to https://ui.perfetto.dev/

**Option 2: TensorBoard**
```bash
tensorboard --logdir=<checkpoint_dir>/profiler_traces/
```

## Programmatic Usage

```python
from train.profiler import Profiler

# In training loop
profiler = Profiler(num_steps=5, log_dir='./traces')

for step in range(num_steps):
    profiler.begin_step(step)

    # Your training code
    state, metrics = train_step(state, batch)

    profiler.end_step()

    # Trigger profiling at step 100
    if step == 100:
        profiler.capture_async()
```

## One-off Profiling

For profiling a specific code block:

```python
from train.profiler import ProfileContext

with ProfileContext('./traces'):
    # Code to profile
    state, metrics = train_step(state, batch)
```

## What Gets Profiled

JAX profiler captures:
- **Device operations**: Matrix multiplies, convolutions, attention
- **Memory transfers**: Host → Device, Device → Host
- **Compilation time**: XLA compilation (first step)
- **Communication**: AllReduce for gradient sync (multi-device)
- **Host overhead**: Python, data loading

## Performance Tips

1. **Ignore first few steps**: Compilation happens on first run
2. **Profile steady state**: Start at step 100+ after warmup
3. **Profile 5-10 steps**: Enough to see patterns
4. **Check for bottlenecks**:
   - Long gaps = data loading issues
   - Large AllReduce = gradient sync overhead
   - High compilation time = XLA inefficiency

## Example Output

```
📊 Profiler enabled for next 5 steps
🚀 Starting profiler at step 100 for 5 steps
   Trace directory: ./checkpoints/profiler_traces
   ✓ JAX profiler started
⏸️  Stopping profiler after 5 steps
   ✓ Trace saved to: ./checkpoints/profiler_traces/
   📁 View at: https://ui.perfetto.dev/
```

## Troubleshooting

**"Profiler already active"**
- Wait for current profiling session to finish
- Each session must complete before starting a new one

**Empty trace file**
- Ensure JAX is installed correctly: `pip install jax[tpu]` or `jax[cuda]`
- Check permissions on output directory

**Slow profiling**
- Profiling adds ~5-10% overhead
- Disable after collecting traces

## Cross-Platform Notes

- **TPU**: Traces show TPU core utilization, HBM bandwidth
- **GPU**: Traces show CUDA kernels, VRAM usage
- **CPU**: Traces show thread activity (limited detail)

Same code works on all platforms!
