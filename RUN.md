# EfficientIDS Flax - Quick Start

## 1. Debug Test (2 minutes)

```bash
python train_efficientids.py --config debug
```

Verifies setup works. 200 steps, completes quickly.

---

## 2. Full Training (TPU/GPU)

```bash
python train_efficientids.py \
  --config tpu_optimized \
  --pretrained_path /home/ritik.r/2b
```

- Loads pretrained Gemma 2B
- Freezes transformer (trains only adapters)
- Auto-shards across available devices (TPU/GPU)
- 10K steps, saves checkpoints every 1000

---

## 3. With Profiling

```bash
python train_efficientids.py \
  --config tpu_optimized \
  --pretrained_path /home/ritik.r/2b \
  --enable_profiling
```

Same as #2, plus profiles steps 150-155.

**View traces:**
1. Go to https://ui.perfetto.dev/
2. Drag & drop trace file from `checkpoints/tpu_gemma/profiler_traces/`
3. Interactive timeline shows device ops, memory, communication

---

## Notes

- **Profiling works on TPU & GPU** - same command
- **Sharding is automatic** - 1 device = no shard, 4 devices = shard across 4
- **Custom args:** `--batch_size 8`, `--max_steps 20000`, `--profiler_start_step 200`
