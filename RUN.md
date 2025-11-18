# EfficientIDS Flax - Quick Start

## 1. Debug Test (2 minutes)

```bash
python train_efficientids.py --config debug
```

Verifies setup works. 200 steps, completes quickly.

---

## 2. Full Training (TPU/GPU)

### Gemma 2B (Frozen)
```bash
python train_efficientids.py --config gemma_2b
```

### Gemma 7B (Frozen)
```bash
python train_efficientids.py --config gemma_7b
```

- Auto-detects checkpoint path (2B: `/home/ritik.r/2b`, 7B: `/home/ritik.r/7b`)
- Loads pretrained weights and freezes transformer
- Trains only adapters + item embeddings
- Auto-shards across available devices (TPU/GPU)
- 10K steps, saves checkpoints every 1000

---

## 3. With Profiling

```bash
# Gemma 2B with profiling
python train_efficientids.py --config gemma_2b --enable_profiling

# Gemma 7B with profiling
python train_efficientids.py --config gemma_7b --enable_profiling
```

Same as #2, plus profiles steps 150-155 (after warmup).

**View traces:**
1. Go to https://ui.perfetto.dev/
2. Drag & drop trace file from `checkpoints/gemma_2b_frozen/profiler_traces/` (or `gemma_7b_frozen/`)
3. Interactive timeline shows device ops, memory, communication

---

## Notes

- **Profiling works on TPU & GPU** - same command
- **Sharding is automatic** - 1 device = no shard, 4 devices = shard across 4
- **Custom args:** `--batch_size 8`, `--max_steps 20000`, `--profiler_start_step 200`
