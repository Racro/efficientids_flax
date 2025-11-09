# EfficientIDS - Pure JAX/Flax Implementation

Minimal, production-ready implementation of EfficientIDS using pure JAX/Flax.

**Why this port?**
- Compatible with newer JAX versions (0.4.30+)
- Supports latest GPU generations (H100, H200, etc.)
- No PAXml dependency issues
- Simpler, more maintainable codebase

## Architecture

```
efficientids_flax/
├── core/
│   ├── models.py          # Flax model modules (transformer, softmax)
│   ├── hierarchical.py    # Hierarchical softmax implementation
│   └── embeddings.py      # Item embeddings and adapters
├── train/
│   ├── trainer.py         # Pure JAX training loop
│   ├── optimizer.py       # Optax optimizers and schedules
│   └── checkpointing.py   # Checkpoint save/load
├── data/
│   ├── pipeline.py        # Data loading (tf.data or numpy)
│   └── preprocessing.py   # Feature conversion
├── configs/
│   └── base_config.py     # Training configurations
└── inference.py           # Inference utilities
```

## Key Differences from Original

| Component | Original (PAXml) | Flax Port |
|-----------|------------------|-----------|
| Framework | PAXml/Praxis | Pure JAX/Flax |
| Training Loop | PAXml programs | Custom JAX loop |
| Optimizers | Praxis optimizers | Optax |
| Checkpoints | Orbax (PAXml) | Orbax (direct) |
| Data | SeqIO | tf.data / numpy |
| Sharding | PAXml mesh | jax.sharding native |

## Installation

```bash
# Newer JAX for latest GPUs
pip install "jax[cuda12]>=0.4.30"  # or cuda11
pip install flax optax orbax-checkpoint
pip install tensorflow tensorflow-datasets  # for data only
```

## Usage

```python
# Training
python -m efficientids_flax.train \
    --config configs/qwen_128.py \
    --data_dir ./data/ml1m_processed

# Inference
python -m efficientids_flax.inference \
    --checkpoint ./checkpoints/step_10000 \
    --input_items "1,2,3,4,5"
```

## Compatibility

- **JAX**: 0.4.30+ (latest)
- **Flax**: 0.8.0+
- **Python**: 3.10+
- **GPUs**: A100, H100, H200, etc.

## Development Status

**Phase 1: Core Models** ✓
- [x] Transformer architecture
- [x] Hierarchical softmax
- [x] Item embeddings

**Phase 2: Training** ⏳
- [ ] Training loop
- [ ] Distributed training
- [ ] Checkpointing

**Phase 3: Data** ⏳
- [ ] Data pipeline
- [ ] Preprocessing

**Phase 4: Production**
- [ ] Model loading
- [ ] Inference API
- [ ] Evaluation

## Migration from Original

To migrate existing checkpoints:
```python
from efficientids_flax.utils import convert_paxml_checkpoint
convert_paxml_checkpoint(
    paxml_path="./old_checkpoint",
    output_path="./flax_checkpoint"
)
```
