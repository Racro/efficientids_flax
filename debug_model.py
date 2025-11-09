"""
Debug script to check model forward pass and identify NaN source.
"""

import jax
import jax.numpy as jnp
import numpy as np
from data.dataset import MovieLensDataset
from core.models import SimpleEfficientIDSModel

print("=" * 80)
print("Model Forward Pass Check")
print("=" * 80)

# Load dataset
dataset = MovieLensDataset(
    data_dir="data/ml1m_processed/processed",
    split="train",
    max_seq_len=32,
    batch_size=2,
    shuffle=False,
    max_sequences=10,
)

# Get clustering info
clustering_info = dataset.get_clustering_info()

# Create model
model = SimpleEfficientIDSModel(
    num_items=dataset.num_items,
    num_clusters=dataset.num_clusters,
    item_embedding_dim=64,
    model_dims=128,
    clustering_info=clustering_info,
)

# Get a batch
batch = next(iter(dataset))

print(f"\n1. Initializing model...")
key = jax.random.PRNGKey(42)
variables = model.init(key, **batch, training=True)
params = variables['params']

print(f"   ✓ Model initialized")
print(f"   Num parameters: {sum(x.size for x in jax.tree_util.tree_leaves(params)):,}")

print(f"\n2. Checking parameter initialization:")
for layer_name, layer_params in params.items():
    print(f"   {layer_name}:")
    if isinstance(layer_params, dict):
        for param_name, param_value in layer_params.items():
            has_nan = jnp.isnan(param_value).any()
            has_inf = jnp.isinf(param_value).any()
            print(f"      {param_name}: shape={param_value.shape}, "
                  f"mean={jnp.mean(param_value):.4f}, "
                  f"std={jnp.std(param_value):.4f}, "
                  f"NaN={has_nan}, Inf={has_inf}")
    else:
        has_nan = jnp.isnan(layer_params).any()
        has_inf = jnp.isinf(layer_params).any()
        print(f"      value: shape={layer_params.shape}, "
              f"mean={jnp.mean(layer_params):.4f}, "
              f"std={jnp.std(layer_params):.4f}, "
              f"NaN={has_nan}, Inf={has_inf}")

print(f"\n3. Running forward pass...")
outputs = model.apply(variables, **batch, training=True)

print(f"   ✓ Forward pass complete")
print(f"   Output keys: {outputs.keys()}")

print(f"\n4. Checking outputs:")
for key, value in outputs.items():
    if isinstance(value, jnp.ndarray):
        has_nan = jnp.isnan(value).any()
        has_inf = jnp.isinf(value).any()
        print(f"   {key}: shape={value.shape}, "
              f"mean={jnp.mean(value):.4f}, "
              f"std={jnp.std(value):.4f}, "
              f"min={jnp.min(value):.4f}, "
              f"max={jnp.max(value):.4f}, "
              f"NaN={has_nan}, Inf={has_inf}")
    else:
        print(f"   {key}: {value}")

print(f"\n5. Computing loss manually:")
if 'total_loss' in outputs:
    loss = outputs['total_loss']
    print(f"   Loss from model: {loss}")
    print(f"   Loss is NaN: {jnp.isnan(loss)}")
    print(f"   Loss is Inf: {jnp.isinf(loss)}")

    # Check intermediate losses
    if 'cluster_loss' in outputs:
        print(f"   Cluster loss: {outputs['cluster_loss']}")
    if 'item_loss' in outputs:
        print(f"   Item loss: {outputs['item_loss']}")

print(f"\n6. Checking embeddings:")
item_embeddings = params['item_embedding_table']
print(f"   Item embeddings shape: {item_embeddings.shape}")
print(f"   Item embeddings mean: {jnp.mean(item_embeddings):.4f}")
print(f"   Item embeddings std: {jnp.std(item_embeddings):.4f}")
print(f"   Item embeddings NaN: {jnp.isnan(item_embeddings).any()}")

print("\n" + "=" * 80)
if 'total_loss' in outputs and jnp.isnan(outputs['total_loss']):
    print("❌ Loss is NaN! Need to debug hierarchical softmax.")
else:
    print("✅ Forward pass OK!")
