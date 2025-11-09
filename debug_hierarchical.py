"""
Debug hierarchical softmax to find NaN source.
"""

import jax
import jax.numpy as jnp
import numpy as np
from data.dataset import MovieLensDataset
from core.hierarchical_simple import SimpleHierarchicalSoftmax, JaxClusteringInfo

print("=" * 80)
print("Hierarchical Softmax Debug")
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

clustering_info = dataset.get_clustering_info()

# Convert to JAX arrays for compatibility with Flax 0.8+
jax_clustering = JaxClusteringInfo.from_numpy_clustering_info(clustering_info)

# Create hierarchical softmax
hs = SimpleHierarchicalSoftmax(
    num_items=dataset.num_items,
    num_clusters=dataset.num_clusters,
    item_embedding_dim=64,
    cluster_assignments=jax_clustering.cluster_assignments,
    cluster_indices=jax_clustering.cluster_indices,
)

# Create synthetic inputs
key = jax.random.PRNGKey(42)
batch_size, seq_len, dim = 2, 8, 64

hidden_states = jax.random.normal(key, (batch_size, seq_len, dim))
item_embeddings = jax.random.normal(key, (dataset.num_items, dim))
targets = jax.random.randint(key, (batch_size, seq_len), 0, dataset.num_items)
loss_mask = jnp.ones((batch_size, seq_len), dtype=jnp.float32)

print(f"\n1. Inputs:")
print(f"   hidden_states: {hidden_states.shape}")
print(f"   item_embeddings: {item_embeddings.shape}")
print(f"   targets: {targets.shape}")
print(f"   loss_mask: {loss_mask.shape}")

print(f"\n2. Initializing hierarchical softmax...")
variables = hs.init(
    key,
    hidden_states=hidden_states,
    item_embeddings=item_embeddings,
    targets=targets,
    loss_mask=loss_mask,
    training=True,
)

print(f"   ✓ Initialized")

print(f"\n3. Running forward pass with detailed inspection...")

# Manually run the forward pass step by step
params = variables['params']
cluster_embeddings = params['cluster_embeddings']

print(f"\n   Step 1: Cluster embeddings")
print(f"      Shape: {cluster_embeddings.shape}")
print(f"      Mean: {jnp.mean(cluster_embeddings):.4f}")
print(f"      Std: {jnp.std(cluster_embeddings):.4f}")
print(f"      NaN: {jnp.isnan(cluster_embeddings).any()}")

print(f"\n   Step 2: Cluster logits")
cluster_logits = jnp.einsum('...d,cd->...c', hidden_states, cluster_embeddings)
print(f"      Shape: {cluster_logits.shape}")
print(f"      Mean: {jnp.mean(cluster_logits):.4f}")
print(f"      Std: {jnp.std(cluster_logits):.4f}")
print(f"      Min: {jnp.min(cluster_logits):.4f}")
print(f"      Max: {jnp.max(cluster_logits):.4f}")
print(f"      NaN: {jnp.isnan(cluster_logits).any()}")

print(f"\n   Step 3: Target cluster IDs")
target_cluster_ids = jnp.take(jax_clustering.cluster_assignments, targets)
print(f"      Shape: {target_cluster_ids.shape}")
print(f"      Sample: {target_cluster_ids[0, :5]}")
print(f"      Min: {jnp.min(target_cluster_ids)}")
print(f"      Max: {jnp.max(target_cluster_ids)}")
print(f"      Valid range: {(target_cluster_ids >= 0).all() and (target_cluster_ids < dataset.num_clusters).all()}")

print(f"\n   Step 4: Cluster log probs")
cluster_log_probs = jax.nn.log_softmax(cluster_logits)
print(f"      Shape: {cluster_log_probs.shape}")
print(f"      Mean: {jnp.mean(cluster_log_probs):.4f}")
print(f"      Min: {jnp.min(cluster_log_probs):.4f}")
print(f"      Max: {jnp.max(cluster_log_probs):.4f}")
print(f"      NaN: {jnp.isnan(cluster_log_probs).any()}")
print(f"      Inf: {jnp.isinf(cluster_log_probs).any()}")

print(f"\n   Step 5: Cluster members")
cluster_members = jnp.take(jax_clustering.cluster_indices, target_cluster_ids, axis=0)
print(f"      Shape: {cluster_members.shape}")
print(f"      Sample [0,0,:10]: {cluster_members[0, 0, :10]}")
print(f"      Has -1 padding: {(cluster_members == -1).any()}")
print(f"      Valid count [0,0]: {(cluster_members[0, 0] != -1).sum()}")

print(f"\n   Step 6: Item logits within cluster")
valid_mask = cluster_members != -1
cluster_member_embeddings = jnp.take(item_embeddings, jnp.maximum(cluster_members, 0), axis=0)
item_logits = jnp.einsum('...d,...md->...m', hidden_states, cluster_member_embeddings)
masked_item_logits = jnp.where(valid_mask, item_logits, -1e9)

print(f"      Item logits shape: {item_logits.shape}")
print(f"      Item logits mean: {jnp.mean(item_logits):.4f}")
print(f"      Masked item logits min: {jnp.min(masked_item_logits):.4f}")
print(f"      Masked item logits max: {jnp.max(masked_item_logits):.4f}")
print(f"      NaN in item_logits: {jnp.isnan(item_logits).any()}")

print(f"\n   Step 7: Item log probs")
item_log_probs = jax.nn.log_softmax(masked_item_logits)
item_log_probs = jnp.where(valid_mask, item_log_probs, 0.0)
print(f"      Shape: {item_log_probs.shape}")
print(f"      Mean: {jnp.mean(item_log_probs):.4f}")
print(f"      Min: {jnp.min(item_log_probs):.4f}")
print(f"      Max: {jnp.max(item_log_probs):.4f}")
print(f"      NaN: {jnp.isnan(item_log_probs).any()}")
print(f"      Inf: {jnp.isinf(item_log_probs).any()}")

print(f"\n   Step 8: Find target in cluster")
target_match = jnp.equal(cluster_members, jnp.expand_dims(targets, axis=-1))
target_position = jnp.argmax(target_match, axis=-1)
print(f"      Target position shape: {target_position.shape}")
print(f"      Sample positions [0,:5]: {target_position[0, :5]}")
print(f"      Any target not found: {(~jnp.any(target_match, axis=-1)).any()}")

print(f"\n   Step 9: Final loss")
target_cluster_log_prob = jnp.take_along_axis(
    cluster_log_probs, jnp.expand_dims(target_cluster_ids, axis=-1), axis=-1
).squeeze(axis=-1)
target_item_log_prob = jnp.take_along_axis(
    item_log_probs, jnp.expand_dims(target_position, axis=-1), axis=-1
).squeeze(axis=-1)

hierarchical_log_prob = target_cluster_log_prob + target_item_log_prob
hierarchical_xent = -hierarchical_log_prob * loss_mask
total_loss = jnp.sum(hierarchical_xent) / (jnp.sum(loss_mask) + 1e-8)

print(f"      Cluster log prob mean: {jnp.mean(target_cluster_log_prob):.4f}")
print(f"      Item log prob mean: {jnp.mean(target_item_log_prob):.4f}")
print(f"      Combined log prob mean: {jnp.mean(hierarchical_log_prob):.4f}")
print(f"      Total loss: {total_loss:.4f}")
print(f"      Loss is NaN: {jnp.isnan(total_loss)}")
print(f"      Loss is Inf: {jnp.isinf(total_loss)}")

print("\n" + "=" * 80)
if jnp.isnan(total_loss):
    print("❌ FOUND NaN in hierarchical softmax!")
    print("\nPossible causes:")
    print("1. Invalid cluster assignments (items not in their assigned cluster)")
    print("2. Empty clusters (no items in cluster)")
    print("3. log_softmax of all -inf values")
    print("4. Division by zero in loss computation")
else:
    print("✅ Hierarchical softmax OK!")
