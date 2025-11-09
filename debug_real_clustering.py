"""
Debug real clustering data to find the NaN source.
"""

import numpy as np
import jax.numpy as jnp
from data.dataset import MovieLensDataset

print("=" * 80)
print("Real Clustering Data Debug")
print("=" * 80)

# Load real data
dataset = MovieLensDataset(
    data_dir='data/ml1m_processed/processed',
    split='train',
    max_seq_len=32,
    batch_size=2,
    shuffle=False,
    max_sequences=10,
)

clustering = dataset.get_clustering_info()

print('\n1. Clustering info shapes:')
print(f'   cluster_indices: {clustering.cluster_indices.shape}')
print(f'   cluster_assignments: {clustering.cluster_assignments.shape}')
print(f'   Sample cluster_indices[0]: {clustering.cluster_indices[0][:10]}')
print(f'   Has -1: {(clustering.cluster_indices == -1).any()}')
print(f'   Min value: {clustering.cluster_indices.min()}')
print(f'   Max value: {clustering.cluster_indices.max()}')

# Get a real batch
batch = next(iter(dataset))
targets = batch['targets'][:2, :5]  # Small sample
print(f'\n2. Targets:')
print(f'   Shape: {targets.shape}')
print(f'   Sample [0]: {targets[0]}')
print(f'   Min: {targets.min()}')
print(f'   Max: {targets.max()}')

# Get cluster IDs
target_cluster_ids = jnp.take(clustering.cluster_assignments, targets)
print(f'\n3. Target cluster IDs:')
print(f'   Shape: {target_cluster_ids.shape}')
print(f'   Sample [0]: {target_cluster_ids[0]}')
print(f'   Min: {target_cluster_ids.min()}')
print(f'   Max: {target_cluster_ids.max()}')

# Try simple indexing (correct approach for JAX)
print(f'\n4. Using simple indexing:')
print(f'   cluster_indices shape: {clustering.cluster_indices.shape}')
print(f'   target_cluster_ids shape: {target_cluster_ids.shape}')

# Convert to JAX array and use simple indexing
cluster_indices_jax = jnp.array(clustering.cluster_indices)
cluster_members = cluster_indices_jax[target_cluster_ids]  # [2, 5, 195]

print(f'   Result shape: {cluster_members.shape}')
print(f'   Has -1: {(cluster_members == -1).any()}')
print(f'   Sample [0,0,:10]: {cluster_members[0, 0, :10]}')
print(f'   Min: {cluster_members.min()}')
print(f'   Max: {cluster_members.max()}')

# Check if targets are in their clusters
print(f'\n5. Checking if targets are in their assigned clusters:')
for b in range(min(2, targets.shape[0])):
    for s in range(min(3, targets.shape[1])):
        target = targets[b, s]
        cluster_id = target_cluster_ids[b, s]
        members = cluster_members[b, s]
        is_in_cluster = jnp.any(members == target)
        print(f'   [{b},{s}] target={target}, cluster={cluster_id}, in_cluster={is_in_cluster}')
        if not is_in_cluster:
            print(f'      WARNING: Target {target} NOT in cluster {cluster_id}!')
            print(f'      Cluster members: {members[members != -1][:10]}')

# Now test the embedding lookup
print(f'\n6. Testing embedding lookup:')
item_embeddings = jnp.ones((dataset.num_items, 64))  # Dummy embeddings
valid_mask = cluster_members != -1
safe_cluster_members = jnp.where(valid_mask, cluster_members, 0)

print(f'   safe_cluster_members shape: {safe_cluster_members.shape}')
print(f'   safe_cluster_members min: {safe_cluster_members.min()}')
print(f'   safe_cluster_members max: {safe_cluster_members.max()}')
print(f'   Any >= num_items: {(safe_cluster_members >= dataset.num_items).any()}')

# Flatten and gather
orig_shape = safe_cluster_members.shape
flat_indices = safe_cluster_members.reshape(-1)
print(f'   flat_indices shape: {flat_indices.shape}')
print(f'   flat_indices range: [{flat_indices.min()}, {flat_indices.max()}]')

flat_embeddings = item_embeddings[flat_indices]
print(f'   flat_embeddings shape: {flat_embeddings.shape}')
print(f'   Has NaN: {jnp.isnan(flat_embeddings).any()}')

cluster_member_embeddings = flat_embeddings.reshape(*orig_shape, -1)
print(f'   cluster_member_embeddings shape: {cluster_member_embeddings.shape}')
print(f'   Has NaN: {jnp.isnan(cluster_member_embeddings).any()}')

print("\n" + "=" * 80)
print("Debug complete!")
