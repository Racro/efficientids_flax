"""
Debug script to check data validity and identify NaN source.
"""

import jax
import jax.numpy as jnp
import numpy as np
from data.dataset import MovieLensDataset

print("=" * 80)
print("Data Validation Check")
print("=" * 80)

# Load dataset
dataset = MovieLensDataset(
    data_dir="data/ml1m_processed/processed",
    split="train",
    max_seq_len=32,
    batch_size=4,
    shuffle=False,
    max_sequences=100,
)

print(f"\n1. Dataset Info:")
print(f"   Sequences: {len(dataset.sequences)}")
print(f"   Num items: {dataset.num_items}")
print(f"   Num clusters: {dataset.num_clusters}")

# Get first batch
batch = next(iter(dataset))

print(f"\n2. Batch Shapes:")
for k, v in batch.items():
    print(f"   {k}: {v.shape}, dtype={v.dtype}")

print(f"\n3. Batch Value Ranges:")
print(f"   item_ids - min: {batch['item_ids'].min()}, max: {batch['item_ids'].max()}")
print(f"   targets - min: {batch['targets'].min()}, max: {batch['targets'].max()}")
print(f"   weights - min: {batch['weights'].min()}, max: {batch['weights'].max()}")
print(f"   cluster_ids - min: {batch['cluster_ids'].min()}, max: {batch['cluster_ids'].max()}")
print(f"   in_cluster_ids - min: {batch['in_cluster_ids'].min()}, max: {batch['in_cluster_ids'].max()}")

print(f"\n4. Check for Invalid Values:")
print(f"   item_ids - any NaN: {np.isnan(batch['item_ids']).any()}")
print(f"   item_ids - any negative: {(batch['item_ids'] < 0).any()}")
print(f"   item_ids - any >= num_items: {(batch['item_ids'] >= dataset.num_items).any()}")
print(f"   targets - any negative: {(batch['targets'] < 0).any()}")
print(f"   targets - any >= num_items: {(batch['targets'] >= dataset.num_items).any()}")
print(f"   cluster_ids - any >= num_clusters: {(batch['cluster_ids'] >= dataset.num_clusters).any()}")

print(f"\n5. Sample Data (First Sequence):")
print(f"   item_ids[0, :10]: {batch['item_ids'][0, :10]}")
print(f"   targets[0, :10]: {batch['targets'][0, :10]}")
print(f"   weights[0, :10]: {batch['weights'][0, :10]}")
print(f"   cluster_ids[0, :10]: {batch['cluster_ids'][0, :10]}")

print(f"\n6. Clustering Validity:")
clustering = dataset.get_clustering_info()
print(f"   Cluster assignments shape: {clustering.cluster_assignments.shape}")
print(f"   Cluster indices shape: {clustering.cluster_indices.shape}")
print(f"   In-cluster IDs shape: {clustering.in_cluster_id.shape}")

# Check if clustering is valid
print(f"\n7. Clustering Sanity Checks:")
print(f"   All items have cluster: {(clustering.cluster_assignments >= 0).all()}")
print(f"   All items have valid cluster: {(clustering.cluster_assignments < dataset.num_clusters).all()}")
print(f"   Cluster indices has -1 padding: {(clustering.cluster_indices == -1).any()}")

# Count items per cluster
for cluster_id in range(min(5, dataset.num_clusters)):
    items_in_cluster = (clustering.cluster_assignments == cluster_id).sum()
    print(f"   Cluster {cluster_id}: {items_in_cluster} items")

print("\n" + "=" * 80)
print("✅ Data validation complete!")
