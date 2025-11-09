"""
Debug the embedding lookup specifically.
"""

import jax
import jax.numpy as jnp
import numpy as np

print("=" * 80)
print("Embedding Lookup Debug")
print("=" * 80)

# Simulate the problem
item_embeddings = jax.random.normal(jax.random.PRNGKey(42), (3260, 64))
cluster_members = jnp.array([
    [[16, 55, 72, -1, -1],
     [100, 200, 300, -1, -1]],
    [[1, 2, 3, 4, 5],
     [50, 51, 52, -1, -1]]
])  # Shape: (2, 2, 5) - batch=2, seq=2, max_cluster_size=5

print(f"\n1. Input shapes:")
print(f"   item_embeddings: {item_embeddings.shape}")
print(f"   cluster_members: {cluster_members.shape}")

print(f"\n2. Checking item_embeddings:")
print(f"   Has NaN: {jnp.isnan(item_embeddings).any()}")
print(f"   Has Inf: {jnp.isinf(item_embeddings).any()}")
print(f"   Mean: {jnp.mean(item_embeddings):.4f}")
print(f"   Std: {jnp.std(item_embeddings):.4f}")

print(f"\n3. Checking cluster_members:")
print(f"   Min: {cluster_members.min()}")
print(f"   Max: {cluster_members.max()}")
print(f"   Shape: {cluster_members.shape}")

print(f"\n4. Safe indexing (clamp -1 to 0):")
safe_cluster_members = jnp.maximum(cluster_members, 0)
print(f"   safe_cluster_members shape: {safe_cluster_members.shape}")
print(f"   safe_cluster_members[0,0]: {safe_cluster_members[0, 0]}")

print(f"\n5. Method 1: Advanced indexing (current approach):")
try:
    cluster_embs_v1 = item_embeddings[safe_cluster_members]
    print(f"   Result shape: {cluster_embs_v1.shape}")
    print(f"   Has NaN: {jnp.isnan(cluster_embs_v1).any()}")
    print(f"   Mean: {jnp.mean(cluster_embs_v1):.4f}")
except Exception as e:
    print(f"   ERROR: {e}")

print(f"\n6. Method 2: Reshape and gather:")
try:
    # Flatten cluster_members
    batch, seq, cluster_size = cluster_members.shape
    flat_indices = safe_cluster_members.reshape(-1)  # [batch * seq * cluster_size]

    # Gather embeddings
    flat_embs = item_embeddings[flat_indices]  # [batch * seq * cluster_size, dim]

    # Reshape back
    cluster_embs_v2 = flat_embs.reshape(batch, seq, cluster_size, -1)

    print(f"   Result shape: {cluster_embs_v2.shape}")
    print(f"   Has NaN: {jnp.isnan(cluster_embs_v2).any()}")
    print(f"   Mean: {jnp.mean(cluster_embs_v2):.4f}")
except Exception as e:
    print(f"   ERROR: {e}")

print(f"\n7. Method 3: vmap approach:")
try:
    def gather_for_position(indices):
        """Gather embeddings for one position."""
        safe_indices = jnp.maximum(indices, 0)
        return item_embeddings[safe_indices]

    # Vmap over batch and sequence
    cluster_embs_v3 = jax.vmap(jax.vmap(gather_for_position))(cluster_members)

    print(f"   Result shape: {cluster_embs_v3.shape}")
    print(f"   Has NaN: {jnp.isnan(cluster_embs_v3).any()}")
    print(f"   Mean: {jnp.mean(cluster_embs_v3):.4f}")
except Exception as e:
    print(f"   ERROR: {e}")

print(f"\n8. Test einsum with result:")
try:
    hidden_states = jax.random.normal(jax.random.PRNGKey(123), (2, 2, 64))

    # Use method 2 result
    item_logits = jnp.einsum('...d,...md->...m', hidden_states, cluster_embs_v2)

    print(f"   hidden_states shape: {hidden_states.shape}")
    print(f"   cluster_embs shape: {cluster_embs_v2.shape}")
    print(f"   item_logits shape: {item_logits.shape}")
    print(f"   item_logits has NaN: {jnp.isnan(item_logits).any()}")
    print(f"   item_logits mean: {jnp.mean(item_logits):.4f}")
except Exception as e:
    print(f"   ERROR: {e}")

print("\n" + "=" * 80)
print("Which method works?")
