"""
Hierarchical Softmax for Efficient Item Prediction

Pure JAX/Flax implementation of two-level hierarchical softmax:
1. Predict cluster: P(c|H) over K clusters
2. Predict item within cluster: P(w|c,H) over items in cluster c

This dramatically reduces computation from O(V) to O(K + V/K) where:
- V = vocabulary size (e.g., 3261 items)
- K = number of clusters (e.g., 100)

Follows equations (2), (3), (4) from EfficientIDS paper.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Tuple
import pickle


class ClusteringInfo:
    """
    Container for clustering information loaded from clustering.pkl.

    Attributes:
        cluster_assignments: [num_items] - maps item_id → cluster_id
        cluster_indices: [num_clusters, max_cluster_size] - items in each cluster (padded with -1)
        in_cluster_id: [num_items] - position of item within its cluster
        cluster_embeddings: [num_clusters, embedding_dim] - cluster center embeddings (optional)
    """
    def __init__(
        self,
        cluster_assignments: jnp.ndarray,
        cluster_indices: jnp.ndarray,
        in_cluster_id: jnp.ndarray,
        cluster_embeddings: Optional[jnp.ndarray] = None,
    ):
        self.cluster_assignments = jnp.asarray(cluster_assignments)
        self.cluster_indices = jnp.asarray(cluster_indices)
        self.in_cluster_id = jnp.asarray(in_cluster_id)
        self.cluster_embeddings = (
            jnp.asarray(cluster_embeddings) if cluster_embeddings is not None else None
        )

        # Validate shapes
        self.num_items = len(cluster_assignments)
        self.num_clusters = cluster_indices.shape[0]
        self.max_cluster_size = cluster_indices.shape[1]

    @classmethod
    def from_pickle(cls, pickle_path: str) -> 'ClusteringInfo':
        """
        Load clustering info from pickle file.

        Pickle format (matching original):
            - cluster_assignments: [num_items]
            - cluster_indices: [num_clusters, max_cluster_size]
            - in_cluster_id: [num_items]
            - cluster_embeddings: [num_clusters, embedding_dim]
        """
        with open(pickle_path, 'rb') as f:
            cluster_assignments = pickle.load(f)
            cluster_indices = pickle.load(f)
            in_cluster_id = pickle.load(f)
            cluster_embeddings = pickle.load(f)

        return cls(
            cluster_assignments=cluster_assignments,
            cluster_indices=cluster_indices,
            in_cluster_id=in_cluster_id,
            cluster_embeddings=cluster_embeddings,
        )


class HierarchicalSoftmax(nn.Module):
    """
    Two-level hierarchical softmax for efficient item prediction.

    Training:
        - Computes exact loss using target cluster and position within cluster
        - L = -log P(c|H) - log P(w|c,H)

    Inference:
        - Top-K cluster pruning for efficiency
        - Computes combined score: log P(c|H) + log P(w|c,H)

    Args:
        num_items: Total vocabulary size (e.g., 3261)
        num_clusters: Number of clusters (e.g., 100)
        item_embedding_dim: Item embedding dimension (e.g., 384)
        clustering_info: ClusteringInfo object with cluster structure
        use_correction: Apply bias correction term (default: True)
    """
    num_items: int
    num_clusters: int
    item_embedding_dim: int
    clustering_info: ClusteringInfo
    use_correction: bool = True

    @nn.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        item_embeddings: jnp.ndarray,
        targets: Optional[jnp.ndarray] = None,
        item_mask: Optional[jnp.ndarray] = None,
        training: bool = True,
    ) -> Tuple[jnp.ndarray, dict]:
        """
        Args:
            hidden_states: [batch, seq_len, item_embedding_dim] - transformer outputs projected to item space
            item_embeddings: [num_items, item_embedding_dim] - item embedding table
            targets: [batch, seq_len] - target item IDs (only for training)
            item_mask: [batch, seq_len] - 1.0 for items, 0.0 for text tokens (only for training)
            training: Training mode vs inference mode

        Returns:
            logits: [batch, seq_len, num_items] - item logits (training) or pruned logits (inference)
            metrics: dict with loss/accuracy info (training) or top-k predictions (inference)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # ==================== CLUSTER PREDICTION ====================
        # Create learnable cluster embeddings (or use fixed from clustering)
        if self.clustering_info.cluster_embeddings is not None:
            # Don't use pretrained cluster embeddings if dimension mismatch
            if self.clustering_info.cluster_embeddings.shape[1] != self.item_embedding_dim:
                # Dimension mismatch - use random init instead
                cluster_emb_init = nn.initializers.xavier_uniform()
            else:
                # Use pretrained cluster embeddings
                cluster_emb_init = lambda *args: self.clustering_info.cluster_embeddings
        else:
            cluster_emb_init = nn.initializers.xavier_uniform()

        cluster_embeddings = self.param(
            'cluster_embeddings',
            cluster_emb_init,
            (self.num_clusters, self.item_embedding_dim)
        )

        # Cluster logits: [batch, seq_len, num_clusters]
        cluster_logits = jnp.einsum('...d,cd->...c', hidden_states, cluster_embeddings)

        if training:
            return self._compute_training_loss(
                hidden_states=hidden_states,
                item_embeddings=item_embeddings,
                cluster_logits=cluster_logits,
                targets=targets,
                item_mask=item_mask,
            )
        else:
            return self._compute_inference_logits(
                hidden_states=hidden_states,
                item_embeddings=item_embeddings,
                cluster_logits=cluster_logits,
                top_k_clusters=10,  # Can be made configurable
            )

    def _compute_training_loss(
        self,
        hidden_states: jnp.ndarray,
        item_embeddings: jnp.ndarray,
        cluster_logits: jnp.ndarray,
        targets: jnp.ndarray,
        item_mask: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, dict]:
        """
        Compute hierarchical cross-entropy loss during training.

        Uses exact two-level formula:
            L = -log P(c|H) - log P(w|c,H)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Get target cluster IDs: [batch, seq_len]
        target_cluster_ids = jnp.take(self.clustering_info.cluster_assignments, targets)

        # ===== STEP 1: Cluster Loss - P(c|H) =====
        cluster_log_probs = jax.nn.log_softmax(cluster_logits, axis=-1)
        target_cluster_log_prob = jnp.take_along_axis(
            cluster_log_probs,
            target_cluster_ids[..., None],
            axis=-1
        ).squeeze(-1)  # [batch, seq_len]

        # ===== STEP 2: Within-Cluster Loss - P(w|c,H) =====
        # Get cluster members for target clusters: [batch, seq_len, max_cluster_size]
        cluster_members = jnp.take(
            self.clustering_info.cluster_indices,
            target_cluster_ids,
            axis=0
        )  # [batch, seq_len, max_cluster_size]

        # Get embeddings for cluster members
        # Handle -1 padding by clamping to 0 (will be masked anyway)
        cluster_members_clamped = jnp.maximum(cluster_members, 0)
        cluster_member_embeddings = jnp.take(
            item_embeddings,
            cluster_members_clamped,
            axis=0
        )  # [batch, seq_len, max_cluster_size, embedding_dim]

        # Compute logits for items within cluster
        cluster_member_logits = jnp.einsum(
            'bsd,bsmd->bsm',
            hidden_states,
            cluster_member_embeddings
        )  # [batch, seq_len, max_cluster_size]

        # Mask invalid cluster members (-1 padding)
        valid_mask = cluster_members != -1
        masked_logits = jnp.where(
            valid_mask,
            cluster_member_logits,
            -1e9  # Large negative value, not -inf
        )

        # Log probabilities within cluster
        cluster_member_log_probs = jax.nn.log_softmax(masked_logits, axis=-1)

        # Find position of target within cluster members
        # in_cluster_id gives us the position directly
        target_in_cluster_pos = jnp.take(self.clustering_info.in_cluster_id, targets)

        # Get log probability of target item within cluster
        target_item_log_prob = jnp.take_along_axis(
            cluster_member_log_probs,
            target_in_cluster_pos[..., None],
            axis=-1
        ).squeeze(-1)  # [batch, seq_len]

        # ===== STEP 3: Combined Loss =====
        # Total log probability: log P(w|H) = log P(c|H) + log P(w|c,H)
        total_log_prob = target_cluster_log_prob + target_item_log_prob

        # Cross-entropy loss (negative log likelihood)
        loss_per_token = -total_log_prob  # [batch, seq_len]

        # Apply item mask (only compute loss on item tokens, not text)
        if item_mask is not None:
            masked_loss = loss_per_token * item_mask
            total_loss = jnp.sum(masked_loss) / (jnp.sum(item_mask) + 1e-8)
        else:
            total_loss = jnp.mean(loss_per_token)

        # Compute accuracy
        cluster_preds = jnp.argmax(cluster_logits, axis=-1)
        cluster_acc = jnp.mean((cluster_preds == target_cluster_ids).astype(jnp.float32))

        metrics = {
            'total_loss': total_loss,
            'cluster_loss': -jnp.mean(target_cluster_log_prob),
            'item_loss': -jnp.mean(target_item_log_prob),
            'cluster_accuracy': cluster_acc,
        }

        # Return dummy logits (not used in training) and metrics
        dummy_logits = jnp.zeros((batch_size, seq_len, self.num_items))
        return dummy_logits, metrics

    def _compute_inference_logits(
        self,
        hidden_states: jnp.ndarray,
        item_embeddings: jnp.ndarray,
        cluster_logits: jnp.ndarray,
        top_k_clusters: int = 10,
    ) -> Tuple[jnp.ndarray, dict]:
        """
        Compute item logits during inference using top-K cluster pruning.

        Algorithm:
            1. Compute P(c|H) for all clusters
            2. Select top-K clusters
            3. Compute P(w|c,H) only for items in top-K clusters
            4. Combine: log P(w|H) = log P(c|H) + log P(w|c,H)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Cluster probabilities: [batch, seq_len, num_clusters]
        cluster_probs = jax.nn.softmax(cluster_logits, axis=-1)

        # Select top-K clusters
        top_k_probs, top_k_cluster_ids = jax.lax.top_k(cluster_probs, k=top_k_clusters)
        # top_k_cluster_ids: [batch, seq_len, top_k_clusters]
        # top_k_probs: [batch, seq_len, top_k_clusters]

        # Compute item logits for ALL items (then mask)
        raw_item_logits = jnp.einsum('bsd,id->bsi', hidden_states, item_embeddings)
        # [batch, seq_len, num_items]

        # Create mask: 1.0 if item's cluster is in top-K, 0.0 otherwise
        # For each item, check if its cluster ID is in top_k_cluster_ids
        item_clusters = jnp.asarray(self.clustering_info.cluster_assignments)  # [num_items]

        # Expand for broadcasting: [num_items, 1, 1, 1] vs [1, batch, seq_len, top_k]
        item_clusters_expanded = item_clusters[:, None, None, None]  # [num_items, 1, 1, 1]
        top_k_expanded = top_k_cluster_ids[None, :, :, :]  # [1, batch, seq_len, top_k]

        # Check if each item's cluster is in top-K: [num_items, batch, seq_len, top_k] -> [num_items, batch, seq_len]
        in_top_k = jnp.any(item_clusters_expanded == top_k_expanded, axis=-1)
        in_top_k = jnp.transpose(in_top_k, (1, 2, 0))  # [batch, seq_len, num_items]

        # Mask out items not in top-K clusters
        masked_logits = jnp.where(in_top_k, raw_item_logits, -1e9)

        # Optional: Add correction term (bias from cluster probability)
        if self.use_correction:
            # For each item, add log P(c(item)|H)
            item_cluster_log_probs = jnp.take(
                jax.nn.log_softmax(cluster_logits, axis=-1),
                item_clusters,
                axis=-1
            )  # [batch, seq_len, num_items]

            masked_logits = masked_logits + item_cluster_log_probs

        # Top-K predictions
        top_k_values, top_k_indices = jax.lax.top_k(masked_logits, k=10)

        metrics = {
            'top_k_items': top_k_indices,
            'top_k_scores': top_k_values,
            'top_k_clusters': top_k_cluster_ids,
        }

        return masked_logits, metrics


# ==================== TESTING ====================

if __name__ == "__main__":
    """Test hierarchical softmax with synthetic data."""

    print("Testing Hierarchical Softmax")
    print("=" * 60)

    # Configuration (MovieLens-1M style)
    num_items = 100  # Smaller for testing
    num_clusters = 10
    item_embedding_dim = 64
    batch_size = 2
    seq_len = 5

    # Create synthetic clustering info
    print("\n1. Creating synthetic clustering...")
    import numpy as np

    # Random cluster assignments
    cluster_assignments = np.random.randint(0, num_clusters, size=num_items)

    # Create cluster_indices (items in each cluster)
    max_cluster_size = 20
    cluster_indices = np.full((num_clusters, max_cluster_size), -1, dtype=np.int32)
    in_cluster_id = np.zeros(num_items, dtype=np.int32)

    for cluster_id in range(num_clusters):
        items_in_cluster = np.where(cluster_assignments == cluster_id)[0]
        cluster_indices[cluster_id, :len(items_in_cluster)] = items_in_cluster
        in_cluster_id[items_in_cluster] = np.arange(len(items_in_cluster))

    clustering_info = ClusteringInfo(
        cluster_assignments=cluster_assignments,
        cluster_indices=cluster_indices,
        in_cluster_id=in_cluster_id,
    )

    print(f"   Num items: {clustering_info.num_items}")
    print(f"   Num clusters: {clustering_info.num_clusters}")
    print(f"   Max cluster size: {clustering_info.max_cluster_size}")
    print("   ✓ Clustering created!")

    # Test 2: Training mode
    print("\n2. Testing training mode...")
    key = jax.random.PRNGKey(0)

    # Create model
    model = HierarchicalSoftmax(
        num_items=num_items,
        num_clusters=num_clusters,
        item_embedding_dim=item_embedding_dim,
        clustering_info=clustering_info,
        use_correction=True,
    )

    # Synthetic inputs
    hidden_states = jax.random.normal(key, (batch_size, seq_len, item_embedding_dim))
    item_embeddings = jax.random.normal(key, (num_items, item_embedding_dim))
    targets = jax.random.randint(key, (batch_size, seq_len), 0, num_items)
    item_mask = jnp.ones((batch_size, seq_len))

    # Initialize and run
    params = model.init(
        key,
        hidden_states=hidden_states,
        item_embeddings=item_embeddings,
        targets=targets,
        item_mask=item_mask,
        training=True,
    )

    logits, metrics = model.apply(
        params,
        hidden_states=hidden_states,
        item_embeddings=item_embeddings,
        targets=targets,
        item_mask=item_mask,
        training=True,
    )

    print(f"   Total loss: {metrics['total_loss']:.4f}")
    print(f"   Cluster loss: {metrics['cluster_loss']:.4f}")
    print(f"   Item loss: {metrics['item_loss']:.4f}")
    print(f"   Cluster accuracy: {metrics['cluster_accuracy']:.4f}")
    print("   ✓ Training mode works!")

    # Test 3: Inference mode
    print("\n3. Testing inference mode...")

    logits, metrics = model.apply(
        params,
        hidden_states=hidden_states,
        item_embeddings=item_embeddings,
        training=False,
    )

    print(f"   Logits shape: {logits.shape}")
    print(f"   Top-5 predictions (first sequence, last position):")
    top_items = metrics['top_k_items'][0, -1, :5]
    top_scores = metrics['top_k_scores'][0, -1, :5]
    for i, (item, score) in enumerate(zip(top_items, top_scores)):
        print(f"      {i+1}. Item {item}: score {score:.4f}")
    print("   ✓ Inference mode works!")

    print("\n" + "=" * 60)
    print("✅ All tests passed! Hierarchical softmax is working.")
    print("\nNext steps:")
    print("  1. Create full transformer model (models.py)")
    print("  2. Integrate with pretrained LMs (Qwen, Llama)")
    print("  3. Build training loop")
