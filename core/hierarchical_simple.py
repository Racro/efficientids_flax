"""
Simplified Hierarchical Softmax for EfficientIDS

This is a simplified version that works with SimpleEfficientIDSModel
without requiring adapters or cluster embeddings.
"""

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from typing import Optional, Tuple, Dict, Any

# ClusteringInfo will be passed in from dataset.py
# It's a dataclass with np.ndarray fields


class SimpleHierarchicalSoftmax(nn.Module):
    """
    Simplified two-level hierarchical softmax for EfficientIDS.

    Works directly with item embeddings without requiring adapters.
    Compatible with SimpleEfficientIDSModel.
    """
    num_items: int
    num_clusters: int
    item_embedding_dim: int
    clustering_info: Any  # ClusteringInfo from data.dataset

    def setup(self):
        """Initialize cluster embeddings."""
        # Create learnable cluster embeddings
        self.cluster_embeddings = self.param(
            'cluster_embeddings',
            nn.initializers.xavier_uniform(),
            (self.num_clusters, self.item_embedding_dim)
        )

    def __call__(
        self,
        hidden_states: jnp.ndarray,  # [batch, seq_len, item_embedding_dim]
        item_embeddings: jnp.ndarray,  # [num_items, item_embedding_dim]
        targets: Optional[jnp.ndarray] = None,
        loss_mask: Optional[jnp.ndarray] = None,
        training: bool = True,
    ) -> Tuple[jnp.ndarray, Dict]:
        """
        Forward pass with hierarchical softmax.

        Args:
            hidden_states: [batch, seq_len, dim] - Model hidden states
            item_embeddings: [num_items, dim] - Item embedding table
            targets: [batch, seq_len] - Target item IDs (training only)
            loss_mask: [batch, seq_len] - Mask for valid positions
            training: Training mode flag

        Returns:
            (logits, metrics) tuple
        """
        if training and targets is not None:
            return self._compute_training_loss(
                hidden_states, item_embeddings, targets, loss_mask
            )
        else:
            # Inference: return dummy logits
            dummy_logits = jnp.zeros(hidden_states.shape[:-1] + (self.num_items,))
            return dummy_logits, {}

    def _compute_training_loss(
        self,
        hidden_states: jnp.ndarray,
        item_embeddings: jnp.ndarray,
        targets: jnp.ndarray,
        loss_mask: Optional[jnp.ndarray],
    ) -> Tuple[jnp.ndarray, Dict]:
        """Compute hierarchical softmax loss."""

        if loss_mask is None:
            loss_mask = jnp.ones(targets.shape, dtype=jnp.float32)

        # Step 1: Compute cluster logits
        cluster_logits = jnp.einsum('...d,cd->...c', hidden_states, self.cluster_embeddings)

        # Step 2: Get target cluster IDs
        target_cluster_ids = jnp.take(self.clustering_info.cluster_assignments, targets)

        # Step 3: Cluster loss
        cluster_log_probs = jax.nn.log_softmax(cluster_logits)
        target_cluster_log_prob = jnp.take_along_axis(
            cluster_log_probs,
            jnp.expand_dims(target_cluster_ids, axis=-1),
            axis=-1
        ).squeeze(axis=-1)

        # Step 4: Item-within-cluster loss
        # Get cluster members for target clusters (matching PAXml approach - flatten first!)
        # PAXml lines 1413-1430: flatten, take_along_axis, reshape
        batch_shape = target_cluster_ids.shape  # e.g., [batch, seq_len]

        # Flatten target_cluster_ids
        flat_cluster_ids = jnp.reshape(target_cluster_ids, [-1])  # [batch * seq_len]

        # Get cluster members for each flattened position
        cluster_members_flat = jnp.take_along_axis(
            self.clustering_info.cluster_indices,
            jnp.expand_dims(flat_cluster_ids, axis=-1),  # [batch*seq_len, 1]
            axis=0
        )  # [batch*seq_len, max_cluster_size]

        # Reshape back to [batch, seq_len, max_cluster_size]
        max_cluster_size = self.clustering_info.cluster_indices.shape[1]
        cluster_members = jnp.reshape(cluster_members_flat, (*batch_shape, max_cluster_size))

        # Mask for valid cluster members
        valid_mask = cluster_members != -1

        # Get embeddings for cluster members (matching PAXml's array_lookup behavior)
        # JAX gather: for each position [b,s,c], gather item_embeddings[cluster_members[b,s,c]]
        safe_cluster_members = jnp.where(valid_mask, cluster_members, 0)  # Replace -1 with 0

        # Reshape for batched gather
        orig_shape = safe_cluster_members.shape  # [batch, seq_len, max_cluster_size]
        flat_indices = safe_cluster_members.reshape(-1)  # [batch * seq_len * max_cluster_size]

        # Gather embeddings
        flat_embeddings = item_embeddings[flat_indices]  # [batch * seq_len * max_cluster_size, dim]

        # Reshape back
        cluster_member_embeddings = flat_embeddings.reshape(*orig_shape, -1)  # [batch, seq_len, max_cluster_size, dim]

        # Compute logits for items within cluster (matching PAXml einsum pattern)
        item_logits = jnp.einsum(
            '...d,...md->...m',
            hidden_states,
            cluster_member_embeddings
        )

        # Apply mask AFTER einsum (matching PAXml approach at line 1467-1470)
        # Set invalid items to large negative value BEFORE log_softmax
        masked_item_logits = jnp.where(valid_mask, item_logits, -1e9)

        # Item log probabilities within cluster
        item_log_probs = jax.nn.log_softmax(masked_item_logits)
        item_log_probs = jnp.where(valid_mask, item_log_probs, 0.0)

        # Find position of target within cluster members
        target_match = jnp.equal(cluster_members, jnp.expand_dims(targets, axis=-1))
        target_position = jnp.argmax(target_match, axis=-1)

        # Get log prob of target item within cluster
        target_item_log_prob = jnp.take_along_axis(
            item_log_probs,
            jnp.expand_dims(target_position, axis=-1),
            axis=-1
        ).squeeze(axis=-1)

        # Combined hierarchical log probability
        hierarchical_log_prob = target_cluster_log_prob + target_item_log_prob
        hierarchical_xent = -hierarchical_log_prob * loss_mask

        # Total loss
        total_loss = jnp.sum(hierarchical_xent) / (jnp.sum(loss_mask) + 1e-8)

        # Metrics
        cluster_preds = jnp.argmax(cluster_logits, axis=-1)
        cluster_acc = jnp.sum(
            (cluster_preds == target_cluster_ids).astype(jnp.float32) * loss_mask
        ) / (jnp.sum(loss_mask) + 1e-8)

        metrics = {
            'total_loss': total_loss,
            'cluster_loss': -jnp.sum(target_cluster_log_prob * loss_mask) / (jnp.sum(loss_mask) + 1e-8),
            'item_loss': -jnp.sum(target_item_log_prob * loss_mask) / (jnp.sum(loss_mask) + 1e-8),
            'cluster_accuracy': cluster_acc,
        }

        # Return dummy logits
        dummy_logits = jnp.zeros(hidden_states.shape[:-1] + (self.num_items,))
        return dummy_logits, metrics
