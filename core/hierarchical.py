"""
Hierarchical Softmax for EfficientIDS - Flax Port

Direct port from PAXml's ItemLanguageSoftmax.
Only replaces PAXml classes with Flax equivalents.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Tuple, Any
import dataclasses


@dataclasses.dataclass
class ClusteringInfo:
    """Clustering information for hierarchical softmax."""
    cluster_assignments: jnp.ndarray  # [num_items] -> cluster_id
    cluster_indices: jnp.ndarray      # [num_clusters, max_cluster_size] -> item_ids
    in_cluster_id: jnp.ndarray        # [num_items] -> position_in_cluster
    cluster_embeddings: Optional[jnp.ndarray] = None  # [num_clusters, dim]

    @property
    def num_items(self):
        return len(self.cluster_assignments)

    @property
    def num_clusters(self):
        return len(self.cluster_indices)

    @property
    def max_cluster_size(self):
        return self.cluster_indices.shape[1]


class HierarchicalSoftmax(nn.Module):
    """
    Two-level hierarchical softmax - exact port from PAXml.

    PAXml: ItemLanguageSoftmax.get_logits_training + _compute_xent_unified_hierarchical
    Mode: use_item_input_dnn_everywhere=True
    """
    num_items: int
    num_clusters: int
    item_embedding_dim: int  # HIGH-dimensional (2048) when use_item_input_dnn_everywhere=True
    clustering_info: ClusteringInfo
    use_item_input_dnn_everywhere: bool = True
    item_input_adapter: Any = None  # For item embeddings (384→2048)
    cluster_input_adapter: Any = None  # For cluster embeddings (118→2048)

    @nn.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,  # [..., hidden_size] e.g. 2048
        item_embeddings: jnp.ndarray,  # [num_items, 384] - will be projected
        targets: Optional[jnp.ndarray] = None,
        loss_mask: Optional[jnp.ndarray] = None,
        training: bool = True,
    ) -> Tuple[jnp.ndarray, dict]:
        """
        PAXml exact port with use_item_input_dnn_everywhere=True.

        Flow:
        1. Project cluster embeddings (118-dim → 2048-dim)
        2. Project item embeddings (384-dim → 2048-dim)
        3. Compute cluster logits: einsum(hidden[2048], cluster_embs[2048])
        4. Compute item logits: einsum(hidden[2048], item_embs[2048])
        5. Two-level hierarchical loss
        """

        # PAXml lines 1088-1109: Project embeddings to model dimension
        # if self.use_item_input_dnn_everywhere:
        #     all_item_embeddings = self.item_input_dnn(all_item_embeddings)
        #     cluster_embeddings = self.item_input_dnn(cluster_embeddings)

        # Project cluster embeddings to high-dimensional space
        cluster_embeddings_raw = self.clustering_info.cluster_embeddings  # [100, 118]
        cluster_embeddings = self.cluster_input_adapter(cluster_embeddings_raw)  # [100, 2048]

        # Compute cluster logits in high-dimensional space
        # PAXml: jnp.einsum('...j,ij->...i', inputs, cluster_embeddings)
        cluster_logits = jnp.einsum(
            '...d,cd->...c',
            hidden_states,
            cluster_embeddings
        )

        if training and targets is not None:
            return self._compute_training_loss(
                hidden_states=hidden_states,
                item_embeddings=item_embeddings,
                cluster_logits=cluster_logits,
                targets=targets,
                loss_mask=loss_mask,
            )
        else:
            # Inference: compute full logits for all items
            logits = self._compute_inference_logits(
                hidden_states=hidden_states,
                item_embeddings=item_embeddings,
            )
            return logits, {}

    def _compute_training_loss(
        self,
        hidden_states: jnp.ndarray,  # [batch, seq_len, 2048]
        item_embeddings: jnp.ndarray,  # [num_items, 384]
        cluster_logits: jnp.ndarray,
        targets: jnp.ndarray,
        loss_mask: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, dict]:
        """
        Exact port of PAXml's _compute_xent_unified_hierarchical.

        PAXml lines 1380-1561 in interleaved_transformer_lm.py
        """

        # Get target cluster IDs
        item_cluster_ids = jnp.take(
            self.clustering_info.cluster_assignments,
            targets
        )

        # Get cluster members for target clusters
        cluster_members = jnp.take(
            self.clustering_info.cluster_indices,
            item_cluster_ids,
            axis=0
        )

        # ===== P(c|H): Probability of target cluster =====
        cluster_log_probs = jax.nn.log_softmax(cluster_logits.astype(jnp.float32))
        target_cluster_log_prob = jnp.take_along_axis(
            cluster_log_probs,
            jnp.expand_dims(item_cluster_ids, axis=-1),
            axis=-1
        ).squeeze(axis=-1)

        # ===== P(w|c(w),H): Probability of target item within cluster =====
        # PAXml: Project item embeddings to high-dimensional space
        # all_item_embeddings = self.item_input_dnn(all_item_embeddings)
        item_embeddings_projected = self.item_input_adapter(item_embeddings)  # [num_items, 2048]

        # Get embeddings for cluster members
        valid_mask = cluster_members != -1
        cluster_member_embeddings = jnp.take(
            item_embeddings_projected,
            jnp.maximum(cluster_members, 0),
            axis=0
        )

        # PAXml lines 1138-1146: Compute logits WITH masking (apply -inf during computation)
        logits_of_cluster_members = jnp.where(
            valid_mask,
            jnp.einsum(
                '...d,...md->...m',
                hidden_states,
                cluster_member_embeddings
            ),
            -jnp.inf  # PAXml uses -jnp.inf here, then replaces with -1e9 later
        )

        # PAXml line 1467-1471: Replace -inf with -1e9 before log_softmax
        masked_cluster_member_logits = jnp.where(
            valid_mask,
            logits_of_cluster_members,
            -1e9  # Use -1e9 like original, not -jnp.inf
        )

        # PAXml: Compute log probabilities within cluster
        cluster_member_log_probs = jax.nn.log_softmax(
            masked_cluster_member_logits.astype(jnp.float32)
        )

        # PAXml line 1478: Set invalid positions to 0.0
        cluster_member_log_probs = jnp.where(valid_mask, cluster_member_log_probs, 0.0)

        # PAXml lines 1481-1484: Find position of target item within cluster members
        target_in_cluster_match = jnp.equal(cluster_members, jnp.expand_dims(targets, axis=-1))
        target_position_in_cluster = jnp.argmax(target_in_cluster_match, axis=-1)

        # Safety: Check if target was actually found in cluster
        target_found_in_cluster = jnp.any(target_in_cluster_match, axis=-1)

        # PAXml lines 1486-1490: Get log probability of target item within cluster
        target_item_in_cluster_log_prob = jnp.take_along_axis(
            cluster_member_log_probs,
            jnp.expand_dims(target_position_in_cluster, axis=-1),
            axis=-1
        ).squeeze(axis=-1)

        # Safety: If target not found, set log_prob to 0 (will be masked by loss_mask anyway)
        target_item_in_cluster_log_prob = jnp.where(
            target_found_in_cluster,
            target_item_in_cluster_log_prob,
            0.0
        )

        # ===== Combined Loss =====
        # PAXml lines 1521-1522: Standard unified hierarchical softmax
        # log P(w|H) = log P(c(w)|H) + log P(w|c(w),H)
        hierarchical_log_prob = target_cluster_log_prob + target_item_in_cluster_log_prob
        hierarchical_xent = -hierarchical_log_prob * loss_mask

        # Total loss
        total_loss = jnp.sum(hierarchical_xent) / (jnp.sum(loss_mask) + 1e-8)

        # Compute metrics (match PAXml's summary computations)
        cluster_preds = jnp.argmax(cluster_logits, axis=-1)
        cluster_acc = jnp.sum((cluster_preds == item_cluster_ids).astype(jnp.float32) * loss_mask) / (jnp.sum(loss_mask) + 1e-8)

        metrics = {
            'total_loss': total_loss,
            'cluster_loss': -jnp.sum(target_cluster_log_prob * loss_mask) / (jnp.sum(loss_mask) + 1e-8),
            'item_loss': -jnp.sum(target_item_in_cluster_log_prob * loss_mask) / (jnp.sum(loss_mask) + 1e-8),
            'cluster_accuracy': cluster_acc,
        }

        # Return dummy logits
        dummy_logits = jnp.zeros(hidden_states.shape[:-1] + (self.num_items,))
        return dummy_logits, metrics

    def _compute_inference_logits(
        self,
        hidden_states: jnp.ndarray,  # [batch, seq_len, model_dims]
        item_embeddings: jnp.ndarray,  # [num_items, item_embedding_dim]
    ) -> jnp.ndarray:
        """
        Compute full item logits for inference.

        Since we use adapters, we need to project item embeddings to model space first.

        Args:
            hidden_states: [batch, seq_len, model_dims] (e.g., 2048)
            item_embeddings: [num_items, item_embedding_dim] (e.g., 384)

        Returns:
            logits: [batch, seq_len, num_items]
        """
        # Project item embeddings to model space (384 -> 2048)
        item_embeddings_projected = self.item_input_adapter(item_embeddings)  # [num_items, model_dims]

        # Compute logits in model space
        logits = jnp.einsum('...d,id->...i', hidden_states, item_embeddings_projected)
        return logits
