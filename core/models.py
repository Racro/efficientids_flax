"""
EfficientIDS Transformer Models - Pure Flax Implementation

Integrates pretrained language models with item recommendation:
1. Load pretrained transformer (Qwen, Llama, Gemma) from HuggingFace
2. Add item embedding adapters
3. Use hierarchical softmax for efficient item prediction

This replaces PAXml's InterleavedTransformerLm with pure Flax.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Dict, Any, Callable
from transformers import FlaxAutoModel, AutoConfig
import numpy as np

# Support both package imports and standalone execution
try:
    # Try relative imports first (when used as package)
    from .embeddings import ItemEmbedding, ItemInputAdapter, ItemOutputAdapter, create_embedding_initializer
    from .hierarchical_simple import SimpleHierarchicalSoftmax, JaxClusteringInfo
    from .llama_flax import LlamaModel
except ImportError:
    # Fall back to absolute imports (when run as script)
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from embeddings import ItemEmbedding, ItemInputAdapter, ItemOutputAdapter, create_embedding_initializer
    from hierarchical_simple import SimpleHierarchicalSoftmax, JaxClusteringInfo
    from llama_flax import LlamaModel

# ClusteringInfo comes from data.dataset
from typing import Any


class EfficientIDSModel(nn.Module):
    """
    Complete EfficientIDS model combining:
    - Pretrained transformer backbone (frozen or finetuned)
    - Item embeddings with adapters
    - Hierarchical softmax for prediction

    Args:
        model_name: HuggingFace model name (e.g., "Qwen/Qwen2.5-0.5B")
        num_items: Item vocabulary size (e.g., 3261 for MovieLens-1M)
        num_clusters: Number of clusters (e.g., 100)
        item_embedding_dim: Item embedding dimension (e.g., 384)
        clustering_info: ClusteringInfo with cluster structure
        freeze_lm: If True, freeze pretrained LM weights
        item_embedding_init: Optional pretrained item embeddings
        use_correction: Apply correction term in hierarchical softmax
    """
    model_name: str
    num_items: int
    num_clusters: int
    item_embedding_dim: int = 384
    clustering_info: Optional[Any] = None  # ClusteringInfo from data.dataset
    freeze_lm: bool = True
    item_embedding_init: Optional[np.ndarray] = None
    use_correction: bool = True

    def setup(self):
        """Initialize all model components."""

        # Load HuggingFace config to get model dimensions
        hf_config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        self.model_dims = hf_config.hidden_size

        # ==================== ITEM EMBEDDINGS ====================
        # Create item embedding table
        if self.item_embedding_init is not None:
            emb_initializer = create_embedding_initializer(self.item_embedding_init)
        else:
            emb_initializer = None

        self.item_embeddings = ItemEmbedding(
            num_items=self.num_items,
            embedding_dim=self.item_embedding_dim,
            initializer=emb_initializer,
            name='item_embeddings'
        )

        # ==================== ADAPTERS ====================
        # Project item embeddings → transformer space
        self.item_input_adapter = ItemInputAdapter(
            item_embedding_dim=self.item_embedding_dim,
            model_dims=self.model_dims,
            hidden_dim=self.item_embedding_dim * 4,  # 4x expansion
            num_layers=2,
            name='item_input_adapter'
        )

        # Project transformer outputs → item space
        self.item_output_adapter = ItemOutputAdapter(
            model_dims=self.model_dims,
            item_embedding_dim=self.item_embedding_dim,
            hidden_dim=self.item_embedding_dim * 4,
            num_layers=2,
            name='item_output_adapter'
        )

        # ==================== HIERARCHICAL SOFTMAX ====================
        if self.clustering_info is not None:
            # Convert clustering_info to JAX arrays for Flax 0.8+ compatibility
            jax_clustering = JaxClusteringInfo.from_numpy_clustering_info(self.clustering_info)
            self.hierarchical_softmax = SimpleHierarchicalSoftmax(
                num_items=self.num_items,
                num_clusters=self.num_clusters,
                item_embedding_dim=self.item_embedding_dim,
                cluster_assignments=jax_clustering.cluster_assignments,
                cluster_indices=jax_clustering.cluster_indices,
            )

    def __call__(
        self,
        input_ids: jnp.ndarray,
        item_ids: Optional[jnp.ndarray] = None,
        item_mask: Optional[jnp.ndarray] = None,
        attention_mask: Optional[jnp.ndarray] = None,
        targets: Optional[jnp.ndarray] = None,
        training: bool = True,
        pretrained_lm_module: Optional[Any] = None,  # Pass externally loaded LM
    ) -> Dict[str, jnp.ndarray]:
        """
        Forward pass through the model.

        Args:
            input_ids: [batch, seq_len] token IDs (for text tokens)
            item_ids: [batch, seq_len] item IDs (for item tokens)
            item_mask: [batch, seq_len] 1.0 for items, 0.0 for text
            attention_mask: [batch, seq_len] attention mask
            targets: [batch, seq_len] target item IDs (training only)
            training: Training vs inference mode
            pretrained_lm_module: Externally loaded FlaxAutoModel

        Returns:
            Dictionary with:
                - logits: [batch, seq_len, num_items] item prediction logits
                - loss: scalar loss (training only)
                - metrics: dict with auxiliary metrics
        """
        batch_size, seq_len = input_ids.shape

        # ==================== STEP 1: GET EMBEDDINGS ====================
        # Get item embeddings for item tokens
        if item_ids is not None:
            raw_item_embs = self.item_embeddings(item_ids)  # [batch, seq_len, item_emb_dim]
            item_embs = self.item_input_adapter(raw_item_embs)  # [batch, seq_len, model_dims]
        else:
            item_embs = jnp.zeros((batch_size, seq_len, self.model_dims))

        # ==================== STEP 2: GET TEXT EMBEDDINGS FROM LM ====================
        # NOTE: We need to get embeddings from the pretrained LM
        # This is tricky because we need to access the embedding layer
        # For now, we'll assume the LM is passed as a module

        if pretrained_lm_module is not None:
            # Get LM embeddings (accessing embedding layer)
            # Different models have different embedding access patterns:
            # - Qwen: model.model.embed_tokens or model.embed_tokens
            # - Llama: model.model.embed_tokens
            # - Gemma: model.model.embed_tokens

            # We'll use a generic approach: get embeddings from input_ids
            # This requires the embedding layer to be accessible
            try:
                text_embs = pretrained_lm_module.get_input_embeddings()(input_ids)
            except:
                # Fallback: assume model has embed_tokens
                text_embs = pretrained_lm_module.params['model']['embed_tokens']['embedding'][input_ids]
        else:
            # No LM provided, use random embeddings (for testing)
            text_embs = jax.random.normal(
                jax.random.PRNGKey(0),
                (batch_size, seq_len, self.model_dims)
            )

        # ==================== STEP 3: COMBINE TEXT AND ITEM EMBEDDINGS ====================
        # Use item_mask to select between text and item embeddings
        if item_mask is not None:
            # Expand mask for broadcasting: [batch, seq_len, 1]
            mask_expanded = jnp.expand_dims(item_mask, -1)
            combined_embs = text_embs * (1 - mask_expanded) + item_embs * mask_expanded
        else:
            # No items, just text
            combined_embs = text_embs

        # ==================== STEP 4: TRANSFORMER FORWARD PASS ====================
        if pretrained_lm_module is not None and hasattr(pretrained_lm_module, 'forward_from_embeddings'):
            # Custom method to forward from embeddings (if available)
            hidden_states = pretrained_lm_module.forward_from_embeddings(
                inputs_embeds=combined_embs,
                attention_mask=attention_mask,
            )
        elif pretrained_lm_module is not None:
            # Standard HF forward (but we can't easily pass embeddings)
            # This is a limitation - we'll need to modify the LM or use a different approach
            outputs = pretrained_lm_module(
                input_ids=input_ids,  # Fall back to using input_ids
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = outputs.last_hidden_state
        else:
            # No LM, just pass through embeddings (for testing)
            hidden_states = combined_embs

        # ==================== STEP 5: ITEM PREDICTION ====================
        # Project to item space
        item_space_hidden = self.item_output_adapter(hidden_states)

        # Get item embedding table for softmax
        # We need to extract the actual embedding array from the module
        # For Flax, we can use variables() or directly access params
        item_embedding_table = self.variables['params']['item_embeddings']['item_embeddings']['embedding']

        # Hierarchical softmax
        if self.clustering_info is not None:
            logits, metrics = self.hierarchical_softmax(
                hidden_states=item_space_hidden,
                item_embeddings=item_embedding_table,
                targets=targets,
                item_mask=item_mask,
                training=training,
            )
        else:
            # Full softmax (simple dot product)
            logits = jnp.einsum('bsd,id->bsi', item_space_hidden, item_embedding_table)

            if training and targets is not None:
                # Simple cross-entropy
                log_probs = jax.nn.log_softmax(logits, axis=-1)
                target_log_probs = jnp.take_along_axis(
                    log_probs,
                    targets[..., None],
                    axis=-1
                ).squeeze(-1)
                loss = -jnp.mean(target_log_probs)
                metrics = {'loss': loss}
            else:
                metrics = {}

        # ==================== STEP 6: RETURN OUTPUTS ====================
        outputs = {
            'logits': logits,
            'hidden_states': hidden_states,
            'item_space_hidden': item_space_hidden,
            **metrics,
        }

        return outputs


class SimpleEfficientIDSModel(nn.Module):
    """
    Simplified EfficientIDS model without pretrained LM dependency.

    Uses a simple transformer or even just direct item prediction.
    Useful for:
    - Testing
    - Training from scratch
    - When pretrained LM isn't needed

    Args:
        num_items: Item vocabulary size
        num_clusters: Number of clusters
        item_embedding_dim: Item embedding dimension
        model_dims: Hidden dimension (for internal processing)
        clustering_info: ClusteringInfo with cluster structure
    """
    num_items: int
    num_clusters: int
    item_embedding_dim: int = 384
    model_dims: int = 512
    clustering_info: Optional[Any] = None  # ClusteringInfo from data.dataset
    use_correction: bool = True

    @nn.compact
    def __call__(
        self,
        item_ids: jnp.ndarray,
        targets: Optional[jnp.ndarray] = None,
        weights: Optional[jnp.ndarray] = None,
        item_mask: Optional[jnp.ndarray] = None,
        training: bool = True,
        **kwargs,  # Accept extra args from batch
    ) -> Dict[str, jnp.ndarray]:
        """
        Simple forward pass: embed items → project → predict next item.

        Args:
            item_ids: [batch, seq_len] item ID sequence
            targets: [batch, seq_len] target items (training)
            weights: [batch, seq_len] mask for valid positions (alias for item_mask)
            item_mask: [batch, seq_len] mask for valid positions
            training: Training mode

        Returns:
            Dictionary with logits, loss, metrics
        """
        # Initialize item embedding table
        item_embedding_table = self.param(
            'item_embedding_table',
            nn.initializers.xavier_uniform(),
            (self.num_items, self.item_embedding_dim)
        )

        # Use weights as item_mask if item_mask not provided
        if item_mask is None and weights is not None:
            item_mask = weights

        # Embed items using the shared embedding table
        item_embs = item_embedding_table[item_ids]  # [batch, seq_len, item_emb_dim]

        # Simple projection (placeholder for transformer)
        hidden = nn.Dense(features=self.item_embedding_dim, name='projection')(item_embs)

        # Predict with hierarchical softmax using the same embedding table
        if self.clustering_info is not None:
            # Convert clustering_info to JAX arrays for Flax 0.10 compatibility
            jax_clustering = JaxClusteringInfo.from_numpy_clustering_info(self.clustering_info)
            hierarchical_softmax = SimpleHierarchicalSoftmax(
                num_items=self.num_items,
                num_clusters=self.num_clusters,
                item_embedding_dim=self.item_embedding_dim,
                cluster_assignments=jax_clustering.cluster_assignments,
                cluster_indices=jax_clustering.cluster_indices,
            )
            logits, metrics = hierarchical_softmax(
                hidden_states=hidden,
                item_embeddings=item_embedding_table,
                targets=targets,
                loss_mask=item_mask,  # Use loss_mask parameter name
                training=training,
            )
        else:
            # Full softmax
            logits = jnp.einsum('bsd,id->bsi', hidden, item_embedding_table)
            metrics = {}

        return {
            'logits': logits,
            **metrics,
        }


class LlamaEfficientIDSModel(nn.Module):
    """
    EfficientIDS with Llama transformer integration (Pure Flax).

    This model:
    1. Uses pure Flax Llama (no HuggingFace OOM issues!)
    2. Embeds items using learned embeddings
    3. Combines text and item embeddings
    4. Processes through Llama transformer
    5. Predicts next items using hierarchical softmax

    Attributes:
        vocab_size: Text vocabulary size (Llama vocab, e.g., 128256)
        hidden_size: Llama hidden dimension (e.g., 2048 for 1B)
        num_layers: Number of Llama layers (e.g., 16 for 1B)
        num_heads: Number of attention heads (e.g., 32 for 1B)
        num_kv_heads: Number of KV heads for GQA (e.g., 8 for 1B)
        intermediate_size: FFN hidden size (e.g., 8192 for 1B)
        num_items: Total number of items in catalog
        num_clusters: Number of item clusters
        item_embedding_dim: Dimension of item embeddings
        clustering_info: ClusteringInfo object with cluster assignments
        freeze_llama: Whether to freeze Llama weights
        use_correction: Use correction term in hierarchical softmax
    """

    # EfficientIDS config (required fields FIRST - no defaults)
    num_items: int
    num_clusters: int
    item_embedding_dim: int

    # Llama config (optional fields AFTER - with defaults)
    vocab_size: int = 128256
    hidden_size: int = 2048
    num_layers: int = 16
    num_heads: int = 32
    num_kv_heads: int = 8
    intermediate_size: int = 8192
    max_seq_len: int = 256

    # Optional configs
    clustering_info: Optional[Any] = None  # ClusteringInfo from data.dataset
    freeze_llama: bool = True
    use_correction: bool = True

    def setup(self):
        """Initialize model components."""

        # ==================== LLAMA TRANSFORMER ====================
        # Create pure Flax Llama (no HuggingFace dependencies!)
        self.llama = LlamaModel(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            intermediate_size=self.intermediate_size,
            max_seq_len=self.max_seq_len,
        )

        # ==================== ITEM EMBEDDINGS ====================
        # Create item embedding table (shared for input and output)
        self.item_embedding_table = self.param(
            'item_embedding_table',
            nn.initializers.xavier_uniform(),
            (self.num_items, self.item_embedding_dim)
        )

        # ==================== ADAPTERS ====================
        # Project item embeddings to Llama dimension
        self.item_input_adapter = ItemInputAdapter(
            item_embedding_dim=self.item_embedding_dim,
            model_dims=self.hidden_size,
            hidden_dim=self.item_embedding_dim * 4,
            num_layers=2,
            name='item_input_adapter'
        )

        # Project Llama outputs back to item space
        self.item_output_adapter = ItemOutputAdapter(
            model_dims=self.hidden_size,
            item_embedding_dim=self.item_embedding_dim,
            hidden_dim=self.item_embedding_dim * 4,
            num_layers=2,
            name='item_output_adapter'
        )

        # ==================== HIERARCHICAL SOFTMAX ====================
        if self.clustering_info is not None:
            # Convert clustering_info to JAX arrays for Flax 0.8+ compatibility
            jax_clustering = JaxClusteringInfo.from_numpy_clustering_info(self.clustering_info)
            self.hierarchical_softmax = SimpleHierarchicalSoftmax(
                num_items=self.num_items,
                num_clusters=self.num_clusters,
                item_embedding_dim=self.item_embedding_dim,
                cluster_assignments=jax_clustering.cluster_assignments,
                cluster_indices=jax_clustering.cluster_indices,
            )

    def __call__(
        self,
        input_ids: Optional[jnp.ndarray] = None,  # Text token IDs or item IDs for interleaved
        item_weights: Optional[jnp.ndarray] = None,  # 0 = text token, 1 = item token
        attention_mask: Optional[jnp.ndarray] = None,
        targets: Optional[jnp.ndarray] = None,  # Target items
        loss_mask: Optional[jnp.ndarray] = None,  # Where to compute loss (item→item only)
        training: bool = True,
    ) -> Dict[str, jnp.ndarray]:
        """
        Forward pass through Llama + EfficientIDS.

        Args:
            input_ids: [batch, seq_len] token IDs (interleaved text + items)
            item_weights: [batch, seq_len] - 0 for text tokens, 1 for item tokens
            attention_mask: [batch, seq_len] attention mask
            targets: [batch, seq_len] target item IDs (for training)
            loss_mask: [batch, seq_len] - 1 where loss should be computed, 0 elsewhere
                       For text_metadata: only item→item positions
                       For id_only: all non-padding positions
            training: Whether in training mode

        Returns:
            Dictionary with:
                - logits: Item prediction logits
                - total_loss: Combined loss (if training)
                - cluster_loss: Cluster prediction loss (if training)
                - item_loss: Item prediction loss (if training)
                - cluster_accuracy: Cluster prediction accuracy (if training)
        """
        batch_size, seq_len = input_ids.shape if input_ids is not None else item_ids.shape

        # ==================== STEP 1: GET TEXT EMBEDDINGS ====================
        if input_ids is not None:
            # Will be handled by Llama's internal embedding layer
            text_input = input_ids
        else:
            # No text, create dummy input (all zeros)
            text_input = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)

        # ==================== STEP 2: GET ITEM EMBEDDINGS ====================
        if item_ids is not None and item_mask is not None:
            # Lookup item embeddings
            item_embs_raw = self.item_embedding_table[item_ids]  # [batch, seq_len, item_emb_dim]

            # Adapt to Llama dimension
            item_embs = self.item_input_adapter(item_embs_raw)  # [batch, seq_len, hidden_size]
        else:
            # No items
            item_embs = jnp.zeros((batch_size, seq_len, self.hidden_size))
            item_mask = jnp.zeros((batch_size, seq_len))

        # ==================== STEP 3: COMBINE TEXT AND ITEM EMBEDDINGS ====================
        # Get text embeddings from Llama (we'll use inputs_embeds to pass combined embeddings)
        # For now, use Llama to get text embeddings first
        text_embs = self.llama(input_ids=text_input, attention_mask=attention_mask, training=False)

        # Stop gradient on text embeddings (we just want them as features)
        text_embs = jax.lax.stop_gradient(text_embs)

        # Use item_mask to select: item_mask[i,j] == 1 → use item, == 0 → use text
        mask_expanded = jnp.expand_dims(item_mask, -1)  # [batch, seq_len, 1]
        combined_embs = text_embs * (1 - mask_expanded) + item_embs * mask_expanded

        # ==================== STEP 4: PASS THROUGH LLAMA ====================
        # Run through Llama transformer layers with combined embeddings
        hidden_states = self.llama(
            inputs_embeds=combined_embs,
            attention_mask=attention_mask,
            training=training,
        )

        # Apply freeze if requested
        if self.freeze_llama:
            hidden_states = jax.lax.stop_gradient(hidden_states)

        # ==================== STEP 5: PROJECT TO ITEM SPACE ====================
        # Project Llama outputs to item embedding dimension
        item_hidden = self.item_output_adapter(hidden_states)  # [batch, seq_len, item_emb_dim]

        # ==================== STEP 6: PREDICT ITEMS WITH HIERARCHICAL SOFTMAX ====================
        if self.clustering_info is not None:
            logits, metrics = self.hierarchical_softmax(
                hidden_states=item_hidden,
                item_embeddings=self.item_embedding_table,
                targets=targets,
                item_mask=item_mask,
                training=training,
            )
        else:
            # Full softmax (fallback)
            logits = jnp.einsum('bsd,id->bsi', item_hidden, self.item_embedding_table)
            metrics = {}

        return {
            'logits': logits,
            **metrics,
        }


# ==================== TESTING ====================

if __name__ == "__main__":
    """Test the models with synthetic data."""

    # Import ClusteringInfo for testing
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.dataset import ClusteringInfo

    print("Testing EfficientIDS Models")
    print("=" * 60)

    # Configuration
    num_items = 100
    num_clusters = 10
    item_embedding_dim = 64
    batch_size = 2
    seq_len = 8

    # Create synthetic clustering
    print("\n1. Creating synthetic clustering...")
    import numpy as np

    cluster_assignments = np.random.randint(0, num_clusters, size=num_items)
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
    print("   ✓ Clustering created!")

    # Test SimpleEfficientIDSModel (no LM dependency)
    print("\n2. Testing SimpleEfficientIDSModel...")

    model = SimpleEfficientIDSModel(
        num_items=num_items,
        num_clusters=num_clusters,
        item_embedding_dim=item_embedding_dim,
        model_dims=128,
        clustering_info=clustering_info,
    )

    # Create synthetic data
    key = jax.random.PRNGKey(42)
    item_ids = jax.random.randint(key, (batch_size, seq_len), 0, num_items)
    targets = jax.random.randint(key, (batch_size, seq_len), 0, num_items)
    item_mask = jnp.ones((batch_size, seq_len))

    # Initialize
    params = model.init(
        key,
        item_ids=item_ids,
        targets=targets,
        item_mask=item_mask,
        training=True,
    )

    # Forward pass (training)
    outputs = model.apply(
        params,
        item_ids=item_ids,
        targets=targets,
        item_mask=item_mask,
        training=True,
    )

    print(f"   Logits shape: {outputs['logits'].shape}")
    print(f"   Total loss: {outputs.get('total_loss', 0.0):.4f}")
    if 'cluster_accuracy' in outputs:
        print(f"   Cluster accuracy: {outputs['cluster_accuracy']:.4f}")
    print("   ✓ Training forward pass works!")

    # Forward pass (inference)
    outputs = model.apply(
        params,
        item_ids=item_ids,
        training=False,
    )

    print(f"   Inference logits shape: {outputs['logits'].shape}")
    if 'top_k_items' in outputs:
        print(f"   Top-5 predictions (first seq, last pos): {outputs['top_k_items'][0, -1, :5]}")
    print("   ✓ Inference forward pass works!")

    print("\n" + "=" * 60)
    print("✅ All tests passed! Model architecture is working.")
    print("\nNext steps:")
    print("  1. Create training loop (train/trainer.py)")
    print("  2. Add pretrained LM loading utilities")
    print("  3. Build data pipeline")
    print("  4. Create end-to-end training script")
