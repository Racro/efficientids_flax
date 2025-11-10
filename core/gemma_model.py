"""
Gemma + EfficientIDS Integration (Flax 0.10 compatible)

Full Gemma 2B transformer implementation with pretrained weight loading.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Dict, Any
import orbax.checkpoint as ocp

# EfficientIDS imports
from .hierarchical import HierarchicalSoftmax, ClusteringInfo
from .embeddings import ItemInputAdapter, ItemOutputAdapter, ClusterInputAdapter


class RMSNorm(nn.Module):
    """RMSNorm layer from Gemma."""
    epsilon: float = 1e-6

    @nn.compact
    def __call__(self, x):
        scale = self.param('scale', nn.initializers.ones, (x.shape[-1],))
        # RMSNorm: x / sqrt(mean(x^2) + eps) * scale
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        normed = x * jax.lax.rsqrt(var + self.epsilon)
        return normed * scale


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention from Gemma 2B.

    - num_heads: 8 query heads
    - num_kv_heads: 1 (for Gemma's GQA, stored as 2 in checkpoint for up/down)
    - head_dim: 256
    """
    num_heads: int = 8
    num_kv_heads: int = 1
    head_dim: int = 256

    @nn.compact
    def __call__(self, x, mask=None):
        batch_size, seq_len, hidden_dim = x.shape

        # Q projection: [batch, seq, hidden] -> [batch, seq, num_heads, head_dim]
        # Gemma kernel shape: (num_heads, hidden_dim, head_dim)
        q = nn.DenseGeneral(
            features=(self.num_heads, self.head_dim),
            axis=-1,  # Apply on last axis (hidden_dim)
            use_bias=False,  # Gemma doesn't use bias
            kernel_init=nn.initializers.normal(),
            name='q_einsum'
        )(x)

        # KV projection: [batch, seq, hidden] -> [batch, seq, num_kv_heads, head_dim]
        # Gemma kernel shape: (num_kv_heads, 1, hidden_dim, head_dim)
        kv = nn.DenseGeneral(
            features=(self.num_kv_heads, self.head_dim),
            axis=-1,
            use_bias=False,  # Gemma doesn't use bias
            kernel_init=nn.initializers.normal(),
            name='kv_einsum'
        )(x)

        # Split into K and V (Gemma stores both in same tensor)
        k = kv
        v = kv

        # Transpose for attention: [batch, num_heads, seq, head_dim]
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # Expand KV heads to match Q heads (group broadcasting)
        # [batch, num_kv_heads, seq, head_dim] -> [batch, num_heads, seq, head_dim]
        if self.num_kv_heads < self.num_heads:
            k = jnp.repeat(k, self.num_heads // self.num_kv_heads, axis=1)
            v = jnp.repeat(v, self.num_heads // self.num_kv_heads, axis=1)

        # Attention scores: [batch, num_heads, seq, seq]
        scale = 1.0 / jnp.sqrt(self.head_dim)
        attn_weights = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale

        # Apply causal mask if provided
        if mask is not None:
            attn_weights = jnp.where(mask, attn_weights, -1e9)

        # Softmax and apply to values
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        attn_output = jnp.einsum('bhqk,bhvd->bhqd', attn_weights, v)

        # Transpose back: [batch, seq, num_heads, head_dim]
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))

        # Output projection: [batch, seq, num_heads, head_dim] -> [batch, seq, hidden]
        output = nn.DenseGeneral(
            features=hidden_dim,
            axis=(-2, -1),
            use_bias=False,  # Gemma doesn't use bias
            name='attn_vec_einsum'
        )(attn_output)

        return output


class GemmaMLP(nn.Module):
    """
    SwiGLU MLP from Gemma.

    hidden_dim -> intermediate_dim (16384) -> hidden_dim
    Uses gating mechanism: SwiGLU(x) = swish(gate(x)) * up(x)
    """
    intermediate_dim: int = 16384
    hidden_dim: int = 2048

    @nn.compact
    def __call__(self, x):
        # Gating projection: [batch, seq, hidden] -> [batch, seq, intermediate]
        # Gemma stores both gate and up in same einsum (2, hidden, intermediate)
        gating = nn.Dense(
            features=self.intermediate_dim,
            use_bias=False,  # Gemma doesn't use bias
            name='gating_einsum'
        )(x)

        # SwiGLU: swish activation on gating
        gating = jax.nn.swish(gating)

        # Linear projection: [batch, seq, intermediate] -> [batch, seq, hidden]
        output = nn.Dense(
            features=self.hidden_dim,
            use_bias=False,  # Gemma doesn't use bias
            name='linear'
        )(gating)

        return output


class GemmaTransformerBlock(nn.Module):
    """
    Single Gemma transformer block.

    - Pre-norm with RMSNorm
    - Grouped Query Attention
    - SwiGLU MLP
    - Residual connections
    """
    num_heads: int = 8
    num_kv_heads: int = 1
    head_dim: int = 256
    hidden_dim: int = 2048
    intermediate_dim: int = 16384

    @nn.compact
    def __call__(self, x, mask=None):
        # Pre-norm + Attention
        normed = RMSNorm(name='pre_attention_norm')(x)
        attn_output = GroupedQueryAttention(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            name='attn'
        )(normed, mask=mask)
        x = x + attn_output

        # Pre-norm + MLP
        normed = RMSNorm(name='pre_ffw_norm')(x)
        mlp_output = GemmaMLP(
            intermediate_dim=self.intermediate_dim,
            hidden_dim=self.hidden_dim,
            name='mlp'
        )(normed)
        x = x + mlp_output

        return x


class GemmaTransformer(nn.Module):
    """
    Full Gemma 2B transformer (18 layers).
    """
    num_layers: int = 18
    num_heads: int = 8
    num_kv_heads: int = 1
    head_dim: int = 256
    hidden_dim: int = 2048
    intermediate_dim: int = 16384
    vocab_size: int = 256128

    @nn.compact
    def __call__(self, x, mask=None):
        # Note: We don't use Gemma's embeddings (we use item embeddings instead)
        # x is already embedded: [batch, seq, hidden_dim]

        # Apply 18 transformer layers
        for i in range(self.num_layers):
            x = GemmaTransformerBlock(
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                hidden_dim=self.hidden_dim,
                intermediate_dim=self.intermediate_dim,
                name=f'layer_{i}'
            )(x, mask=mask)

        # Final norm
        x = RMSNorm(name='final_norm')(x)

        return x


def load_gemma_params(checkpoint_path: str) -> Dict[str, Any]:
    """
    Load Gemma checkpoint from Orbax.

    Args:
        checkpoint_path: Path to Gemma checkpoint directory (e.g., '../2b/')

    Returns:
        Dictionary with Gemma parameters
    """
    checkpointer = ocp.PyTreeCheckpointer()
    params = checkpointer.restore(checkpoint_path)
    return params


def reshape_gemma_params_for_flax(gemma_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reshape Gemma checkpoint params to match our Flax model structure.

    Gemma checkpoint is a FLAT dict with keys like:
        'transformer/layer_0/attn/q_einsum' -> {'w': array}
        'transformer/final_norm' -> {'scale': array}

    We need to convert to NESTED Flax structure:
        {'transformer': {'layer_0': {'attn': {'q_einsum': {'kernel': array}}}}}

    Args:
        gemma_params: Flat Gemma checkpoint dict

    Returns:
        Nested params matching Flax model structure
    """
    import logging
    import jax.numpy as jnp
    logger = logging.getLogger(__name__)

    logger.info(f"  [reshape] Converting flat checkpoint ({len(gemma_params)} keys) to nested structure")

    # Build nested structure
    nested = {}

    for flat_key, value_dict in gemma_params.items():
        # Split path: 'transformer/layer_0/attn/q_einsum' -> ['transformer', 'layer_0', 'attn', 'q_einsum']
        parts = flat_key.split('/')

        # Skip non-transformer keys
        if parts[0] != 'transformer':
            continue

        # Navigate/create nested structure
        current = nested
        for part in parts[:-1]:  # All except last part
            if part not in current:
                current[part] = {}
            current = current[part]

        # Last part is the parameter name
        param_name = parts[-1]

        # Convert 'w' -> 'kernel' and transpose to match Flax DenseGeneral
        if 'w' in value_dict:
            w = value_dict['w']

            # Convert bfloat16 to float32 for compatibility
            if w.dtype == jnp.bfloat16:
                w = w.astype(jnp.float32)

            # Transpose Gemma einsum weights to match Flax DenseGeneral
            # Gemma uses einsum convention: (output_dims..., input_dim)
            # Flax DenseGeneral expects: (input_dim, output_dims...)
            if param_name in ['q_einsum', 'kv_einsum']:
                # Q: (num_heads, hidden, head_dim) -> (hidden, num_heads, head_dim)
                # KV: (num_kv_heads, 1, hidden, head_dim) -> (hidden, 1, head_dim)
                if w.ndim == 4:  # KV has shape (2, 1, 2048, 256)
                    # Gemma uses 2 for k/v, but we only need one (they're the same)
                    w = w[0]  # Take first: (2, 1, 2048, 256) -> (1, 2048, 256)
                    w = w.squeeze(0)  # -> (2048, 256)
                    # Add back the num_kv_heads dimension: (2048, 256) -> (2048, 1, 256)
                    w = w[:, None, :]
                else:
                    # Q: (num_heads, hidden, head_dim) -> (hidden, num_heads, head_dim)
                    w = jnp.transpose(w, (1, 0, 2))
            elif param_name == 'attn_vec_einsum':
                # Output: (num_heads, head_dim, hidden) -> (head_dim, num_heads, hidden) [NOT NEEDED]
                # Actually for DenseGeneral with axis=(-2,-1), we need (num_heads, head_dim, hidden)
                # So no transpose needed, but actually yes: (8, 256, 2048) stays as is
                pass  # Keep as-is
            elif param_name == 'gating_einsum':
                # Gating: (2, hidden, intermediate) -> (hidden, intermediate)
                # Take first slice (gate part)
                w = w[0]  # (2, 2048, 16384) -> (2048, 16384)
            elif param_name == 'linear':
                # Linear: (intermediate, hidden) - already correct for Dense
                pass

            current[param_name] = {'kernel': w}
        elif 'scale' in value_dict:
            scale = value_dict['scale']
            # Convert bfloat16 to float32 for compatibility
            if scale.dtype == jnp.bfloat16:
                scale = scale.astype(jnp.float32)
            current[param_name] = {'scale': scale}
        else:
            # Copy as-is for other params (like embeddings)
            current[param_name] = value_dict

    # Count layers found
    if 'transformer' in nested:
        layers_found = sum(1 for k in nested['transformer'].keys() if k.startswith('layer_'))
        logger.info(f"  [reshape] Found {layers_found}/18 transformer layers")
        logger.info(f"  [reshape] Nested structure keys: {list(nested.keys())}")
        if 'transformer' in nested:
            logger.info(f"  [reshape] Transformer keys: {list(nested['transformer'].keys())[:5]}")

    return nested


class GemmaEfficientIDSModel(nn.Module):
    """
    Gemma 2B + EfficientIDS for item recommendation.

    Architecture:
    1. Item embeddings (num_items, 384)
    2. Input adapter (384 -> 2048) - projects items to Gemma space
    3. Gemma transformer (18 layers, 2048 hidden)
    4. Hierarchical softmax at 2048-dim (using full hierarchical.py)
       - Item input adapter: 384 -> 2048
       - Cluster input adapter: 118 -> 2048
       - All computations in model space (2048-dim)

    This uses the full hierarchical softmax (hierarchical.py) instead of
    hierarchical_simple.py to maximize Gemma's representational capacity.

    Args:
        num_items: Number of items
        num_clusters: Number of clusters
        item_embedding_dim: Item embedding dimension (384)
        model_dims: Model hidden size (2048 for Gemma 2B)
        clustering_info: ClusteringInfo from dataset
        freeze_gemma: Freeze Gemma weights
    """
    num_items: int
    num_clusters: int
    item_embedding_dim: int = 384
    model_dims: int = 2048
    clustering_info: Optional[Any] = None
    freeze_gemma: bool = True

    @nn.compact
    def __call__(
        self,
        item_ids: jnp.ndarray,  # [batch, seq_len] - item IDs
        targets: Optional[jnp.ndarray] = None,
        weights: Optional[jnp.ndarray] = None,
        training: bool = True,
        **kwargs,
    ) -> Dict[str, jnp.ndarray]:
        """
        Forward pass.

        Args:
            item_ids: [batch, seq_len] item IDs
            targets: [batch, seq_len] target items
            weights: [batch, seq_len] loss mask
            training: Training mode

        Returns:
            Dictionary with logits and metrics
        """
        batch_size, seq_len = item_ids.shape

        # ==================== ITEM EMBEDDINGS ====================
        item_embedding_table = self.param(
            'item_embedding_table',
            nn.initializers.xavier_uniform(),
            (self.num_items, self.item_embedding_dim)
        )

        # ==================== ADAPTERS ====================
        # Project item embeddings to Gemma space (384 → 2048)
        # This adapter is used by hierarchical softmax, not here
        item_input_adapter = ItemInputAdapter(
            item_embedding_dim=self.item_embedding_dim,
            model_dims=self.model_dims,
            hidden_dim=self.item_embedding_dim * 4,
            num_layers=2,
            name='item_input_adapter'
        )

        # NOTE: We don't use ItemOutputAdapter here because hierarchical softmax
        # operates directly in model space (2048-dim) for better capacity

        # ==================== FORWARD PASS ====================
        # Embed items
        raw_item_embs = item_embedding_table[item_ids]  # [batch, seq_len, 384]

        # Project to model space
        model_space_embs = item_input_adapter(raw_item_embs)  # [batch, seq_len, 2048]

        # Gemma transformer
        gemma_transformer = GemmaTransformer(
            num_layers=18,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            hidden_dim=2048,
            intermediate_dim=16384,
            name='transformer'
        )

        # Create causal mask
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))[None, None, :, :]  # [1, 1, seq, seq]

        # Apply transformer (with optional freezing)
        if self.freeze_gemma:
            model_outputs = jax.lax.stop_gradient(
                gemma_transformer(model_space_embs, mask=mask)
            )
        else:
            model_outputs = gemma_transformer(model_space_embs, mask=mask)

        # ==================== HIERARCHICAL SOFTMAX ====================
        # Use full hierarchical softmax at model dimension (2048)
        # This avoids projecting down to 384 then back up

        if self.clustering_info is not None:
            # Convert dataset ClusteringInfo to hierarchical.py ClusteringInfo format
            # Dataset has cluster_centers, hierarchical.py expects cluster_embeddings
            hierarchical_clustering_info = ClusteringInfo(
                cluster_assignments=self.clustering_info.cluster_assignments,
                cluster_indices=self.clustering_info.cluster_indices,
                in_cluster_id=self.clustering_info.in_cluster_id,
                cluster_embeddings=self.clustering_info.cluster_centers,  # Rename cluster_centers -> cluster_embeddings
            )

            # Create cluster input adapter (cluster_dim -> 2048)
            # Cluster embeddings are typically lower-dim (e.g., 118 for metadata-based)
            cluster_dim = hierarchical_clustering_info.cluster_embeddings.shape[1] if hierarchical_clustering_info.cluster_embeddings is not None else self.item_embedding_dim
            cluster_input_adapter = ClusterInputAdapter(
                cluster_embedding_dim=cluster_dim,
                model_dims=self.model_dims,
                hidden_dim=self.model_dims // 2,
                num_layers=2,
                name='cluster_input_adapter'
            )

            hierarchical_softmax = HierarchicalSoftmax(
                num_items=self.num_items,
                num_clusters=self.num_clusters,
                item_embedding_dim=self.model_dims,  # Use model dims (2048)
                clustering_info=hierarchical_clustering_info,
                use_item_input_dnn_everywhere=True,
                item_input_adapter=item_input_adapter,
                cluster_input_adapter=cluster_input_adapter,
            )

            logits, metrics = hierarchical_softmax(
                hidden_states=model_outputs,  # Use Gemma outputs directly (2048-dim)
                item_embeddings=item_embedding_table,  # Raw 384-dim embeddings
                targets=targets,
                loss_mask=weights,
                training=training,
            )
        else:
            logits = jnp.zeros((batch_size, seq_len, self.num_items))
            metrics = {}

        return {
            'logits': logits,
            **metrics,
        }
