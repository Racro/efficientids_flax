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
from .hierarchical_simple import SimpleHierarchicalSoftmax, JaxClusteringInfo
from .embeddings import ItemInputAdapter, ItemOutputAdapter


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
        q = nn.DenseGeneral(
            features=(self.num_heads, self.head_dim),
            name='q_einsum'
        )(x)

        # KV projection: [batch, seq, hidden] -> [batch, seq, num_kv_heads, head_dim]
        kv = nn.DenseGeneral(
            features=(self.num_kv_heads, self.head_dim),
            axis=-1,
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
        gating = nn.DenseGeneral(
            features=self.intermediate_dim,
            name='gating_einsum'
        )(x)

        # SwiGLU: swish activation on gating
        gating = jax.nn.swish(gating)

        # Linear projection: [batch, seq, intermediate] -> [batch, seq, hidden]
        output = nn.Dense(
            features=self.hidden_dim,
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

    Gemma checkpoint structure:
        transformer/layer_X/attn/{q_einsum,kv_einsum,attn_vec_einsum}/w
        transformer/layer_X/mlp/{gating_einsum,linear}/w
        transformer/layer_X/{pre_attention_norm,pre_ffw_norm}/scale
        transformer/final_norm/scale

    Our Flax structure:
        layer_X/attn/{q_einsum,kv_einsum,attn_vec_einsum}/kernel
        layer_X/mlp/{gating_einsum,linear}/kernel
        layer_X/{pre_attention_norm,pre_ffw_norm}/scale
        final_norm/scale

    Args:
        gemma_params: Raw Gemma checkpoint params

    Returns:
        Reshaped params matching Flax model
    """
    transformer_params = gemma_params.get('transformer', gemma_params)
    flax_params = {}

    # Process each layer
    for i in range(18):
        layer_key = f'layer_{i}'
        if layer_key not in transformer_params:
            continue

        layer_params = transformer_params[layer_key]
        flax_layer = {}

        # Attention params
        if 'attn' in layer_params:
            attn_params = layer_params['attn']
            flax_attn = {}

            # Q, KV, output projections (rename 'w' -> 'kernel')
            for param_name in ['q_einsum', 'kv_einsum', 'attn_vec_einsum']:
                if param_name in attn_params and 'w' in attn_params[param_name]:
                    flax_attn[param_name] = {'kernel': attn_params[param_name]['w']}

            flax_layer['attn'] = flax_attn

        # MLP params
        if 'mlp' in layer_params:
            mlp_params = layer_params['mlp']
            flax_mlp = {}

            # Gating and linear projections
            for param_name in ['gating_einsum', 'linear']:
                if param_name in mlp_params and 'w' in mlp_params[param_name]:
                    flax_mlp[param_name] = {'kernel': mlp_params[param_name]['w']}

            flax_layer['mlp'] = flax_mlp

        # RMSNorm params
        for norm_name in ['pre_attention_norm', 'pre_ffw_norm']:
            if norm_name in layer_params and 'scale' in layer_params[norm_name]:
                flax_layer[norm_name] = {'scale': layer_params[norm_name]['scale']}

        flax_params[layer_key] = flax_layer

    # Final norm
    if 'final_norm' in transformer_params and 'scale' in transformer_params['final_norm']:
        flax_params['final_norm'] = {'scale': transformer_params['final_norm']['scale']}

    return flax_params


class GemmaEfficientIDSModel(nn.Module):
    """
    Gemma 2B + EfficientIDS for item recommendation.

    Architecture:
    1. Item embeddings (num_items, 384)
    2. Input adapter (384 -> 2048)
    3. Gemma transformer (18 layers, 2048 hidden)
    4. Output adapter (2048 -> 384)
    5. Hierarchical softmax (cluster + item)

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
        item_input_adapter = ItemInputAdapter(
            item_embedding_dim=self.item_embedding_dim,
            model_dims=self.model_dims,
            hidden_dim=self.item_embedding_dim * 4,
            num_layers=2,
            name='item_input_adapter'
        )

        # Project Gemma outputs to item space (2048 → 384)
        item_output_adapter = ItemOutputAdapter(
            model_dims=self.model_dims,
            item_embedding_dim=self.item_embedding_dim,
            hidden_dim=self.item_embedding_dim * 4,
            num_layers=2,
            name='item_output_adapter'
        )

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

        # Project back to item space
        item_space_hidden = item_output_adapter(model_outputs)  # [batch, seq_len, 384]

        # ==================== HIERARCHICAL SOFTMAX ====================
        if self.clustering_info is not None and training and targets is not None:
            jax_clustering = JaxClusteringInfo.from_numpy_clustering_info(self.clustering_info)

            hierarchical_softmax = SimpleHierarchicalSoftmax(
                num_items=self.num_items,
                num_clusters=self.num_clusters,
                item_embedding_dim=self.item_embedding_dim,
                cluster_assignments=jax_clustering.cluster_assignments,
                cluster_indices=jax_clustering.cluster_indices,
            )

            logits, metrics = hierarchical_softmax(
                hidden_states=item_space_hidden,
                item_embeddings=item_embedding_table,
                targets=targets,
                loss_mask=weights,
                training=True,
            )
        else:
            logits = jnp.zeros((batch_size, seq_len, self.num_items))
            metrics = {}

        return {
            'logits': logits,
            **metrics,
        }
