"""
Item Embeddings and Adapters - Core Building Blocks

Pure JAX/Flax implementation of item embeddings with MLP adapters.
This is the foundation layer with no external dependencies.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Callable


class ItemEmbedding(nn.Module):
    """
    Simple item embedding table with optional initialization from numpy array.

    Args:
        num_items: Vocabulary size (e.g., 3261 for MovieLens-1M)
        embedding_dim: Embedding dimension (e.g., 384)
        initializer: Optional custom initializer (for pretrained embeddings)
    """
    num_items: int
    embedding_dim: int
    initializer: Optional[Callable] = None

    @nn.compact
    def __call__(self, item_ids: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            item_ids: [batch, seq_len] integer item IDs

        Returns:
            embeddings: [batch, seq_len, embedding_dim] item embeddings
        """
        # Use custom initializer if provided (e.g., from .npy file)
        # Otherwise use default Xavier initialization
        if self.initializer is None:
            initializer = nn.initializers.xavier_uniform()
        else:
            initializer = self.initializer

        embeddings = nn.Embed(
            num_embeddings=self.num_items,
            features=self.embedding_dim,
            embedding_init=initializer,
            name='item_embeddings'
        )(item_ids)

        return embeddings


class MLPAdapter(nn.Module):
    """
    MLP adapter for projecting item embeddings to/from transformer space.

    This matches the item_input_dnn and item_output_dnn from original code.

    Args:
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        num_layers: Number of MLP layers (1 or 2)
        activation: Activation function (default: gelu)
    """
    hidden_dim: int
    output_dim: int
    num_layers: int = 2
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: [batch, seq_len, input_dim] input tensor

        Returns:
            output: [batch, seq_len, output_dim] projected tensor
        """
        for i in range(self.num_layers):
            x = nn.Dense(
                features=self.hidden_dim if i < self.num_layers - 1 else self.output_dim,
                name=f'layer_{i}'
            )(x)

            # Apply activation for all layers (including final, as in original)
            x = self.activation(x)

        return x


class ItemInputAdapter(nn.Module):
    """
    Projects item embeddings → transformer space.

    item_embedding_dim → hidden_dim → model_dims

    Example: 384 → 1536 → 896 (for Qwen 0.6B)
    """
    item_embedding_dim: int
    model_dims: int
    hidden_dim: int = 1536
    num_layers: int = 2

    @nn.compact
    def __call__(self, item_embeddings: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            item_embeddings: [batch, seq_len, item_embedding_dim]

        Returns:
            transformer_embeddings: [batch, seq_len, model_dims]
        """
        # MLP projection
        adapter = MLPAdapter(
            hidden_dim=self.hidden_dim,
            output_dim=self.model_dims,
            num_layers=self.num_layers,
            name='item_input_adapter'
        )

        return adapter(item_embeddings)


class ItemOutputAdapter(nn.Module):
    """
    Projects transformer space → item embedding space for prediction.

    model_dims → hidden_dim → item_embedding_dim

    Example: 896 → 1536 → 384 (for Qwen 0.6B)
    """
    model_dims: int
    item_embedding_dim: int
    hidden_dim: int = 1536
    num_layers: int = 2

    @nn.compact
    def __call__(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            hidden_states: [batch, seq_len, model_dims] transformer outputs

        Returns:
            item_representations: [batch, seq_len, item_embedding_dim]
        """
        # MLP projection
        adapter = MLPAdapter(
            hidden_dim=self.hidden_dim,
            output_dim=self.item_embedding_dim,
            num_layers=self.num_layers,
            name='item_output_adapter'
        )

        return adapter(hidden_states)


class ClusterInputAdapter(nn.Module):
    """
    Projects cluster embeddings → transformer space.

    cluster_embedding_dim → hidden_dim → model_dims

    Example: 118 → 1024 → 2048 (for Gemma 2B with metadata clusters)
    """
    cluster_embedding_dim: int
    model_dims: int
    hidden_dim: int = 1024
    num_layers: int = 2

    @nn.compact
    def __call__(self, cluster_embeddings: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            cluster_embeddings: [num_clusters, cluster_embedding_dim]

        Returns:
            transformer_embeddings: [num_clusters, model_dims]
        """
        # MLP projection
        adapter = MLPAdapter(
            hidden_dim=self.hidden_dim,
            output_dim=self.model_dims,
            num_layers=self.num_layers,
            name='cluster_input_adapter'
        )

        return adapter(cluster_embeddings)


def create_embedding_initializer(embedding_array: jnp.ndarray) -> Callable:
    """
    Create a Flax initializer from a pretrained embedding array.

    This allows loading embeddings from .npy files (random/metadata/WALS).

    Args:
        embedding_array: [num_items, embedding_dim] pretrained embeddings

    Returns:
        initializer: Flax initializer function

    Example:
        >>> import numpy as np
        >>> pretrained = np.load('item_embeddings_metadata.npy')
        >>> initializer = create_embedding_initializer(pretrained)
        >>> model = ItemEmbedding(num_items=3261, embedding_dim=384,
        ...                        initializer=initializer)
    """
    def init_fn(key, shape, dtype=jnp.float32):
        # Ignore key and shape, return the pretrained embeddings
        assert shape == embedding_array.shape, (
            f"Shape mismatch: expected {shape}, got {embedding_array.shape}"
        )
        return jnp.array(embedding_array, dtype=dtype)

    return init_fn


# ==================== TESTING ====================

if __name__ == "__main__":
    """Quick test of embedding and adapter modules."""

    print("Testing Item Embeddings and Adapters")
    print("=" * 60)

    # Configuration (MovieLens-1M with Qwen 0.6B)
    num_items = 3261
    item_embedding_dim = 384
    model_dims = 896  # Qwen 0.6B hidden size
    batch_size = 4
    seq_len = 10

    # Test 1: Basic item embedding
    print("\n1. Testing ItemEmbedding...")
    item_emb = ItemEmbedding(num_items=num_items, embedding_dim=item_embedding_dim)

    # Initialize with random key
    key = jax.random.PRNGKey(0)
    item_ids = jax.random.randint(key, (batch_size, seq_len), 0, num_items)

    params = item_emb.init(key, item_ids)
    embeddings = item_emb.apply(params, item_ids)

    print(f"   Input shape: {item_ids.shape}")
    print(f"   Output shape: {embeddings.shape}")
    print(f"   Expected: ({batch_size}, {seq_len}, {item_embedding_dim})")
    assert embeddings.shape == (batch_size, seq_len, item_embedding_dim)
    print("   ✓ Passed!")

    # Test 2: Item input adapter
    print("\n2. Testing ItemInputAdapter...")
    input_adapter = ItemInputAdapter(
        item_embedding_dim=item_embedding_dim,
        model_dims=model_dims,
        hidden_dim=1536,
        num_layers=2
    )

    params = input_adapter.init(key, embeddings)
    transformer_emb = input_adapter.apply(params, embeddings)

    print(f"   Input shape: {embeddings.shape}")
    print(f"   Output shape: {transformer_emb.shape}")
    print(f"   Expected: ({batch_size}, {seq_len}, {model_dims})")
    assert transformer_emb.shape == (batch_size, seq_len, model_dims)
    print("   ✓ Passed!")

    # Test 3: Item output adapter
    print("\n3. Testing ItemOutputAdapter...")
    output_adapter = ItemOutputAdapter(
        model_dims=model_dims,
        item_embedding_dim=item_embedding_dim,
        hidden_dim=1536,
        num_layers=2
    )

    # Simulate transformer hidden states
    hidden_states = jax.random.normal(key, (batch_size, seq_len, model_dims))

    params = output_adapter.init(key, hidden_states)
    item_repr = output_adapter.apply(params, hidden_states)

    print(f"   Input shape: {hidden_states.shape}")
    print(f"   Output shape: {item_repr.shape}")
    print(f"   Expected: ({batch_size}, {seq_len}, {item_embedding_dim})")
    assert item_repr.shape == (batch_size, seq_len, item_embedding_dim)
    print("   ✓ Passed!")

    # Test 4: Pretrained embedding loading
    print("\n4. Testing pretrained embedding loading...")
    import numpy as np

    # Create fake pretrained embeddings
    fake_pretrained = np.random.randn(num_items, item_embedding_dim).astype(np.float32)
    initializer = create_embedding_initializer(fake_pretrained)

    item_emb_pretrained = ItemEmbedding(
        num_items=num_items,
        embedding_dim=item_embedding_dim,
        initializer=initializer
    )

    params = item_emb_pretrained.init(key, item_ids)

    # Check that embeddings match pretrained
    loaded_embeddings = params['params']['item_embeddings']['embedding']
    print(f"   Pretrained shape: {fake_pretrained.shape}")
    print(f"   Loaded shape: {loaded_embeddings.shape}")
    assert np.allclose(loaded_embeddings, fake_pretrained, rtol=1e-5)
    print("   ✓ Embeddings loaded correctly!")

    print("\n" + "=" * 60)
    print("✅ All tests passed! Core embeddings are working.")
    print("\nNext steps:")
    print("  1. Create hierarchical softmax (hierarchical.py)")
    print("  2. Create transformer model (models.py)")
    print("  3. Build training loop (train/trainer.py)")
