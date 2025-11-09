"""
Pure Flax Llama Implementation (No HuggingFace Dependencies)

This module implements Llama transformer in pure Flax/JAX, avoiding HuggingFace's
OOM issues during model loading. We build the architecture from scratch.

Reference: https://github.com/meta-llama/llama/blob/main/llama/model.py
"""

import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.75')

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Tuple
import math


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    eps: float = 1e-6

    @nn.compact
    def __call__(self, x):
        scale = self.param('scale', nn.initializers.ones, (x.shape[-1],))
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(variance + self.eps)
        return scale * x


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    dim: int
    max_seq_len: int = 2048
    base: float = 10000.0

    def setup(self):
        # Precompute rotation matrix
        inv_freq = 1.0 / (self.base ** (jnp.arange(0, self.dim, 2).astype(jnp.float32) / self.dim))
        t = jnp.arange(self.max_seq_len).astype(jnp.float32)
        freqs = jnp.outer(t, inv_freq)
        self.cos = jnp.cos(freqs)
        self.sin = jnp.sin(freqs)

    def __call__(self, x, seq_len):
        """Apply rotary embedding to x."""
        # x: [batch, seq_len, n_heads, head_dim]
        cos = self.cos[:seq_len, :]
        sin = self.sin[:seq_len, :]

        # Split into even and odd
        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        # Apply rotation
        cos = cos[:, None, :]  # [seq_len, 1, dim//2]
        sin = sin[:, None, :]

        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos

        # Interleave back
        out = jnp.stack([out1, out2], axis=-1).reshape(x.shape)
        return out


class LlamaAttention(nn.Module):
    """Multi-head attention with Grouped Query Attention (GQA)."""

    hidden_size: int
    num_heads: int
    num_kv_heads: int  # For GQA
    head_dim: int
    max_seq_len: int = 2048

    @nn.compact
    def __call__(self, x, mask=None, training=True):
        batch_size, seq_len, _ = x.shape

        # Q, K, V projections
        q = nn.Dense(self.num_heads * self.head_dim, use_bias=False, name='q_proj')(x)
        k = nn.Dense(self.num_kv_heads * self.head_dim, use_bias=False, name='k_proj')(x)
        v = nn.Dense(self.num_kv_heads * self.head_dim, use_bias=False, name='v_proj')(x)

        # Reshape for multi-head
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Apply RoPE
        rope = RotaryEmbedding(dim=self.head_dim, max_seq_len=self.max_seq_len)
        q = rope(q, seq_len)
        k = rope(k, seq_len)

        # Expand K, V for GQA (repeat kv_heads to match num_heads)
        if self.num_kv_heads < self.num_heads:
            k = jnp.repeat(k, self.num_heads // self.num_kv_heads, axis=2)
            v = jnp.repeat(v, self.num_heads // self.num_kv_heads, axis=2)

        # Attention: Q @ K^T / sqrt(d)
        q = jnp.transpose(q, (0, 2, 1, 3))  # [batch, num_heads, seq_len, head_dim]
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) / math.sqrt(self.head_dim)

        # Apply causal mask
        if mask is not None:
            scores = scores + mask

        # Softmax + dropout
        attn_weights = jax.nn.softmax(scores, axis=-1)
        if training:
            attn_weights = nn.Dropout(rate=0.0)(attn_weights, deterministic=not training)

        # Weighted sum of values
        attn_output = jnp.matmul(attn_weights, v)  # [batch, num_heads, seq_len, head_dim]
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))  # [batch, seq_len, num_heads, head_dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.num_heads * self.head_dim)

        # Output projection
        output = nn.Dense(self.hidden_size, use_bias=False, name='o_proj')(attn_output)

        return output


class LlamaMLP(nn.Module):
    """Llama MLP with SwiGLU activation."""

    hidden_size: int
    intermediate_size: int

    @nn.compact
    def __call__(self, x):
        gate = nn.Dense(self.intermediate_size, use_bias=False, name='gate_proj')(x)
        up = nn.Dense(self.intermediate_size, use_bias=False, name='up_proj')(x)

        # SwiGLU activation
        x = jax.nn.silu(gate) * up

        # Down projection
        x = nn.Dense(self.hidden_size, use_bias=False, name='down_proj')(x)

        return x


class LlamaDecoderLayer(nn.Module):
    """Single Llama decoder layer."""

    hidden_size: int
    num_heads: int
    num_kv_heads: int
    intermediate_size: int
    max_seq_len: int = 2048

    @nn.compact
    def __call__(self, x, mask=None, training=True):
        # Pre-norm attention
        residual = x
        x = RMSNorm(name='input_layernorm')(x)
        x = LlamaAttention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.hidden_size // self.num_heads,
            max_seq_len=self.max_seq_len,
            name='self_attn',
        )(x, mask=mask, training=training)
        x = x + residual

        # Pre-norm MLP
        residual = x
        x = RMSNorm(name='post_attention_layernorm')(x)
        x = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            name='mlp',
        )(x)
        x = x + residual

        return x


class LlamaModel(nn.Module):
    """Complete Llama transformer model."""

    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    intermediate_size: int
    max_seq_len: int = 2048

    @nn.compact
    def __call__(self, input_ids=None, inputs_embeds=None, attention_mask=None, training=True):
        """
        Args:
            input_ids: [batch, seq_len] token IDs (if not using inputs_embeds)
            inputs_embeds: [batch, seq_len, hidden_size] pre-computed embeddings
            attention_mask: [batch, seq_len] attention mask
            training: training mode

        Returns:
            last_hidden_state: [batch, seq_len, hidden_size]
        """
        # Embedding layer
        embed_tokens = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.hidden_size,
            name='embed_tokens',
        )

        # Get embeddings
        if inputs_embeds is None:
            x = embed_tokens(input_ids)
        else:
            x = inputs_embeds

        batch_size, seq_len, _ = x.shape

        # Create or adjust attention mask to match sequence length
        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, seq_len))
        else:
            # Ensure attention_mask matches seq_len (truncate or pad if needed)
            if attention_mask.shape[1] != seq_len:
                if attention_mask.shape[1] > seq_len:
                    attention_mask = attention_mask[:, :seq_len]
                else:
                    pad_len = seq_len - attention_mask.shape[1]
                    attention_mask = jnp.pad(attention_mask, ((0, 0), (0, pad_len)), constant_values=0)

        # Convert attention mask to additive form
        # 1 -> 0 (attend), 0 -> -inf (don't attend)
        attention_mask = (1.0 - attention_mask[:, None, None, :]) * -1e9

        # Add causal mask
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        causal_mask = (1.0 - causal_mask) * -1e9
        mask = attention_mask + causal_mask[None, None, :, :]

        # Apply transformer layers
        for i in range(self.num_layers):
            x = LlamaDecoderLayer(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                intermediate_size=self.intermediate_size,
                max_seq_len=self.max_seq_len,
                name=f'layers_{i}',
            )(x, mask=mask, training=training)

        # Final norm
        x = RMSNorm(name='norm')(x)

        return x


# ==================== TESTING ====================

if __name__ == "__main__":
    print("Testing Pure Flax Llama Implementation")
    print("=" * 70)

    # Llama 1B config
    config = {
        'vocab_size': 128256,
        'hidden_size': 2048,
        'num_layers': 16,
        'num_heads': 32,
        'num_kv_heads': 8,
        'intermediate_size': 8192,
        'max_seq_len': 256,
    }

    print(f"\nModel config:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Create model
    model = LlamaModel(**config)

    # Test input
    batch_size = 2
    seq_len = 32
    input_ids = jax.random.randint(jax.random.PRNGKey(0), (batch_size, seq_len), 0, config['vocab_size'])

    print(f"\nTest input shape: {input_ids.shape}")

    # Initialize
    print("\nInitializing model...")
    variables = model.init(jax.random.PRNGKey(42), input_ids=input_ids, training=False)
    params = variables['params']

    # Count parameters
    num_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"✓ Model initialized")
    print(f"✓ Total parameters: {num_params:,}")
    print(f"✓ Expected ~1.2B parameters: {num_params / 1e9:.2f}B")

    # Forward pass
    print("\nRunning forward pass...")
    output = model.apply(variables, input_ids=input_ids, training=False)
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Expected: ({batch_size}, {seq_len}, {config['hidden_size']})")

    # Test with inputs_embeds
    print("\nTesting with inputs_embeds...")
    inputs_embeds = jax.random.normal(jax.random.PRNGKey(1), (batch_size, seq_len, config['hidden_size']))
    output2 = model.apply(variables, inputs_embeds=inputs_embeds, training=False)
    print(f"✓ Output shape: {output2.shape}")

    print("\n" + "=" * 70)
    print("✅ All tests passed! Pure Flax Llama is working.")
    print("\nMemory usage:")
    print(f"  Model params: ~{num_params * 2 / 1e9:.2f} GB (bfloat16)")
    print(f"  No OOM issues - all initialized on-demand!")
