"""
Llama Model Loader for EfficientIDS Flax

This module provides utilities to load and integrate Llama models from HuggingFace
into the EfficientIDS recommendation system using pure Flax/JAX.

Supported Models:
-----------------
- Llama-3.2-1B (meta-llama/Llama-3.2-1B): 2048d, 16 layers
- Meta-Llama-3-8B (meta-llama/Meta-Llama-3-8B): 4096d, 32 layers

Key Features:
-------------
- Direct HuggingFace integration (no PAXml)
- Flax-based transformer
- Frozen or trainable LM weights
- Item embedding injection into transformer
"""

# CRITICAL: Set JAX memory config BEFORE any imports that use JAX
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.75')

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn
from transformers import FlaxAutoModelForCausalLM, AutoConfig, AutoTokenizer
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== MODEL CONFIGURATIONS ====================

LLAMA_CONFIGS = {
    '1b': {
        'model_name': 'meta-llama/Llama-3.2-1B',
        'hidden_size': 2048,
        'intermediate_size': 8192,
        'num_hidden_layers': 16,
        'num_attention_heads': 32,
        'num_key_value_heads': 8,
        'vocab_size': 128256,
        'max_position_embeddings': 131072,
    },
    '3b': {
        'model_name': 'meta-llama/Llama-3.2-3B',
        'hidden_size': 3072,
        'intermediate_size': 8192,
        'num_hidden_layers': 28,
        'num_attention_heads': 24,
        'num_key_value_heads': 8,
        'vocab_size': 128256,
        'max_position_embeddings': 131072,
    },
    '8b': {
        'model_name': 'meta-llama/Meta-Llama-3-8B',
        'hidden_size': 4096,
        'intermediate_size': 14336,
        'num_hidden_layers': 32,
        'num_attention_heads': 32,
        'num_key_value_heads': 8,
        'vocab_size': 128256,
        'max_position_embeddings': 8192,
    },
}


class LlamaLoader:
    """Utility class for loading Llama models from HuggingFace."""

    def __init__(
        self,
        model_size: str = '1b',
        cache_dir: Optional[str] = None,
        hf_token: Optional[str] = None,
    ):
        """
        Initialize Llama loader.

        Args:
            model_size: Model size ('1b', '3b', '8b')
            cache_dir: Directory to cache model files
            hf_token: HuggingFace authentication token (required for Llama models)
        """
        if model_size not in LLAMA_CONFIGS:
            raise ValueError(f"Unknown model size: {model_size}. Options: {list(LLAMA_CONFIGS.keys())}")

        self.model_size = model_size
        self.config = LLAMA_CONFIGS[model_size]
        self.model_name = self.config['model_name']
        self.cache_dir = cache_dir or f"./model_cache/llama/{model_size}"

        # Get HF token from argument or environment
        self.hf_token = hf_token or os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')

    def authenticate(self) -> bool:
        """Authenticate with HuggingFace."""
        if not self.hf_token:
            logger.error("❌ No HuggingFace token provided!")
            logger.error("💡 To fix this:")
            logger.error("   1. Get token at: https://huggingface.co/settings/tokens")
            logger.error("   2. Accept license at: https://huggingface.co/meta-llama/Llama-3.2-1B")
            logger.error("   3. Set token: export HF_TOKEN=your_token")
            logger.error("   4. Or pass token: LlamaLoader(hf_token='your_token')")
            return False

        try:
            from huggingface_hub import login
            login(token=self.hf_token)
            logger.info("✅ HuggingFace authentication successful!")
            return True
        except Exception as e:
            logger.error(f"❌ HuggingFace authentication failed: {e}")
            return False

    def load_model(
        self,
        dtype: jnp.dtype = jnp.bfloat16,
        load_pretrained_weights: bool = False,
    ) -> Tuple[Any, Any, Any]:
        """
        Load Llama config (and optionally pretrained weights).

        Args:
            dtype: Data type for model parameters (bfloat16 recommended)
            load_pretrained_weights: Whether to load full pretrained weights (memory intensive!)

        Returns:
            (model, params, config): If load_pretrained_weights=True, loads full model.
                                    Otherwise, returns (None, None, config) - config only.
        """
        logger.info(f"📥 Loading Llama {self.model_size.upper()} config from HuggingFace...")
        logger.info(f"   Model: {self.model_name}")
        logger.info(f"   Hidden size: {self.config['hidden_size']}")
        logger.info(f"   Layers: {self.config['num_hidden_layers']}")

        # Authenticate if needed
        if not self.authenticate():
            raise RuntimeError("HuggingFace authentication required for Llama models")

        try:
            # Load config only (lightweight)
            config = AutoConfig.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )

            logger.info("✅ Config loaded successfully!")
            logger.info(f"   Hidden size: {config.hidden_size}")
            logger.info(f"   Layers: {config.num_hidden_layers}")
            logger.info(f"   Vocab size: {config.vocab_size}")

            if not load_pretrained_weights:
                logger.info("   💡 Skipping pretrained weights (will train from scratch)")
                logger.info("   💡 To use pretrained Llama, set load_pretrained_weights=True")
                return None, None, config

            # Only load full model if explicitly requested
            logger.warning("⚠️  Loading full pretrained weights (may cause OOM on some GPUs)...")
            logger.info("   This will download ~2.5GB and use ~16GB GPU memory during load")

            model = FlaxAutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                from_pt=True,  # Convert from PyTorch weights
                dtype=dtype,
                _do_init=True,
            )

            params = model.params
            logger.info("✅ Pretrained weights loaded!")
            logger.info(f"   Parameters: {sum(p.size for p in jax.tree_util.tree_leaves(params)):,}")

            return model, params, config

        except Exception as e:
            logger.error(f"❌ Failed to load: {e}")
            if "Out of memory" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                logger.error("💡 OOM during loading. Try:")
                logger.error("   1. Train from scratch (skip pretrained weights)")
                logger.error("   2. Use smaller model (--size 1b)")
                logger.error("   3. Free up GPU memory (kill other processes)")
            raise

    def load_tokenizer(self):
        """Load the tokenizer for this model."""
        logger.info(f"📥 Loading tokenizer for {self.model_name}...")

        if not self.authenticate():
            raise RuntimeError("HuggingFace authentication required")

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
        )

        logger.info(f"✅ Tokenizer loaded! Vocab size: {len(tokenizer)}")
        return tokenizer


def estimate_model_memory_gb(
    model_size: str,
    batch_size: int = 1,
    seq_len: int = 512,
    dtype_bytes: int = 2,  # bfloat16 = 2 bytes
) -> float:
    """
    Estimate GPU memory requirements for a Llama model.

    Args:
        model_size: Model size ('1b', '3b', '8b')
        batch_size: Batch size for training/inference
        seq_len: Sequence length
        dtype_bytes: Bytes per parameter (2 for bfloat16)

    Returns:
        Estimated memory in GB
    """
    if model_size not in LLAMA_CONFIGS:
        raise ValueError(f"Unknown model size: {model_size}")

    config = LLAMA_CONFIGS[model_size]
    hidden_size = config['hidden_size']
    num_layers = config['num_hidden_layers']
    vocab_size = config['vocab_size']
    intermediate_size = config['intermediate_size']

    # Model parameters
    # Embeddings: vocab * hidden
    embedding_params = vocab_size * hidden_size

    # Each layer: self_attn + mlp + norms
    # self_attn: 4 * (hidden * hidden)  # Q, K, V, O projections
    # mlp: 2 * (hidden * intermediate)  # gate, up, down
    # norms: 2 * hidden  # input norm, post-attn norm
    layer_params = (
        4 * (hidden_size * hidden_size) +  # Attention
        2 * (hidden_size * intermediate_size) +  # MLP
        2 * hidden_size  # Norms
    )

    total_params = embedding_params + num_layers * layer_params

    # Memory breakdown
    # 1. Model weights
    model_memory_gb = (total_params * dtype_bytes) / (1024**3)

    # 2. Activations (batch * seq_len * hidden * layers)
    activation_memory_gb = (batch_size * seq_len * hidden_size * num_layers * 4) / (1024**3)

    # 3. Optimizer states (AdamW = 2x model size for momentum + variance)
    optimizer_memory_gb = model_memory_gb * 2

    # 4. Gradients (same as model)
    gradient_memory_gb = model_memory_gb

    # Total with 20% overhead for misc
    total_memory_gb = (
        model_memory_gb +
        activation_memory_gb +
        optimizer_memory_gb +
        gradient_memory_gb
    ) * 1.2

    logger.info(f"📊 Memory estimate for Llama {model_size.upper()}:")
    logger.info(f"   Model: {model_memory_gb:.2f} GB")
    logger.info(f"   Activations: {activation_memory_gb:.2f} GB")
    logger.info(f"   Optimizer: {optimizer_memory_gb:.2f} GB")
    logger.info(f"   Gradients: {gradient_memory_gb:.2f} GB")
    logger.info(f"   Total: {total_memory_gb:.2f} GB")

    return total_memory_gb


def select_llama_model(available_memory_gb: float) -> str:
    """
    Select best Llama model size based on available GPU memory.

    Args:
        available_memory_gb: Available GPU memory in GB

    Returns:
        Model size string ('1b', '3b', or '8b')
    """
    # Leave 80% headroom for safety
    usable_memory = available_memory_gb * 0.8

    # Check from largest to smallest
    for size in ['8b', '3b', '1b']:
        required_memory = estimate_model_memory_gb(size, batch_size=1, seq_len=256)
        if required_memory <= usable_memory:
            logger.info(f"✅ Selected Llama {size.upper()} for {available_memory_gb}GB GPU")
            return size

    logger.warning(f"⚠️  Even Llama 1B may not fit in {available_memory_gb}GB")
    return '1b'  # Return smallest as fallback


# ==================== TESTING ====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test Llama model loading')
    parser.add_argument('--size', type=str, default='1b', choices=['1b', '3b', '8b'],
                       help='Model size to load')
    parser.add_argument('--token', type=str, help='HuggingFace token')
    parser.add_argument('--estimate-only', action='store_true',
                       help='Only estimate memory, do not load model')
    args = parser.parse_args()

    print("=" * 70)
    print("Llama Model Loader for EfficientIDS")
    print("=" * 70)
    print(f"✓ JAX memory: preallocate={os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE', 'true')}, "
          f"max_fraction={os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.9')}")
    print("=" * 70)

    if args.estimate_only:
        print(f"\n📊 Memory estimates for Llama models:")
        print("-" * 70)
        for size in ['1b', '3b', '8b']:
            memory = estimate_model_memory_gb(size, batch_size=1, seq_len=256)
            print(f"  Llama {size.upper()}: {memory:.2f} GB")

        # Try to detect GPU
        try:
            import jax
            if jax.devices('gpu'):
                gpu_memory = 80  # Assume A100/H100 for now
                print(f"\n🔍 Detected GPU with ~{gpu_memory}GB memory")
                best_model = select_llama_model(gpu_memory)
                print(f"✅ Recommended: Llama {best_model.upper()}")
        except:
            pass
    else:
        # Load model
        loader = LlamaLoader(model_size=args.size, hf_token=args.token)

        try:
            model, params, config = loader.load_model()
            tokenizer = loader.tokenizer()

            print("\n✅ Model loaded successfully!")
            print(f"   Model: {config.name_or_path}")
            print(f"   Hidden size: {config.hidden_size}")
            print(f"   Layers: {config.num_hidden_layers}")
            print(f"   Vocab size: {config.vocab_size}")
            print(f"   Tokenizer vocab: {len(tokenizer)}")

            # Test inference
            print("\n🧪 Testing inference...")
            test_input = "The movie was"
            input_ids = tokenizer(test_input, return_tensors='np')['input_ids']
            input_ids = jnp.array(input_ids)

            outputs = model(input_ids, params=params)
            logits = outputs.logits

            print(f"✅ Inference works! Output shape: {logits.shape}")

        except Exception as e:
            print(f"\n❌ Failed to load model: {e}")
            print("\n💡 Make sure you have:")
            print("   1. Valid HuggingFace token")
            print("   2. Accepted Llama license")
            print("   3. Enough GPU memory")
