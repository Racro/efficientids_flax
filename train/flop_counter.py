"""
FLOP Counter for EfficientIDS Models

Calculates FLOPs for transformer-based recommendation models and computes
Model FLOPs Utilization (MFU) to measure efficiency.

MFU = (Achieved FLOP/s) / (Theoretical Peak FLOP/s)

Usage:
    counter = FLOPCounter(model_config)
    flops_per_batch = counter.estimate_flops(batch_size, seq_len)
    mfu = counter.compute_mfu(tokens_per_sec, batch_size, seq_len)
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


# GPU/TPU theoretical peak FLOP/s (in TFLOP/s for FP16/BF16)
GPU_PEAK_FLOPS = {
    # GPUs
    'A100': 312.0,  # TFLOP/s for FP16/BF16
    'A100-80GB': 312.0,
    'H100': 989.0,  # TFLOP/s for FP16/BF16
    'H100-80GB': 989.0,
    'V100': 125.0,  # TFLOP/s for FP16
    'T4': 65.0,  # TFLOP/s for FP16
    'L4': 121.0,  # TFLOP/s for FP16/BF16
    # TPUs
    'TPU-v4': 275.0,  # TFLOP/s per chip (BF16)
    'TPU-v5e': 197.0,  # TFLOP/s per chip (BF16)
    'TPU-v6e': 197.0,  # TFLOP/s per chip (BF16) - same as v5e
    'TPU v6e': 197.0,  # Alternative naming
    'TPU v6 lite': 197.0,  # TPU v6e lite (actual device_kind string from JAX)
}


def detect_device_flops() -> float:
    """
    Detect the theoretical peak FLOP/s of the current device.

    Returns:
        Peak FLOP/s in TFLOP/s, or 0 if unknown
    """
    try:
        device = jax.devices()[0]
        device_kind = device.device_kind

        logger.info(f"Detecting device: {device} (kind: {device_kind})")

        # Check known GPU/TPU types
        for device_name, flops in GPU_PEAK_FLOPS.items():
            if device_name.lower() in device_kind.lower():
                logger.info(f"✓ Matched device: {device_kind} → {flops} TFLOP/s (BF16)")
                return flops

        # Unknown device - log all available info for debugging
        logger.warning(f"⚠️  Unknown device: {device_kind}")
        logger.warning(f"   Device object: {device}")
        logger.warning(f"   All devices: {jax.devices()}")
        logger.warning(f"   Available in GPU_PEAK_FLOPS: {list(GPU_PEAK_FLOPS.keys())}")
        logger.warning(f"   Cannot estimate peak FLOP/s - will return 0")
        return 0.0

    except Exception as e:
        logger.warning(f"Failed to detect device: {e}")
        import traceback
        logger.warning(traceback.format_exc())
        return 0.0


class FLOPCounter:
    """
    FLOP counter for transformer-based models.

    Estimates FLOPs per forward/backward pass and computes MFU.

    Args:
        num_params: Total model parameters
        model_dims: Hidden dimension size
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        vocab_size: Vocabulary size (for output projection)
        trainable_params: Number of trainable parameters (defaults to num_params if None)
        peak_flops: Theoretical peak FLOP/s in TFLOP/s (auto-detected if None)
        use_mixed_precision: Whether using FP16/BF16 (affects peak FLOP/s)
    """

    def __init__(
        self,
        num_params: int,
        model_dims: int,
        num_layers: int,
        num_heads: int,
        vocab_size: int,
        trainable_params: Optional[int] = None,
        peak_flops: Optional[float] = None,
        use_mixed_precision: bool = True,
    ):
        self.num_params = num_params
        self.trainable_params = trainable_params if trainable_params is not None else num_params
        self.model_dims = model_dims
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.use_mixed_precision = use_mixed_precision

        # Auto-detect peak FLOP/s if not provided
        if peak_flops is None:
            self.peak_flops = detect_device_flops()
        else:
            self.peak_flops = peak_flops

        # Convert TFLOP/s to FLOP/s
        self.peak_flops_absolute = self.peak_flops * 1e12

        logger.info(f"FLOPCounter initialized:")
        logger.info(f"  Total params: {num_params:,}")
        if self.trainable_params != num_params:
            frozen_pct = ((num_params - self.trainable_params) / num_params) * 100
            logger.info(f"  Trainable params: {self.trainable_params:,} ({100-frozen_pct:.1f}%)")
            logger.info(f"  Frozen params: {num_params - self.trainable_params:,} ({frozen_pct:.1f}%)")
        else:
            logger.info(f"  Trainable params: {self.trainable_params:,} (100%)")
        logger.info(f"  Hidden dims: {model_dims}")
        logger.info(f"  Layers: {num_layers}")
        logger.info(f"  Vocab size: {vocab_size}")
        logger.info(f"  Peak FLOP/s: {self.peak_flops:.1f} TFLOP/s")

    def estimate_flops_per_token(self) -> int:
        """
        Estimate FLOPs per token (forward + backward pass).

        Uses the standard formula:
        FLOPs ≈ 6 * trainable_params (for forward + backward)

        For transformer specifically:
        - Forward: 2 * trainable_params (matrix multiplications)
        - Backward: 4 * trainable_params (gradient computation)

        NOTE: Uses trainable_params (not total params) because frozen weights
        don't require backward pass computation.

        Returns:
            FLOPs per token (forward + backward)
        """
        # Rough estimate: 6 * trainable params per token
        # This is a common approximation for transformers
        # IMPORTANT: Use trainable_params for fine-tuning with frozen layers!
        flops_per_token = 6 * self.trainable_params

        return flops_per_token

    def estimate_flops_per_batch(
        self,
        batch_size: int,
        seq_len: int,
        include_backward: bool = True,
    ) -> int:
        """
        Estimate FLOPs for a single batch.

        Args:
            batch_size: Batch size
            seq_len: Sequence length
            include_backward: Include backward pass (default: True for training)

        Returns:
            Total FLOPs for the batch
        """
        total_tokens = batch_size * seq_len

        if include_backward:
            # Training: forward + backward
            flops_per_token = self.estimate_flops_per_token()
        else:
            # Inference: forward only (roughly 2 * params)
            flops_per_token = 2 * self.num_params

        total_flops = flops_per_token * total_tokens

        return total_flops

    def estimate_detailed_flops(
        self,
        batch_size: int,
        seq_len: int,
    ) -> Dict[str, int]:
        """
        Detailed FLOP breakdown for transformer components.

        Args:
            batch_size: Batch size
            seq_len: Sequence length

        Returns:
            Dictionary with FLOP breakdown
        """
        B = batch_size
        L = seq_len
        d = self.model_dims
        V = self.vocab_size
        n_layers = self.num_layers

        # FLOPs per component (forward pass only)
        flops = {}

        # 1. Attention (per layer): O(B * L^2 * d + B * L * d^2)
        # QKV projection: 3 * (B * L * d * d) = 3BLd^2
        # Attention scores: B * num_heads * L * L * (d/num_heads) = BL^2d
        # Attention output: B * num_heads * L * L * (d/num_heads) = BL^2d
        # Output projection: B * L * d * d = BLd^2
        attn_flops = n_layers * (3 * B * L * d * d + 2 * B * L * L * d + B * L * d * d)
        flops['attention'] = attn_flops

        # 2. FFN (per layer): O(B * L * d * ffn_dim)
        # Typically ffn_dim = 4 * d
        ffn_dim = 4 * d
        # Two linear layers: B * L * d * ffn_dim + B * L * ffn_dim * d
        ffn_flops = n_layers * (2 * B * L * d * ffn_dim)
        flops['ffn'] = ffn_flops

        # 3. Embeddings: B * L * d * V (if using output projection)
        # For hierarchical softmax, this is much smaller
        # We'll use a simplified estimate
        embedding_flops = B * L * d * min(V, 10000)  # Cap for hierarchical
        flops['embeddings'] = embedding_flops

        # 4. Layer norms (negligible compared to matmuls)
        # ~2 * n_layers * B * L * d (mean + variance)
        layernorm_flops = 2 * n_layers * B * L * d
        flops['layernorm'] = layernorm_flops

        # Total forward pass
        forward_total = sum(flops.values())
        flops['forward_total'] = forward_total

        # Backward pass: ~2x forward (gradient computation)
        backward_total = 2 * forward_total
        flops['backward_total'] = backward_total

        # Total (forward + backward)
        flops['total'] = forward_total + backward_total

        return flops

    def compute_mfu(
        self,
        tokens_per_sec: float,
        batch_size: int = 1,
        seq_len: int = 1,
    ) -> float:
        """
        Compute Model FLOPs Utilization (MFU).

        MFU = (Achieved FLOP/s) / (Theoretical Peak FLOP/s)

        Args:
            tokens_per_sec: Measured tokens per second
            batch_size: Batch size used (for validation)
            seq_len: Sequence length (for validation)

        Returns:
            MFU as a percentage (0-100)
        """
        if self.peak_flops_absolute == 0:
            logger.warning("Peak FLOP/s not set. Cannot compute MFU.")
            return 0.0

        # FLOPs per token (forward + backward)
        flops_per_token = self.estimate_flops_per_token()

        # Achieved FLOP/s
        achieved_flops = flops_per_token * tokens_per_sec

        # MFU (as percentage)
        mfu = (achieved_flops / self.peak_flops_absolute) * 100.0

        return mfu

    def compute_achieved_flops(
        self,
        tokens_per_sec: float,
    ) -> float:
        """
        Compute achieved FLOP/s in TFLOP/s.

        Args:
            tokens_per_sec: Measured tokens per second

        Returns:
            Achieved FLOP/s in TFLOP/s
        """
        flops_per_token = self.estimate_flops_per_token()
        achieved_flops = flops_per_token * tokens_per_sec
        achieved_tflops = achieved_flops / 1e12

        return achieved_tflops

    def print_summary(
        self,
        batch_size: int,
        seq_len: int,
        tokens_per_sec: float,
    ):
        """
        Print a detailed summary of FLOP statistics.

        Args:
            batch_size: Batch size
            seq_len: Sequence length
            tokens_per_sec: Measured tokens per second
        """
        print("\n" + "=" * 80)
        print("FLOP Statistics Summary")
        print("=" * 80)

        # Model info
        print(f"Model Configuration:")
        print(f"  Parameters: {self.num_params:,}")
        print(f"  Hidden dims: {self.model_dims}")
        print(f"  Layers: {self.num_layers}")
        print(f"  Vocabulary: {self.vocab_size:,}")

        # FLOP estimates
        flops_per_token = self.estimate_flops_per_token()
        flops_per_batch = self.estimate_flops_per_batch(batch_size, seq_len)

        print(f"\nFLOP Estimates:")
        print(f"  FLOPs per token: {flops_per_token:,}")
        print(f"  FLOPs per batch (B={batch_size}, L={seq_len}): {flops_per_batch:,}")

        # Detailed breakdown
        detailed = self.estimate_detailed_flops(batch_size, seq_len)
        print(f"\nDetailed Breakdown (per batch):")
        print(f"  Attention: {detailed['attention']:,} ({detailed['attention']/detailed['total']*100:.1f}%)")
        print(f"  FFN: {detailed['ffn']:,} ({detailed['ffn']/detailed['total']*100:.1f}%)")
        print(f"  Embeddings: {detailed['embeddings']:,} ({detailed['embeddings']/detailed['total']*100:.1f}%)")
        print(f"  Forward total: {detailed['forward_total']:,}")
        print(f"  Backward total: {detailed['backward_total']:,}")
        print(f"  Total: {detailed['total']:,}")

        # Performance metrics
        achieved_tflops = self.compute_achieved_flops(tokens_per_sec)
        mfu = self.compute_mfu(tokens_per_sec, batch_size, seq_len)

        print(f"\nPerformance Metrics:")
        print(f"  Throughput: {tokens_per_sec:.2f} tokens/sec")
        print(f"  Achieved: {achieved_tflops:.2f} TFLOP/s")
        print(f"  Peak (device): {self.peak_flops:.2f} TFLOP/s")
        print(f"  MFU: {mfu:.2f}%")

        # Interpretation
        print(f"\nInterpretation:")
        if mfu < 20:
            print(f"  ⚠️  Very low MFU - likely bottlenecked by data loading or overhead")
        elif mfu < 40:
            print(f"  ⚠️  Low MFU - room for optimization (kernel fusion, batch size, etc.)")
        elif mfu < 60:
            print(f"  ✓  Decent MFU - typical for many frameworks")
        else:
            print(f"  ✓✓ Excellent MFU - well optimized!")

        print("=" * 80 + "\n")


def estimate_model_flops(
    params: Any,
    batch: Dict[str, jnp.ndarray],
    apply_fn: Any,
) -> int:
    """
    Estimate FLOPs for a model using JAX's cost analysis.

    This uses JAX's built-in FLOP counter for more accurate estimates.

    Args:
        params: Model parameters
        batch: Sample input batch
        apply_fn: Model's apply function

    Returns:
        Estimated FLOPs for forward pass
    """
    try:
        # Create a function that wraps the model call
        def model_fn(params, batch):
            return apply_fn({'params': params}, **batch, training=True)

        # Lower to XLA and get cost analysis
        lowered = jax.jit(model_fn).lower(params, batch)
        compiled = lowered.compile()
        cost_analysis = compiled.cost_analysis()

        if cost_analysis and 'flops' in cost_analysis:
            flops = cost_analysis['flops']
            logger.info(f"JAX cost analysis: {flops:,} FLOPs")
            return int(flops)
        else:
            logger.warning("JAX cost analysis did not return FLOP count")
            return 0

    except Exception as e:
        logger.warning(f"Failed to get JAX cost analysis: {e}")
        return 0


def create_flop_counter_from_model(
    model: Any,
    params: Any,
    batch_size: int = 16,
    seq_len: int = 128,
    peak_flops: Optional[float] = None,
) -> FLOPCounter:
    """
    Create a FLOPCounter from a Flax model instance.

    Automatically extracts model configuration and detects frozen parameters.

    Args:
        model: Flax model instance
        params: Model parameters
        batch_size: Batch size for FLOP estimation
        seq_len: Sequence length
        peak_flops: Optional peak FLOP/s (auto-detected if None)

    Returns:
        Configured FLOPCounter
    """
    # Count total parameters
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(params))

    # Detect trainable parameters (for models with frozen weights)
    trainable_params = None

    # Try multiple ways to detect frozen model
    freeze_lm = getattr(model, 'freeze_lm', False)
    freeze_gemma = getattr(model, 'freeze_gemma', False)
    freeze_llama = getattr(model, 'freeze_llama', False)

    # Debug: Log what we found
    logger.info(f"Freeze detection: freeze_lm={freeze_lm}, freeze_gemma={freeze_gemma}, freeze_llama={freeze_llama}")

    # Also check if model class has these as defaults
    if not (freeze_lm or freeze_gemma or freeze_llama):
        model_class = type(model)
        if hasattr(model_class, '__dataclass_fields__'):
            # Flax dataclass module
            fields = model_class.__dataclass_fields__
            freeze_gemma = fields.get('freeze_gemma', type('', (), {'default': False})).default
            freeze_llama = fields.get('freeze_llama', type('', (), {'default': False})).default
            freeze_lm = fields.get('freeze_lm', type('', (), {'default': False})).default
            logger.info(f"From dataclass defaults: freeze_lm={freeze_lm}, freeze_gemma={freeze_gemma}, freeze_llama={freeze_llama}")

    if freeze_lm or freeze_gemma or freeze_llama:
        # Model has frozen weights - count only trainable params
        # For EfficientIDS, this is typically:
        # - Item embeddings
        # - Cluster embeddings
        # - Item input/output adapters
        # - Hierarchical softmax layers
        try:
            trainable_count = 0
            frozen_count = 0
            param_tree = jax.tree_util.tree_map_with_path(
                lambda path, x: x.size,
                params
            )

            # Debug: Log first few param paths to see structure
            logger.info("Sample param paths (first 10):")
            for i, (key_path, size) in enumerate(jax.tree_util.tree_leaves_with_path(param_tree)):
                if i < 10:
                    path_str = '/'.join(str(k) for k in key_path)
                    logger.info(f"  {path_str}: {size:,}")

            # Count params NOT in gemma/llama/lm submodules
            frozen_patterns = ['gemma_transformer', 'llama', 'transformer', 'lm_tpl']
            for key_path, size in jax.tree_util.tree_leaves_with_path(param_tree):
                path_str = '/'.join(str(k) for k in key_path)
                # Skip frozen transformer params
                is_frozen = any(frozen_name in path_str.lower() for frozen_name in frozen_patterns)
                if is_frozen:
                    frozen_count += size
                else:
                    trainable_count += size

            logger.info(f"Param counting result: trainable={trainable_count:,}, frozen={frozen_count:,}, total={num_params:,}")

            if trainable_count > 0 and trainable_count < num_params:
                trainable_params = trainable_count
                logger.info(f"✓ Detected frozen model: {trainable_params:,} trainable / {num_params:,} total params")
            else:
                logger.warning(f"⚠️  Failed to detect frozen params (trainable={trainable_count}, total={num_params})")
                logger.warning("    Using total params for FLOP estimation (may overestimate)")
        except Exception as e:
            logger.warning(f"Failed to count trainable params: {e}")
            logger.warning("Using total params for FLOP estimation (may overestimate)")
            import traceback
            logger.warning(traceback.format_exc())

    # Try to extract model config
    try:
        if hasattr(model, 'model_dims'):
            model_dims = model.model_dims
        else:
            # Default guess
            model_dims = 2048
            logger.warning(f"Could not detect model_dims, using default: {model_dims}")

        if hasattr(model, 'num_items'):
            vocab_size = model.num_items
        else:
            vocab_size = 50000
            logger.warning(f"Could not detect vocab_size, using default: {vocab_size}")

        # Estimate number of layers (heuristic)
        num_layers = 24  # Default for ~2B models
        num_heads = 16

        logger.info(f"Auto-detected model config:")
        logger.info(f"  Total params: {num_params:,}")
        if trainable_params:
            logger.info(f"  Trainable params: {trainable_params:,} ({trainable_params/num_params*100:.1f}%)")
        logger.info(f"  Model dims: {model_dims}")
        logger.info(f"  Vocab: {vocab_size:,}")

    except Exception as e:
        logger.warning(f"Failed to auto-detect model config: {e}")
        model_dims = 2048
        vocab_size = 50000
        num_layers = 24
        num_heads = 16

    return FLOPCounter(
        num_params=num_params,
        trainable_params=trainable_params,
        model_dims=model_dims,
        num_layers=num_layers,
        num_heads=num_heads,
        vocab_size=vocab_size,
        peak_flops=peak_flops,
    )
