#!/usr/bin/env python3
"""
End-to-End Training Script for EfficientIDS (Flax Port)

Fully replaces the original PAXml training pipeline with pure JAX/Flax.
Compatible with the same data format and achieves the same metrics.

Usage:
    # Train with Qwen-like config
    python train_efficientids.py --config qwen --max_steps 10000

    # Train with debug config (fast)
    python train_efficientids.py --config debug

    # Custom config
    python train_efficientids.py --data_dir ./data/ml1m_processed/processed \\
        --num_items 3261 --num_clusters 100 --max_steps 10000
"""

import argparse
import logging
from pathlib import Path
import sys

import jax
import jax.numpy as jnp

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from configs.config import (
    get_qwen_config,
    get_llama_config,
    get_gemma_config,
    get_debug_config,
    get_tpu_optimized_config,
    EfficientIDSConfig,
)
from data.dataset import create_dataloaders, ClusteringInfo
from core.models import SimpleEfficientIDSModel
from train.trainer import Trainer
from train.optimizer import create_optimizer, create_learning_rate_schedule

# Gemma imports - we'll load weights manually without gemma package
import orbax.checkpoint as ocp
GEMMA_AVAILABLE = True  # We can always use Gemma with Orbax

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_model_from_config(config: EfficientIDSConfig, clustering_info: ClusteringInfo):
    """
    Create model from configuration.

    Args:
        config: Configuration object
        clustering_info: Clustering information

    Returns:
        Flax model instance
    """
    model = SimpleEfficientIDSModel(
        num_items=config.model.num_items,
        num_clusters=config.model.num_clusters,
        item_embedding_dim=config.model.item_embedding_dim,
        model_dims=config.model.model_dims,
        clustering_info=clustering_info,
        use_correction=config.model.use_correction,
    )

    logger.info("Model created:")
    logger.info(f"  Items: {config.model.num_items}")
    logger.info(f"  Clusters: {config.model.num_clusters}")
    logger.info(f"  Item embedding dim: {config.model.item_embedding_dim}")
    logger.info(f"  Model dims: {config.model.model_dims}")
    logger.info(f"  Hierarchical softmax: {config.model.use_hierarchical_softmax}")

    return model


def create_optimizer_from_config(config: EfficientIDSConfig):
    """
    Create optimizer from configuration.

    Args:
        config: Configuration object

    Returns:
        Optax optimizer
    """
    # Create learning rate schedule
    schedule = create_learning_rate_schedule(
        base_learning_rate=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        total_steps=config.training.max_steps,
        schedule_type=config.training.schedule_type,
    )

    # Create optimizer
    optimizer = create_optimizer(
        learning_rate=schedule,
        optimizer_type=config.training.optimizer_type,
        weight_decay=config.training.weight_decay,
        clip_grad_norm=config.training.clip_grad_norm,
        beta1=config.training.beta1,
        beta2=config.training.beta2,
        eps=config.training.eps,
    )

    logger.info("Optimizer created:")
    logger.info(f"  Type: {config.training.optimizer_type}")
    logger.info(f"  Learning rate: {config.training.learning_rate}")
    logger.info(f"  Schedule: {config.training.schedule_type}")
    logger.info(f"  Weight decay: {config.training.weight_decay}")
    logger.info(f"  Warmup steps: {config.training.warmup_steps}")

    return optimizer


def create_gemma_model(config: EfficientIDSConfig, clustering_info: ClusteringInfo, pretrained_path: str, freeze_gemma: bool):
    """
    Create Gemma-style EfficientIDS model with pretrained weights.

    Args:
        config: Configuration
        clustering_info: Clustering information
        pretrained_path: Path to Gemma checkpoint (e.g., '../2b/')
        freeze_gemma: Freeze Gemma weights

    Returns:
        model: GemmaEfficientIDSModel
        gemma_params: Pretrained Gemma parameters
    """
    from core.gemma_model import GemmaEfficientIDSModel, load_gemma_params, reshape_gemma_params_for_flax

    logger.info("Creating Gemma model with pretrained weights...")
    logger.info(f"  Loading checkpoint from: {pretrained_path}")

    # Load Gemma checkpoint
    try:
        gemma_params = load_gemma_params(pretrained_path)
        logger.info(f"  ✓ Loaded Gemma checkpoint")

        # Debug: Check what we loaded
        if 'transformer' in gemma_params:
            logger.info(f"  Found 'transformer' key in checkpoint")
            transformer_keys = list(gemma_params['transformer'].keys())
            logger.info(f"  Transformer has {len(transformer_keys)} keys")
            logger.info(f"  First 3 keys: {transformer_keys[:3]}")

        # Reshape params for Flax
        gemma_params_flax = reshape_gemma_params_for_flax(gemma_params)
        logger.info(f"  ✓ Reshaped params for Flax")

        # Debug: Check reshaped structure
        if gemma_params_flax:
            logger.info(f"  Reshaped params have {len(gemma_params_flax)} top-level keys")
            logger.info(f"  Keys: {list(gemma_params_flax.keys())[:5]}")
            if 'transformer' in gemma_params_flax:
                transformer = gemma_params_flax['transformer']
                logger.info(f"  Transformer has {len(transformer)} keys")
                logger.info(f"  Transformer keys: {list(transformer.keys())[:3]}")
                if 'layer_0' in transformer:
                    logger.info(f"  Layer 0 keys: {list(transformer['layer_0'].keys())}")
    except Exception as e:
        logger.warning(f"  Failed to load checkpoint: {e}")
        logger.warning(f"  Will initialize from scratch")
        import traceback
        traceback.print_exc()
        gemma_params_flax = None

    # Create model with Gemma dimensions
    model = GemmaEfficientIDSModel(
        num_items=config.model.num_items,
        num_clusters=config.model.num_clusters,
        item_embedding_dim=config.model.item_embedding_dim,
        model_dims=2048,  # Gemma 2B hidden size
        clustering_info=clustering_info,
        freeze_gemma=freeze_gemma,
    )

    logger.info("Model created:")
    logger.info(f"  Items: {config.model.num_items}")
    logger.info(f"  Clusters: {config.model.num_clusters}")
    logger.info(f"  Item embedding dim: {config.model.item_embedding_dim}")
    logger.info(f"  Model dims: 2048 (Gemma 2B)")
    logger.info(f"  Freeze Gemma: {freeze_gemma}")

    return model, gemma_params_flax


def main(args):
    """Main training function."""

    # ==================== 1. LOAD CONFIGURATION ====================
    logger.info("=" * 80)
    logger.info("EfficientIDS Training (Flax Port)")
    logger.info("=" * 80)

    # Get config
    if args.config == 'qwen':
        config = get_qwen_config(
            num_items=args.num_items,
            num_clusters=args.num_clusters,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            max_steps=args.max_steps,
        )
    elif args.config == 'llama':
        config = get_llama_config(
            num_items=args.num_items,
            num_clusters=args.num_clusters,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            max_steps=args.max_steps,
        )
    elif args.config == 'tpu_optimized':
        config = get_tpu_optimized_config(
            num_items=args.num_items,
            num_clusters=args.num_clusters,
            item_embedding_dim=args.item_embedding_dim,
            max_seq_len=args.max_seq_len,
            max_steps=args.max_steps,
        )
    elif args.config == 'gemma':
        config = get_gemma_config(
            num_items=args.num_items,
            num_clusters=args.num_clusters,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            max_steps=args.max_steps,
        )
    elif args.config == 'debug':
        config = get_debug_config()
    else:
        raise ValueError(f"Unknown config: {args.config}")

    # Override with command-line args
    if args.data_dir:
        config.data.data_dir = args.data_dir
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir

    logger.info(f"\nConfiguration: {config.experiment_name}")
    logger.info(f"  Data dir: {config.data.data_dir}")
    logger.info(f"  Checkpoint dir: {config.checkpoint_dir}")
    logger.info(f"  Max steps: {config.training.max_steps}")
    logger.info(f"  Batch size: {config.training.batch_size}")

    # ==================== 2. LOAD DATA ====================
    logger.info("\n" + "=" * 80)
    logger.info("Loading data...")
    logger.info("=" * 80)

    train_dataset, val_dataset, test_dataset = create_dataloaders(
        data_dir=config.data.data_dir,
        batch_size=config.training.batch_size,
        max_seq_len=config.training.max_seq_len,
        mode=config.data.mode,
        max_sequences={
            'train': config.data.max_train_sequences,
            'validation': config.data.max_val_sequences,
            'test': config.data.max_test_sequences,
        },
    )

    # Get clustering info from dataset
    clustering_info = train_dataset.get_clustering_info()

    logger.info(f"Train dataset: {len(train_dataset.sequences)} sequences")
    logger.info(f"Val dataset: {len(val_dataset.sequences)} sequences")
    logger.info(f"Test dataset: {len(test_dataset.sequences)} sequences")

    # ==================== 3. CREATE MODEL ====================
    logger.info("\n" + "=" * 80)
    logger.info("Creating model...")
    logger.info("=" * 80)

    # Check if using pretrained model (from config or args)
    gemma_params = None
    use_pretrained = args.use_pretrained == 'gemma' or config.model.pretrained_lm_name is not None

    if use_pretrained:
        logger.info(f"Using pretrained model: {config.model.pretrained_lm_name or 'gemma from args'}")
        freeze = args.freeze_pretrained or config.model.freeze_lm
        model, gemma_params = create_gemma_model(
            config, clustering_info, args.pretrained_path, freeze
        )
    else:
        logger.info("Training from scratch (SimpleEfficientIDSModel)")
        model = create_model_from_config(config, clustering_info)

    # ==================== 4. CREATE OPTIMIZER ====================
    logger.info("\n" + "=" * 80)
    logger.info("Creating optimizer...")
    logger.info("=" * 80)

    optimizer = create_optimizer_from_config(config)

    # ==================== 5. CREATE TRAINER ====================
    logger.info("\n" + "=" * 80)
    logger.info("Creating trainer...")
    logger.info("=" * 80)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        checkpoint_dir=config.checkpoint_dir,
        log_every=config.training.log_every,
        eval_every=config.training.eval_every,
        save_every=config.training.save_every,
        use_remat=config.training.use_remat,
        use_mixed_precision=config.training.use_mixed_precision,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        enable_profiling=args.enable_profiling,
        profiler_num_steps=args.profiler_num_steps,
        profiler_start_step=args.profiler_start_step,
    )

    logger.info("Trainer created with memory optimizations")

    # ==================== 6. INITIALIZE TRAINING STATE ====================
    logger.info("\n" + "=" * 80)
    logger.info("Initializing training state...")
    logger.info("=" * 80)

    # Create RNG
    rng = jax.random.PRNGKey(config.training.seed)

    # Get sample batch for initialization
    sample_batch = next(iter(train_dataset))

    # Initialize state (with optional pretrained params)
    freeze = use_pretrained and (args.freeze_pretrained or config.model.freeze_lm)
    state = trainer.create_train_state(
        rng, sample_batch,
        pretrained_params=gemma_params,
        freeze_transformer=freeze
    )

    if freeze:
        logger.info("🔒 Transformer frozen - only training adapters + item embeddings")

    # Count parameters
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    logger.info(f"Model initialized with {num_params:,} parameters")

    # If using pretrained weights, log it
    if gemma_params is not None:
        logger.info(f"✓ Loaded pretrained Gemma weights from {args.pretrained_path}")

    # ==================== 7. TRAIN ====================
    logger.info("\n" + "=" * 80)
    logger.info("Starting training...")
    logger.info("=" * 80)

    # Train
    state, train_metrics = trainer.train(
        state=state,
        train_dataset=iter(train_dataset),
        eval_dataset=iter(val_dataset),
        num_steps=config.training.max_steps,
    )

    logger.info(f"\nTraining metrics: {train_metrics}")

    # ==================== 8. FINAL EVALUATION ====================
    logger.info("\n" + "=" * 80)
    logger.info("Final evaluation on test set...")
    logger.info("=" * 80)

    # Evaluate on test set with full metrics
    from core.metrics import compute_metrics_from_logits

    test_metrics_all = []
    num_test_batches = min(100, len(test_dataset))

    for i, batch in enumerate(iter(test_dataset)):
        if i >= num_test_batches:
            break

        # Forward pass
        outputs = state.apply_fn(
            {'params': state.params},
            **batch,
            training=False,
        )

        # Compute metrics
        metrics = compute_metrics_from_logits(
            logits=outputs['logits'],
            labels=batch['targets'],
            weights=batch['weights'],
            k_values=config.eval.k_values,
            metric_types=config.eval.metric_types,
        )

        test_metrics_all.append(metrics)

    # Average metrics
    final_metrics = {}
    for key in test_metrics_all[0].keys():
        values = [m[key] for m in test_metrics_all]
        final_metrics[key] = sum(values) / len(values)

    logger.info("\nFinal Test Metrics:")
    logger.info("=" * 80)
    for metric_name, value in sorted(final_metrics.items()):
        logger.info(f"  {metric_name:20s}: {value:.4f}")

    logger.info("\n" + "=" * 80)
    logger.info("Training completed successfully!")
    logger.info("=" * 80)
    logger.info(f"Final checkpoint saved to: {config.checkpoint_dir}")
    logger.info(f"Total steps: {state.step}")

    return state, final_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientIDS model (Flax port)")

    # Config selection
    parser.add_argument(
        '--config',
        type=str,
        default='qwen',
        choices=['qwen', 'llama', 'debug', 'gemma', 'tpu_optimized'],
        help='Preset configuration to use'
    )

    # Pretrained model
    parser.add_argument(
        '--use_pretrained',
        type=str,
        default=None,
        choices=[None, 'gemma'],
        help='Use pretrained model (None for simple model, "gemma" for Gemma 2B)'
    )
    parser.add_argument(
        '--pretrained_path',
        type=str,
        default='/repo/uber/ai/michelangelo/sdk/inference/triton_and_llm_inference/2b',
        help='Path to pretrained model checkpoint'
    )
    parser.add_argument(
        '--freeze_pretrained',
        action='store_true',
        help='Freeze pretrained model weights (only train adapters + item embeddings)'
    )

    # Data args
    parser.add_argument('--data_dir', type=str, default=None, help='Data directory')
    parser.add_argument('--num_items', type=int, default=3261, help='Number of items')
    parser.add_argument('--num_clusters', type=int, default=100, help='Number of clusters')
    parser.add_argument('--item_embedding_dim', type=int, default=384, help='Item embedding dimension')

    # Training args
    parser.add_argument('--max_steps', type=int, default=None, help='Max training steps')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--max_seq_len', type=int, default=None, help='Max sequence length')

    # Output args
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='Checkpoint directory')

    # Profiling arguments
    parser.add_argument('--enable_profiling', action='store_true', help='Enable JAX profiler')
    parser.add_argument('--profiler_num_steps', type=int, default=5, help='Number of steps to profile')
    parser.add_argument('--profiler_start_step', type=int, default=150, help='Step to start profiling (default: 150 - after warmup)')

    args = parser.parse_args()

    # Fill in defaults
    if args.max_steps is None:
        args.max_steps = 10000 if args.config != 'debug' else 200
    if args.batch_size is None:
        args.batch_size = 16 if args.config != 'debug' else 4
    if args.max_seq_len is None:
        args.max_seq_len = 128 if args.config != 'debug' else 32

    # Run training
    final_state, final_metrics = main(args)

    print("\n" + "=" * 80)
    print("🎉 Training completed!")
    print("=" * 80)
    print("\nFinal metrics:")
    for k, v in sorted(final_metrics.items()):
        print(f"  {k}: {v:.4f}")
