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
    get_debug_config,
    EfficientIDSConfig,
)
from data.dataset import create_dataloaders, ClusteringInfo
from core.models import SimpleEfficientIDSModel
from core.hierarchical import HierarchicalSoftmax
from train.trainer import Trainer
from train.optimizer import create_optimizer, create_learning_rate_schedule

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
    )

    logger.info("Trainer created")

    # ==================== 6. INITIALIZE TRAINING STATE ====================
    logger.info("\n" + "=" * 80)
    logger.info("Initializing training state...")
    logger.info("=" * 80)

    # Create RNG
    rng = jax.random.PRNGKey(config.training.seed)

    # Get sample batch for initialization
    sample_batch = next(iter(train_dataset))

    # Initialize state
    state = trainer.create_train_state(rng, sample_batch)

    # Count parameters
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    logger.info(f"Model initialized with {num_params:,} parameters")

    # ==================== 7. TRAIN ====================
    logger.info("\n" + "=" * 80)
    logger.info("Starting training...")
    logger.info("=" * 80)

    # Train
    state = trainer.train(
        state=state,
        train_dataset=iter(train_dataset),
        eval_dataset=iter(val_dataset),
        num_steps=config.training.max_steps,
    )

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
        choices=['qwen', 'llama', 'debug'],
        help='Preset configuration to use'
    )

    # Data args
    parser.add_argument('--data_dir', type=str, default=None, help='Data directory')
    parser.add_argument('--num_items', type=int, default=3261, help='Number of items')
    parser.add_argument('--num_clusters', type=int, default=100, help='Number of clusters')

    # Training args
    parser.add_argument('--max_steps', type=int, default=None, help='Max training steps')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--max_seq_len', type=int, default=None, help='Max sequence length')

    # Output args
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='Checkpoint directory')

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
