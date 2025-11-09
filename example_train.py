#!/usr/bin/env python3
"""
Complete Training Example for EfficientIDS-Flax

This script demonstrates how to train an EfficientIDS model from scratch
using the pure JAX/Flax implementation compatible with newer JAX versions.

Usage:
    python example_train.py --data_dir ../efficientids/data/ml1m_processed

Key features demonstrated:
- Loading clustering data
- Creating model with hierarchical softmax
- Setting up optimizer with warmup schedule
- Training loop with evaluation
- Checkpointing

This works with latest JAX (0.4.30+) and modern GPUs (H100, etc.)
"""

import argparse
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path

# Import efficientids_flax components
from core import (
    SimpleEfficientIDSModel,
    ClusteringInfo,
)
from train import (
    Trainer,
    create_optimizer,
    create_learning_rate_schedule,
)


def load_clustering(data_dir: str) -> ClusteringInfo:
    """
    Load clustering information from pickle file.

    Args:
        data_dir: Path to data directory containing clustering.pkl

    Returns:
        ClusteringInfo object
    """
    clustering_path = Path(data_dir) / "processed" / "clustering.pkl"

    if not clustering_path.exists():
        print(f"⚠️  Clustering file not found: {clustering_path}")
        print("Creating synthetic clustering for demo...")

        # Create synthetic clustering (for demo purposes)
        num_items = 3261  # MovieLens-1M
        num_clusters = 100
        max_cluster_size = 50

        cluster_assignments = np.random.randint(0, num_clusters, size=num_items)
        cluster_indices = np.full((num_clusters, max_cluster_size), -1, dtype=np.int32)
        in_cluster_id = np.zeros(num_items, dtype=np.int32)

        for cluster_id in range(num_clusters):
            items_in_cluster = np.where(cluster_assignments == cluster_id)[0]
            cluster_indices[cluster_id, :len(items_in_cluster)] = items_in_cluster
            in_cluster_id[items_in_cluster] = np.arange(len(items_in_cluster))

        return ClusteringInfo(
            cluster_assignments=cluster_assignments,
            cluster_indices=cluster_indices,
            in_cluster_id=in_cluster_id,
        )

    print(f"✓ Loading clustering from: {clustering_path}")
    return ClusteringInfo.from_pickle(str(clustering_path))


def create_synthetic_dataset(
    num_items: int,
    batch_size: int,
    seq_len: int,
    seed: int = 42,
):
    """
    Create synthetic data iterator for testing.

    In production, replace this with your actual data pipeline.
    """
    key = jax.random.PRNGKey(seed)

    while True:
        key, subkey = jax.random.split(key)

        # Random item sequences
        item_ids = jax.random.randint(subkey, (batch_size, seq_len), 0, num_items)

        # Targets = next items (shifted by 1)
        targets = jnp.roll(item_ids, -1, axis=1)

        # All positions are items (no text mixing in this simple example)
        item_mask = jnp.ones((batch_size, seq_len))

        yield {
            'item_ids': item_ids,
            'targets': targets,
            'item_mask': item_mask,
        }


def main(args):
    """Main training function."""

    print("=" * 70)
    print("EfficientIDS-Flax Training Example")
    print("=" * 70)
    print()

    # ==================== CONFIGURATION ====================
    print("📋 Configuration:")
    print(f"  Data dir: {args.data_dir}")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Training steps: {args.num_steps}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Item embedding dim: {args.item_embedding_dim}")
    print()

    # ==================== LOAD DATA ====================
    print("📊 Loading data...")
    clustering_info = load_clustering(args.data_dir)

    num_items = clustering_info.num_items
    num_clusters = clustering_info.num_clusters

    print(f"  ✓ Num items: {num_items}")
    print(f"  ✓ Num clusters: {num_clusters}")
    print(f"  ✓ Max cluster size: {clustering_info.max_cluster_size}")
    print()

    # ==================== CREATE MODEL ====================
    print("🏗️  Creating model...")
    model = SimpleEfficientIDSModel(
        num_items=num_items,
        num_clusters=num_clusters,
        item_embedding_dim=args.item_embedding_dim,
        model_dims=args.model_dims,
        clustering_info=clustering_info,
        use_correction=True,
    )
    print(f"  ✓ Model created")
    print(f"  ✓ Item embedding dim: {args.item_embedding_dim}")
    print(f"  ✓ Model dims: {args.model_dims}")
    print()

    # ==================== CREATE OPTIMIZER ====================
    print("⚙️  Creating optimizer...")

    # Learning rate schedule with warmup
    schedule = create_learning_rate_schedule(
        base_learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        total_steps=args.num_steps,
        schedule_type='cosine',
        min_learning_rate_ratio=0.1,
    )

    # AdamW optimizer with gradient clipping
    optimizer = create_optimizer(
        learning_rate=schedule,
        optimizer_type='adamw',
        weight_decay=args.weight_decay,
        clip_grad_norm=args.clip_grad_norm,
    )
    print(f"  ✓ Optimizer: AdamW")
    print(f"  ✓ LR schedule: cosine with warmup ({args.warmup_steps} steps)")
    print(f"  ✓ Weight decay: {args.weight_decay}")
    print(f"  ✓ Grad clip: {args.clip_grad_norm}")
    print()

    # ==================== CREATE TRAINER ====================
    print("🚂 Creating trainer...")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        checkpoint_dir=args.checkpoint_dir,
        log_every=args.log_every,
        eval_every=args.eval_every,
        save_every=args.save_every,
    )
    print(f"  ✓ Trainer created")
    print(f"  ✓ Checkpoints will be saved to: {args.checkpoint_dir}")
    print()

    # ==================== INITIALIZE STATE ====================
    print("🎲 Initializing model...")
    rng = jax.random.PRNGKey(args.seed)

    # Create sample batch for initialization
    sample_batch = {
        'item_ids': jax.random.randint(rng, (args.batch_size, args.seq_len), 0, num_items),
        'targets': jax.random.randint(rng, (args.batch_size, args.seq_len), 0, num_items),
        'item_mask': jnp.ones((args.batch_size, args.seq_len)),
    }

    state = trainer.create_train_state(rng, sample_batch)
    print(f"  ✓ State initialized")
    print(f"  ✓ Total parameters: {sum(p.size for p in jax.tree.leaves(state.params)):,}")
    print()

    # ==================== CREATE DATASETS ====================
    print("📦 Creating datasets...")
    train_dataset = create_synthetic_dataset(
        num_items=num_items,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        seed=args.seed,
    )

    eval_dataset = create_synthetic_dataset(
        num_items=num_items,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        seed=args.seed + 1,
    )
    print(f"  ✓ Datasets created (synthetic data for demo)")
    print(f"  💡 Replace with real data pipeline for production")
    print()

    # ==================== TRAIN ====================
    print("🔥 Starting training...")
    print("=" * 70)
    print()

    state = trainer.train(
        state=state,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_steps=args.num_steps,
    )

    print()
    print("=" * 70)
    print(f"✅ Training complete! Final step: {state.step}")
    print(f"📁 Checkpoints saved to: {args.checkpoint_dir}")
    print()
    print("🎯 Next steps:")
    print("  1. Replace synthetic data with real MovieLens data")
    print("  2. Add pretrained LM (Qwen/Llama) for better performance")
    print("  3. Implement proper evaluation metrics")
    print("  4. Add distributed training for multi-GPU")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientIDS model")

    # Data
    parser.add_argument("--data_dir", type=str, default="../efficientids/data/ml1m_processed",
                       help="Data directory with clustering.pkl")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                       help="Checkpoint directory")

    # Model
    parser.add_argument("--item_embedding_dim", type=int, default=384,
                       help="Item embedding dimension")
    parser.add_argument("--model_dims", type=int, default=512,
                       help="Internal model dimension")

    # Training
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--seq_len", type=int, default=64,
                       help="Sequence length")
    parser.add_argument("--num_steps", type=int, default=1000,
                       help="Total training steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Peak learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100,
                       help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0,
                       help="Gradient clipping norm")

    # Logging
    parser.add_argument("--log_every", type=int, default=50,
                       help="Log every N steps")
    parser.add_argument("--eval_every", type=int, default=200,
                       help="Evaluate every N steps")
    parser.add_argument("--save_every", type=int, default=500,
                       help="Save checkpoint every N steps")

    # Misc
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    args = parser.parse_args()
    main(args)
