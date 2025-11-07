"""
Training script for Llama-integrated EfficientIDS

This script demonstrates how to train EfficientIDS with a pretrained Llama model.
It loads Llama from HuggingFace, integrates it with item embeddings, and trains
using hierarchical softmax for efficient item prediction.

Usage:
------
# Train with Llama 1B (default)
python train_llama.py --llama_size 1b --data_dir ./data --checkpoint_dir ./checkpoints_llama

# Train with frozen Llama
python train_llama.py --llama_size 1b --freeze_llama --num_steps 1000

# Train with custom settings
python train_llama.py --llama_size 3b --batch_size 8 --learning_rate 5e-5 --hf_token YOUR_TOKEN

Requirements:
-------------
- HuggingFace token (for downloading Llama)
- Accepted Llama license
- GPU with enough memory (see memory estimates)
"""

import argparse
import sys
import os
from pathlib import Path
import pickle
import logging

# Configure JAX memory allocation BEFORE importing JAX
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.75'

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from core.llama_loader import LLAMA_CONFIGS, estimate_model_memory_gb
from core.models import LlamaEfficientIDSModel
from core.hierarchical import ClusteringInfo
from train.optimizer import create_optimizer, create_learning_rate_schedule
from train.trainer import Trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Train EfficientIDS with Llama')

    # Model
    parser.add_argument('--llama_size', type=str, default='1b', choices=['1b', '3b', '8b'],
                       help='Llama model size')
    parser.add_argument('--freeze_llama', action='store_true',
                       help='Freeze Llama weights (only train adapters + item embeddings)')
    parser.add_argument('--item_embedding_dim', type=int, default=384,
                       help='Item embedding dimension')

    # Data
    parser.add_argument('--data_dir', type=str, default='../efficientids/data/ml1m_processed',
                       help='Directory containing processed data')
    parser.add_argument('--num_items', type=int, default=3261,
                       help='Number of items in catalog')
    parser.add_argument('--num_clusters', type=int, default=100,
                       help='Number of item clusters')

    # Training
    parser.add_argument('--num_steps', type=int, default=1000,
                       help='Number of training steps')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--seq_len', type=int, default=64,
                       help='Sequence length')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=100,
                       help='Number of warmup steps')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='Gradient clipping norm')

    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_llama',
                       help='Checkpoint directory')
    parser.add_argument('--save_every', type=int, default=500,
                       help='Save checkpoint every N steps')
    parser.add_argument('--log_every', type=int, default=50,
                       help='Log metrics every N steps')

    # HuggingFace
    parser.add_argument('--hf_token', type=str, default=None,
                       help='HuggingFace token (or set HF_TOKEN env var)')
    parser.add_argument('--cache_dir', type=str, default='./model_cache',
                       help='Model cache directory')

    # Misc
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--estimate_memory_only', action='store_true',
                       help='Only estimate memory requirements, do not train')

    return parser.parse_args()


def load_clustering(data_dir: str) -> ClusteringInfo:
    """Load clustering information from processed data (PAXml format)."""
    clustering_path = Path(data_dir) / 'processed' / 'clustering.pkl'

    if not clustering_path.exists():
        logger.warning(f"⚠️  Clustering file not found: {clustering_path}")
        logger.warning("   Using synthetic clustering...")
        return None

    logger.info(f"📦 Loading clustering from: {clustering_path}")

    # PAXml format: 4 consecutive pickle objects
    try:
        with open(clustering_path, 'rb') as f:
            cluster_assignments = pickle.load(f)  # [num_items] item -> cluster
            cluster_indices = pickle.load(f)      # [num_clusters, max_size] cluster -> items
            in_cluster_id = pickle.load(f)        # [num_items] item -> position in cluster
            cluster_embeddings = pickle.load(f)   # [num_clusters, dim] cluster centers

        clustering_info = ClusteringInfo(
            cluster_assignments=cluster_assignments,
            cluster_indices=cluster_indices,
            in_cluster_id=in_cluster_id,
            cluster_embeddings=cluster_embeddings,
        )

        logger.info(f"   ✓ Num items: {len(clustering_info.cluster_assignments)}")
        logger.info(f"   ✓ Num clusters: {clustering_info.cluster_indices.shape[0]}")
        logger.info(f"   ✓ Max cluster size: {clustering_info.cluster_indices.shape[1]}")

        return clustering_info

    except Exception as e:
        logger.error(f"❌ Failed to load clustering: {e}")
        logger.warning("   Using synthetic clustering...")
        return None


def create_synthetic_data(
    batch_size: int,
    seq_len: int,
    num_items: int,
    vocab_size: int,
    key: jax.Array,
) -> dict:
    """Create synthetic training batch."""
    # Split key for different random operations
    key_text, key_items, key_mask, key_targets = jax.random.split(key, 4)

    # Text token IDs (for Llama input)
    input_ids = jax.random.randint(key_text, (batch_size, seq_len), 0, vocab_size)

    # Item IDs
    item_ids = jax.random.randint(key_items, (batch_size, seq_len), 0, num_items)

    # Item mask: 30% of positions are items, rest are text
    item_mask = (jax.random.uniform(key_mask, (batch_size, seq_len)) < 0.3).astype(jnp.float32)

    # Target items (shifted by 1)
    targets = jax.random.randint(key_targets, (batch_size, seq_len), 0, num_items)

    # Attention mask (all ones for synthetic data)
    attention_mask = jnp.ones((batch_size, seq_len))

    return {
        'input_ids': input_ids,
        'item_ids': item_ids,
        'item_mask': item_mask,
        'targets': targets,
        'attention_mask': attention_mask,
    }


def main(args):
    logger.info("=" * 70)
    logger.info("🦙 Llama + EfficientIDS Training")
    logger.info("=" * 70)

    # Memory estimation
    if args.estimate_memory_only:
        logger.info("\n📊 Memory Estimates:")
        logger.info("-" * 70)
        for size in ['1b', '3b', '8b']:
            memory = estimate_model_memory_gb(size, batch_size=args.batch_size, seq_len=args.seq_len)
            logger.info(f"  Llama {size.upper()}: ~{memory:.1f} GB")
        logger.info("\n💡 These estimates include model + activations + gradients + optimizer")
        logger.info("   Actual usage may vary based on batch size and sequence length")
        return

    # Check GPU
    devices = jax.devices()
    logger.info(f"\n🖥️  Devices: {devices}")
    if 'gpu' in str(devices[0]).lower() or 'cuda' in str(devices[0]).lower():
        logger.info("   ✓ GPU detected!")
    else:
        logger.warning("   ⚠️  No GPU detected - training will be slow!")

    # Configuration
    logger.info(f"\n📋 Configuration:")
    logger.info(f"   Llama model: {args.llama_size.upper()}")
    logger.info(f"   Freeze Llama: {args.freeze_llama}")
    logger.info(f"   Item embedding dim: {args.item_embedding_dim}")
    logger.info(f"   Batch size: {args.batch_size}")
    logger.info(f"   Sequence length: {args.seq_len}")
    logger.info(f"   Learning rate: {args.learning_rate}")
    logger.info(f"   Training steps: {args.num_steps}")

    # Get Llama config
    logger.info(f"\n🦙 Using Llama {args.llama_size.upper()} architecture...")

    if args.llama_size not in LLAMA_CONFIGS:
        logger.error(f"❌ Unknown Llama size: {args.llama_size}")
        logger.error(f"   Available: {list(LLAMA_CONFIGS.keys())}")
        return

    llama_config = LLAMA_CONFIGS[args.llama_size]
    logger.info("   ✓ Llama config:")
    logger.info(f"     Hidden size: {llama_config['hidden_size']}")
    logger.info(f"     Layers: {llama_config['num_hidden_layers']}")
    logger.info(f"     Attention heads: {llama_config['num_attention_heads']}")
    logger.info(f"     KV heads: {llama_config['num_key_value_heads']}")
    logger.info(f"   💡 Training from scratch (random init - no OOM!)")

    # Load clustering
    logger.info(f"\n📦 Loading data...")
    clustering_info = load_clustering(args.data_dir)

    if clustering_info is None:
        # Create synthetic clustering (fallback)
        logger.info("   Creating synthetic clustering...")
        cluster_assignments = np.random.randint(0, args.num_clusters, size=args.num_items)

        # Calculate max cluster size (find actual max, not estimate)
        cluster_sizes = [np.sum(cluster_assignments == i) for i in range(args.num_clusters)]
        max_cluster_size = max(cluster_sizes)

        cluster_indices = np.full((args.num_clusters, max_cluster_size), -1, dtype=np.int32)
        in_cluster_id = np.zeros(args.num_items, dtype=np.int32)

        for cluster_id in range(args.num_clusters):
            items_in_cluster = np.where(cluster_assignments == cluster_id)[0]
            cluster_indices[cluster_id, :len(items_in_cluster)] = items_in_cluster
            in_cluster_id[items_in_cluster] = np.arange(len(items_in_cluster))

        clustering_info = ClusteringInfo(
            cluster_assignments=cluster_assignments,
            cluster_indices=cluster_indices,
            in_cluster_id=in_cluster_id,
        )
        logger.info(f"   ✓ Synthetic clustering created")
        logger.info(f"   ✓ Max cluster size: {max_cluster_size}")

    # Create model
    logger.info(f"\n🏗️  Creating model...")
    model = LlamaEfficientIDSModel(
        # Llama config
        vocab_size=llama_config['vocab_size'],
        hidden_size=llama_config['hidden_size'],
        num_layers=llama_config['num_hidden_layers'],
        num_heads=llama_config['num_attention_heads'],
        num_kv_heads=llama_config['num_key_value_heads'],
        intermediate_size=llama_config['intermediate_size'],
        max_seq_len=args.seq_len,
        # EfficientIDS config
        num_items=args.num_items,
        num_clusters=args.num_clusters,
        item_embedding_dim=args.item_embedding_dim,
        clustering_info=clustering_info,
        freeze_llama=args.freeze_llama,
    )
    logger.info("   ✓ Model created!")
    logger.info(f"   ✓ Llama: {llama_config['num_hidden_layers']} layers, {llama_config['hidden_size']}d")
    logger.info(f"   ✓ Items: {args.num_items}, clusters: {args.num_clusters}")

    # Create optimizer
    logger.info(f"\n⚙️  Creating optimizer...")
    lr_schedule = create_learning_rate_schedule(
        base_learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        total_steps=args.num_steps,
        schedule_type='cosine',
    )

    # Freeze Llama parameters if requested
    frozen_params = ['llama_params'] if args.freeze_llama else None

    optimizer = create_optimizer(
        learning_rate=lr_schedule,
        optimizer_type='adamw',
        weight_decay=args.weight_decay,
        clip_grad_norm=args.grad_clip,
        frozen_params=frozen_params,
    )
    logger.info(f"   ✓ Optimizer: AdamW")
    logger.info(f"   ✓ LR schedule: cosine with warmup ({args.warmup_steps} steps)")
    logger.info(f"   ✓ Frozen params: {frozen_params}")

    # Create trainer
    logger.info(f"\n🚂 Creating trainer...")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        checkpoint_dir=args.checkpoint_dir,
        log_every=args.log_every,
        eval_every=args.num_steps + 1,  # No eval for synthetic data
        save_every=args.save_every,
    )
    logger.info(f"   ✓ Trainer created")
    logger.info(f"   ✓ Checkpoints: {args.checkpoint_dir}")

    # Initialize model
    logger.info(f"\n🎲 Initializing model...")
    rng = jax.random.PRNGKey(args.seed)

    # Create sample batch for initialization
    sample_batch = create_synthetic_data(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_items=args.num_items,
        vocab_size=llama_config['vocab_size'],
        key=rng,
    )

    state = trainer.create_train_state(rng, sample_batch)
    logger.info(f"   ✓ State initialized")
    logger.info(f"   ✓ Total parameters: {sum(p.size for p in jax.tree_util.tree_leaves(state.params)):,}")

    # Training loop
    logger.info(f"\n🔥 Starting training...")
    logger.info("=" * 70)

    for step in tqdm(range(args.num_steps), desc="Training"):
        # Generate batch
        rng, data_key = jax.random.split(rng)
        batch = create_synthetic_data(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            num_items=args.num_items,
            vocab_size=llama_config['vocab_size'],
            key=data_key,
        )

        # Training step
        state, metrics = trainer.train_step(state, batch)

        # Log
        if (step + 1) % args.log_every == 0:
            loss = metrics.get('total_loss', metrics.get('loss', 0.0))
            cluster_acc = metrics.get('cluster_accuracy', 0.0)
            cluster_loss = metrics.get('cluster_loss', 0.0)
            item_loss = metrics.get('item_loss', 0.0)

            logger.info(
                f"Step {step + 1}/{args.num_steps} | "
                f"Loss: {loss:.4f} | "
                f"Cluster: {cluster_loss:.4f} | "
                f"Item: {item_loss:.4f} | "
                f"Acc: {cluster_acc:.3f}"
            )

        # Save checkpoint
        if (step + 1) % args.save_every == 0:
            trainer.save_checkpoint(state, step + 1)
            logger.info(f"   ✓ Checkpoint saved at step {step + 1}")

    # Final save
    trainer.save_checkpoint(state, args.num_steps)

    logger.info("\n" + "=" * 70)
    logger.info("✅ Training complete!")
    logger.info(f"📁 Checkpoints saved to: {args.checkpoint_dir}")
    logger.info("\n🎯 Next steps:")
    logger.info("  1. Load real MovieLens data")
    logger.info("  2. Evaluate on validation set")
    logger.info("  3. Fine-tune with unfrozen Llama")
    logger.info("  4. Add text metadata for items")


if __name__ == "__main__":
    args = parse_args()
    main(args)
