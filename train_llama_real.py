"""
Train EfficientIDS with Llama on Real MovieLens Data

Complete port from PAXml with:
- Real MovieLens sequences
- Text metadata interleaving (75%/25% split)
- Proper loss masking (item→item only)
- Three training modes: train_from_scratch, frozen_pretrained, train_pretrained
"""

import os
import sys
from pathlib import Path
import argparse
import logging

# Configure JAX memory BEFORE any JAX imports
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.75'

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from data.movielens_loader import (
    MovieLensDataLoader,
    MovieLensDataset,
    SentencePieceTokenizer,
)
from core.llama_loader import LLAMA_CONFIGS
from core.llama_flax import LlamaModel
from core.embeddings import ItemEmbedding, ItemInputAdapter, ItemOutputAdapter
from core.hierarchical import ClusteringInfo, HierarchicalSoftmax
from train.optimizer import create_optimizer, create_learning_rate_schedule
import flax.linen as nn
from flax.training import train_state
from typing import Dict, Optional
import orbax.checkpoint as ocp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InterleavedLlamaEfficientIDS(nn.Module):
    """
    EfficientIDS with Llama for interleaved text-item sequences.

    Handles format: [text_token, text_token, item_id, text_token, item_id, ...]
    """
    # EfficientIDS config
    num_items: int
    num_clusters: int
    item_embedding_dim: int

    # Llama config
    vocab_size: int = 128256
    hidden_size: int = 2048
    num_layers: int = 16
    num_heads: int = 32
    num_kv_heads: int = 8
    intermediate_size: int = 8192
    max_seq_len: int = 384

    # Optional configs
    clustering_info: Optional[ClusteringInfo] = None
    freeze_llama: bool = True

    def setup(self):
        """
        Initialize all components.

        PAXml Mode: use_item_input_dnn_everywhere=True
        - Project item embeddings (384-dim) → hidden_size (2048-dim)
        - Project cluster embeddings (118-dim) → hidden_size (2048-dim)
        - All computations in high-dimensional space
        - NO item_output_dnn needed
        """
        # Shared item embedding table (384-dim)
        self.item_embedding_table = self.param(
            'item_embedding_table',
            nn.initializers.xavier_uniform(),
            (self.num_items, self.item_embedding_dim)
        )

        # PAXml: item_input_dnn projects embeddings to model dimension
        # Item adapter: 384-dim → 2048-dim
        self.item_input_adapter = ItemInputAdapter(
            item_embedding_dim=self.item_embedding_dim,
            model_dims=self.hidden_size,
        )

        # Cluster adapter: 118-dim → 2048-dim (separate adapter for different input dim)
        cluster_emb_dim = self.clustering_info.cluster_embeddings.shape[1] if self.clustering_info else 384
        self.cluster_input_adapter = ItemInputAdapter(
            item_embedding_dim=cluster_emb_dim,
            model_dims=self.hidden_size,
        )

        # Text embedding table (shared with Llama)
        self.text_embedding_table = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.hidden_size,
            name='text_embeddings',
        )

        # Llama transformer
        self.llama = LlamaModel(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            intermediate_size=self.intermediate_size,
            max_seq_len=self.max_seq_len,
        )

        # Hierarchical softmax (operates in high-dimensional space)
        if self.clustering_info is not None:
            self.hierarchical_softmax = HierarchicalSoftmax(
                num_items=self.num_items,
                num_clusters=self.num_clusters,
                item_embedding_dim=self.hidden_size,  # HIGH-dim space (2048)
                clustering_info=self.clustering_info,
                use_item_input_dnn_everywhere=True,  # PAXml mode
                item_input_adapter=self.item_input_adapter,  # For item embeddings (384→2048)
                cluster_input_adapter=self.cluster_input_adapter,  # For cluster embeddings (118→2048)
            )

    def __call__(
        self,
        input_ids: jnp.ndarray,  # [batch, seq_len] - interleaved text + item IDs
        item_weights: jnp.ndarray,  # [batch, seq_len] - 0=text, 1=item
        attention_mask: Optional[jnp.ndarray] = None,
        targets: Optional[jnp.ndarray] = None,
        loss_mask: Optional[jnp.ndarray] = None,
        training: bool = True,
    ) -> Dict[str, jnp.ndarray]:
        """
        Forward pass - exact port of PAXml with use_item_input_dnn_everywhere=True

        PAXml flow (InterleavedTransformerLm._prepare_input):
        1. Get text embeddings from Llama's embedding layer
        2. Get item embeddings (384-dim) and project to hidden_size (2048-dim)
        3. Interleave text and projected item embeddings
        4. Pass through transformer
        5. Compute hierarchical softmax in HIGH-dimensional space (2048-dim)
           - Project cluster embeddings (118-dim → 2048-dim)
           - Project item embeddings (384-dim → 2048-dim)
           - All einsum computations in 2048-dim space
        """
        batch_size, seq_len = input_ids.shape

        # ===== PAXml: _prepare_input =====
        # Step 1: Get text embeddings
        # PAXml: input_emb = self.softmax.emb_lookup(inputs)
        text_emb = self.text_embedding_table(input_ids)  # [batch, seq_len, hidden_size]

        # Step 2: Get item embeddings and project to model dimension
        # PAXml: item_embeddings = self.item_input_dnn(item_embeddings)
        item_ids_safe = jnp.clip(input_ids, 0, self.num_items - 1)
        item_embs_raw = self.item_embedding_table[item_ids_safe]  # [batch, seq_len, 384]
        item_embs_projected = self.item_input_adapter(item_embs_raw)  # [batch, seq_len, 2048]

        # Step 3: Interleave text and item embeddings
        # PAXml: input_emb = jnp.where(item_weights, item_embeddings, input_emb)
        mask_expanded = jnp.expand_dims(item_weights, -1)  # [batch, seq_len, 1]
        interleaved_emb = jnp.where(mask_expanded, item_embs_projected, text_emb)

        # Step 4: Pass through transformer
        # PAXml: output = self.transformer(inputs, ...)
        hidden_states = self.llama(
            inputs_embeds=interleaved_emb,
            attention_mask=attention_mask,
            training=training,
        )  # [batch, seq_len, 2048]

        # Step 5: Apply freeze if requested (stop gradient on Llama)
        if self.freeze_llama:
            hidden_states = jax.lax.stop_gradient(hidden_states)

        # Step 6: Hierarchical softmax in HIGH-dimensional space
        # PAXml: All computations happen in 2048-dim space
        # - cluster_embeddings (118-dim) → projected to 2048-dim inside hierarchical_softmax
        # - item_embeddings (384-dim) → projected to 2048-dim inside hierarchical_softmax
        # - einsum('...j,ij->...i', hidden[2048], embeddings[2048])
        if self.clustering_info is not None and training:
            # Clip targets to valid item range (text tokens will be out of bounds)
            # PAXml uses jnp.take which wraps, but loss_mask zeros out text target positions
            targets_safe = jnp.clip(targets, 0, self.num_items - 1)

            logits, metrics = self.hierarchical_softmax(
                hidden_states=hidden_states,  # [batch, seq_len, 2048]
                item_embeddings=self.item_embedding_table,  # [num_items, 384] - will be projected
                targets=targets_safe,  # Clipped to valid range; loss_mask will zero inappropriate predictions
                loss_mask=loss_mask,
                training=True,
            )
        else:
            # Inference: project item embeddings and compute in high-dim
            item_embs_projected = self.item_input_adapter(self.item_embedding_table)  # [num_items, 2048]
            logits = jnp.einsum('bsd,id->bsi', hidden_states, item_embs_projected)
            metrics = {}

        return {
            'logits': logits,
            **metrics,
        }


def parse_args():
    parser = argparse.ArgumentParser(description='Train EfficientIDS with Llama on Real Data')

    # Data
    parser.add_argument('--data_dir', type=str,
                       default='../efficientids/data/ml1m_processed/processed',
                       help='Path to processed MovieLens data')
    parser.add_argument('--mode', type=str, default='text_metadata',
                       choices=['text_metadata', 'id_only'],
                       help='Training mode')

    # Model
    parser.add_argument('--llama_size', type=str, default='1b',
                       choices=['1b', '3b', '8b'],
                       help='Llama model size')
    parser.add_argument('--training_strategy', type=str, default='train_from_scratch',
                       choices=['train_from_scratch', 'frozen_pretrained', 'train_pretrained'],
                       help='Training strategy')

    # Training
    parser.add_argument('--num_steps', type=int, default=5000,
                       help='Number of training steps')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=str, default='1e-4',
                       help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=500,
                       help='Warmup steps')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')

    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_real',
                       help='Checkpoint directory')
    parser.add_argument('--save_every', type=int, default=500,
                       help='Save checkpoint every N steps')
    parser.add_argument('--log_every', type=int, default=50,
                       help='Log every N steps')

    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


def main(args):
    logger.info("=" * 70)
    logger.info("🚀 Training EfficientIDS with Llama on Real MovieLens Data")
    logger.info("=" * 70)

    logger.info(f"\n📋 Configuration:")
    logger.info(f"   Data dir: {args.data_dir}")
    logger.info(f"   Mode: {args.mode}")
    logger.info(f"   Llama: {args.llama_size}")
    logger.info(f"   Strategy: {args.training_strategy}")
    logger.info(f"   Steps: {args.num_steps}")
    logger.info(f"   Batch size: {args.batch_size}")
    logger.info(f"   LR: {args.learning_rate}")

    # Load data
    logger.info(f"\n📦 Loading MovieLens data...")
    data_loader = MovieLensDataLoader(args.data_dir)

    logger.info(f"   ✅ {data_loader.num_items} items")
    logger.info(f"   ✅ {data_loader.num_clusters} clusters")
    logger.info(f"   ✅ {len(data_loader.movie_info)} movies with metadata")

    # Create datasets
    seq_len = 384 if args.mode == 'text_metadata' else 128

    train_dataset = MovieLensDataset(
        data_loader=data_loader,
        split='train',
        mode=args.mode,
        seq_len_text=384,
        seq_len_id_only=128,
        seed=args.seed,
    )

    logger.info(f"   ✅ {len(train_dataset)} training sequences")
    logger.info(f"   ✅ Sequence length: {seq_len}")

    # Create clustering info
    # PAXml: use_item_input_dnn_everywhere=True, so load 118-dim cluster centers from file
    clustering_info = ClusteringInfo(
        cluster_assignments=data_loader.cluster_assignments,
        cluster_indices=data_loader.cluster_indices,
        in_cluster_id=data_loader.in_cluster_id,
        cluster_embeddings=data_loader.cluster_centers,  # 118-dim, will be projected to 2048
    )

    # Get Llama config
    llama_config = LLAMA_CONFIGS[args.llama_size]
    logger.info(f"\n🦙 Llama {args.llama_size.upper()} config:")
    logger.info(f"   Hidden: {llama_config['hidden_size']}")
    logger.info(f"   Layers: {llama_config['num_layers']}")
    logger.info(f"   Heads: {llama_config['num_heads']}")

    # Create model
    freeze_llama = (args.training_strategy in ['train_from_scratch', 'frozen_pretrained'])

    model = InterleavedLlamaEfficientIDS(
        num_items=data_loader.num_items,
        num_clusters=data_loader.num_clusters,
        item_embedding_dim=384,
        vocab_size=llama_config['vocab_size'],
        hidden_size=llama_config['hidden_size'],
        num_layers=llama_config['num_layers'],
        num_heads=llama_config['num_heads'],
        num_kv_heads=llama_config['num_kv_heads'],
        intermediate_size=llama_config['intermediate_size'],
        max_seq_len=seq_len,
        clustering_info=clustering_info,
        freeze_llama=freeze_llama,
    )

    logger.info(f"\n🏗️  Model created")
    logger.info(f"   Freeze Llama: {freeze_llama}")

    # Create optimizer
    lr = float(args.learning_rate)
    lr_schedule = create_learning_rate_schedule(
        base_learning_rate=lr,
        warmup_steps=args.warmup_steps,
        total_steps=args.num_steps,
        schedule_type='cosine',
    )

    optimizer = create_optimizer(
        learning_rate=lr_schedule,
        weight_decay=args.weight_decay,
        grad_clip=1.0,
        frozen_patterns=['llama'] if freeze_llama else [],
    )

    logger.info(f"\n⚙️  Optimizer: AdamW")
    logger.info(f"   LR schedule: cosine with warmup")
    logger.info(f"   Warmup steps: {args.warmup_steps}")

    # Initialize model
    logger.info(f"\n🎲 Initializing model...")
    rng = jax.random.PRNGKey(args.seed)

    # Get a sample batch
    sample_iter = iter(train_dataset)
    sample_batch = next(sample_iter)

    sample_input = {
        'input_ids': jnp.array(sample_batch['input_ids'][None, :]),  # Add batch dim
        'item_weights': jnp.array(sample_batch['item_weights'][None, :]),
        'attention_mask': jnp.array(sample_batch['attention_mask'][None, :]),
        'targets': jnp.array(sample_batch['targets'][None, :]),
        'loss_mask': jnp.array(sample_batch['loss_mask'][None, :]),
    }

    variables = model.init(rng, **sample_input, training=True)

    # Create train state
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optimizer,
    )

    num_params = sum(x.size for x in jax.tree_util.tree_leaves(variables['params']))
    logger.info(f"   ✅ {num_params:,} parameters")

    # Training loop
    logger.info(f"\n🔥 Starting training...")
    logger.info(f"=" * 70)

    # Debug: Run one forward pass without JIT to see what's happening
    logger.info("\n🔍 Debug: Running first forward pass...")
    debug_batch = next(iter(train_dataset))
    debug_input = {
        'input_ids': jnp.array(debug_batch['input_ids'][None, :]),
        'item_weights': jnp.array(debug_batch['item_weights'][None, :]),
        'attention_mask': jnp.array(debug_batch['attention_mask'][None, :]),
        'targets': jnp.array(debug_batch['targets'][None, :]),
        'loss_mask': jnp.array(debug_batch['loss_mask'][None, :]),
    }

    logger.info(f"   Input shapes: input_ids={debug_input['input_ids'].shape}, targets={debug_input['targets'].shape}")
    logger.info(f"   Loss mask sum: {jnp.sum(debug_input['loss_mask'])}")
    logger.info(f"   Item weights sum: {jnp.sum(debug_input['item_weights'])}")

    try:
        debug_outputs = model.apply(
            {'params': variables['params']},
            **debug_input,
            training=True,
        )
        logger.info(f"   ✓ Forward pass successful")
        logger.info(f"   Total loss: {debug_outputs['total_loss']:.4f}")
        logger.info(f"   Cluster loss: {debug_outputs.get('cluster_loss', 0.0):.4f}")
        logger.info(f"   Item loss: {debug_outputs.get('item_loss', 0.0):.4f}")
    except Exception as e:
        logger.error(f"   ❌ Forward pass failed: {e}")
        raise

    @jax.jit
    def train_step(state, batch):
        def loss_fn(params):
            outputs = state.apply_fn(
                {'params': params},
                input_ids=batch['input_ids'],
                item_weights=batch['item_weights'],
                attention_mask=batch['attention_mask'],
                targets=batch['targets'],
                loss_mask=batch['loss_mask'],
                training=True,
            )
            return outputs['total_loss'], outputs

        (loss, outputs), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)

        grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads)))

        return state, {
            'loss': loss,
            'cluster_loss': outputs.get('cluster_loss', 0.0),
            'item_loss': outputs.get('item_loss', 0.0),
            'cluster_accuracy': outputs.get('cluster_accuracy', 0.0),
            'grad_norm': grad_norm,
        }

    # Setup checkpointing
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpointer = ocp.StandardCheckpointer()
    checkpoint_manager = ocp.CheckpointManager(
        checkpoint_dir,
        checkpointer,
        options=ocp.CheckpointManagerOptions(max_to_keep=5, save_interval_steps=args.save_every),
    )

    # Create data iterator (infinite loop)
    def data_generator():
        while True:
            for batch in train_dataset:
                yield {
                    'input_ids': jnp.array(batch['input_ids'][None, :]),
                    'item_weights': jnp.array(batch['item_weights'][None, :]),
                    'attention_mask': jnp.array(batch['attention_mask'][None, :]),
                    'targets': jnp.array(batch['targets'][None, :]),
                    'loss_mask': jnp.array(batch['loss_mask'][None, :]),
                }

    data_iter = data_generator()

    # Training loop
    for step in tqdm(range(1, args.num_steps + 1), desc="Training"):
        batch = next(data_iter)
        state, metrics = train_step(state, batch)

        if step % args.log_every == 0:
            logger.info(
                f"Step {step}/{args.num_steps} | "
                f"Loss: {metrics['loss']:.4f} | "
                f"Cluster: {metrics['cluster_loss']:.4f} | "
                f"Item: {metrics['item_loss']:.4f} | "
                f"Acc: {metrics['cluster_accuracy']:.3f} | "
                f"Grad: {metrics['grad_norm']:.3f}"
            )

        if step % args.save_every == 0:
            checkpoint_manager.save(step, args=ocp.args.StandardSave(state))
            logger.info(f"   💾 Checkpoint saved at step {step}")

    # Final save
    checkpoint_manager.save(args.num_steps, args=ocp.args.StandardSave(state))

    logger.info(f"\n" + "=" * 70)
    logger.info(f"✅ Training complete!")
    logger.info(f"📁 Checkpoints: {checkpoint_dir}")
    logger.info(f"=" * 70)


if __name__ == "__main__":
    args = parse_args()
    main(args)
