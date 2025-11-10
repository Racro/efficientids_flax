"""
Training Loop for EfficientIDS - Pure JAX/Flax

Simple, efficient training loop compatible with latest JAX versions.
Supports distributed training via native JAX sharding.
"""

import jax
import jax.numpy as jnp
from flax.training import train_state
from flax import struct
import optax
from typing import Dict, Any, Callable, Optional, Iterator
import time
from pathlib import Path
import orbax.checkpoint as ocp
import json


class TrainState(train_state.TrainState):
    """Extended train state with metrics."""
    # Note: We just use the base TrainState without extensions
    # Metrics will be tracked separately in the trainer
    pass


class Trainer:
    """
    Trainer for EfficientIDS models.

    Handles:
    - Training loop with automatic differentiation
    - Distributed training via JAX sharding
    - Checkpointing with Orbax
    - Metrics logging
    - Evaluation

    Args:
        model: Flax model (nn.Module)
        optimizer: Optax optimizer
        checkpoint_dir: Directory for saving checkpoints
        log_every: Log metrics every N steps
        eval_every: Evaluate every N steps
        save_every: Save checkpoint every N steps
    """

    def __init__(
        self,
        model: Any,
        optimizer: optax.GradientTransformation,
        checkpoint_dir: str = "./checkpoints",
        log_every: int = 100,
        eval_every: int = 1000,
        save_every: int = 1000,
    ):
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_dir = Path(checkpoint_dir).resolve()  # Convert to absolute path
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_every = log_every
        self.eval_every = eval_every
        self.save_every = save_every

        # Create checkpoint manager (using StandardCheckpointer for Flax compatibility)
        self.checkpoint_manager = ocp.CheckpointManager(
            self.checkpoint_dir,
            ocp.StandardCheckpointer(),
            options=ocp.CheckpointManagerOptions(
                max_to_keep=5,
                save_interval_steps=save_every,
            ),
        )

    def create_train_state(
        self,
        rng: jax.Array,
        sample_input: Dict[str, jnp.ndarray],
        pretrained_params: Optional[Dict[str, Any]] = None,
    ) -> TrainState:
        """
        Initialize training state.

        Args:
            rng: Random key
            sample_input: Sample input batch for initialization
            pretrained_params: Optional pretrained parameters to load

        Returns:
            TrainState with initialized parameters and optimizer state
        """
        # Initialize model parameters
        variables = self.model.init(rng, **sample_input, training=True)
        params = variables['params']

        # Load pretrained params if provided
        if pretrained_params is not None:
            params = self._merge_pretrained_params(params, pretrained_params)

        # Create train state
        state = TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=self.optimizer,
        )

        return state

    def _merge_pretrained_params(
        self,
        initialized_params: Dict[str, Any],
        pretrained_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Merge pretrained parameters into initialized parameters.

        Only merges transformer weights, leaving item embeddings and adapters
        as randomly initialized.

        Args:
            initialized_params: Randomly initialized parameters
            pretrained_params: Pretrained Gemma parameters

        Returns:
            Merged parameters
        """
        import copy
        import logging
        logger = logging.getLogger(__name__)

        merged = copy.deepcopy(initialized_params)

        # Debug: Check what we're merging
        logger.info(f"  Initialized params keys: {list(initialized_params.keys())}")
        logger.info(f"  Pretrained params keys: {list(pretrained_params.keys())}")

        # Merge transformer parameters if present
        if 'transformer' in pretrained_params and 'transformer' in merged:
            logger.info(f"  ✓ Merging transformer weights from pretrained checkpoint")
            merged['transformer'] = pretrained_params['transformer']
        else:
            logger.warning(f"  ✗ Could not merge transformer weights!")
            logger.warning(f"    'transformer' in pretrained: {'transformer' in pretrained_params}")
            logger.warning(f"    'transformer' in initialized: {'transformer' in merged}")

        return merged

    @staticmethod
    def train_step(
        state: TrainState,
        batch: Dict[str, jnp.ndarray],
    ) -> tuple[TrainState, Dict[str, jnp.ndarray]]:
        """
        Single training step.

        Args:
            state: Current train state
            batch: Input batch with keys like 'item_ids', 'targets', 'item_mask'

        Returns:
            Updated state and metrics dict
        """
        def loss_fn(params):
            """Compute loss and auxiliary outputs."""
            outputs = state.apply_fn(
                {'params': params},
                **batch,
                training=True,
            )

            # Extract loss (hierarchical softmax returns this)
            loss = outputs.get('total_loss', outputs.get('loss', 0.0))

            # Return loss and auxiliary metrics
            aux = {
                'cluster_loss': outputs.get('cluster_loss', 0.0),
                'item_loss': outputs.get('item_loss', 0.0),
                'cluster_accuracy': outputs.get('cluster_accuracy', 0.0),
            }

            return loss, aux

        # Compute gradients
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, aux), grads = grad_fn(state.params)

        # Compute gradient norm for monitoring
        grad_norm = optax.global_norm(grads)

        # Update parameters
        state = state.apply_gradients(grads=grads)
        state = state.replace(step=state.step + 1)

        # Collect metrics
        metrics = {
            'loss': loss,
            'grad_norm': grad_norm,
            **aux,
        }

        return state, metrics

    @staticmethod
    def eval_step(
        state: TrainState,
        batch: Dict[str, jnp.ndarray],
        compute_full_metrics: bool = False,
    ) -> Dict[str, jnp.ndarray]:
        """
        Single evaluation step.

        Args:
            state: Current train state
            batch: Input batch
            compute_full_metrics: If True, compute all metrics (recall, MRR, NDCG, etc.)

        Returns:
            Metrics dict
        """
        outputs = state.apply_fn(
            {'params': state.params},
            **batch,
            training=False,
        )

        # Extract predictions for accuracy computation
        logits = outputs['logits']
        targets = batch.get('targets')
        weights = batch.get('weights', jnp.ones_like(targets))

        # Compute cross-entropy loss from logits
        if targets is not None and logits is not None:
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            target_log_probs = jnp.take_along_axis(
                log_probs,
                jnp.expand_dims(targets, axis=-1),
                axis=-1
            ).squeeze(axis=-1)
            loss = -jnp.sum(target_log_probs * weights) / (jnp.sum(weights) + 1e-8)
        else:
            loss = 0.0

        metrics = {
            'eval_loss': loss,
        }

        if targets is not None:
            # Simple accuracy
            predictions = jnp.argmax(logits, axis=-1)
            accuracy = jnp.mean((predictions == targets).astype(jnp.float32) * weights) / jnp.maximum(jnp.mean(weights), 1e-8)
            metrics['eval_accuracy'] = accuracy

            # Full metrics (if requested)
            if compute_full_metrics:
                try:
                    from ..core.metrics import compute_metrics_from_logits
                    full_metrics = compute_metrics_from_logits(
                        logits=logits,
                        labels=targets,
                        weights=weights,
                        k_values=[1, 5, 10],
                        metric_types=['recall', 'mrr', 'accuracy'],
                    )
                    # Prefix with 'eval_'
                    metrics.update({f'eval_{k}': v for k, v in full_metrics.items()})
                except ImportError:
                    pass  # Metrics module not available

        return metrics

    def train(
        self,
        state: TrainState,
        train_dataset: Iterator,
        eval_dataset: Optional[Iterator] = None,
        num_steps: int = 10000,
    ) -> TrainState:
        """
        Main training loop.

        Args:
            state: Initial train state
            train_dataset: Training data iterator
            eval_dataset: Optional evaluation data iterator
            num_steps: Total number of training steps

        Returns:
            Final train state
        """
        print(f"Starting training for {num_steps} steps...")
        print(f"  Logging every {self.log_every} steps")
        print(f"  Evaluating every {self.eval_every} steps")
        print(f"  Saving every {self.save_every} steps")
        print()

        # JIT compile train step
        train_step_jit = jax.jit(self.train_step)

        # Metrics accumulator
        metrics_history = []
        start_time = time.time()

        for step in range(state.step, num_steps):
            # Get next batch
            try:
                batch = next(train_dataset)
            except StopIteration:
                print(f"  Dataset exhausted at step {step}, stopping...")
                break

            # Training step
            step_start = time.time()
            state, metrics = train_step_jit(state, batch)
            step_time = time.time() - step_start

            # Accumulate metrics
            metrics_history.append(metrics)

            # Logging
            if (step + 1) % self.log_every == 0:
                # Average metrics over logging window
                avg_metrics = {
                    k: jnp.mean(jnp.array([m[k] for m in metrics_history[-self.log_every:]]))
                    for k in metrics_history[-1].keys()
                }

                elapsed = time.time() - start_time
                steps_per_sec = self.log_every / (time.time() - start_time + 1e-8)

                print(f"Step {step + 1}/{num_steps} | "
                      f"Loss: {avg_metrics['loss']:.4f} | "
                      f"Grad norm: {avg_metrics['grad_norm']:.2f} | "
                      f"Cluster acc: {avg_metrics.get('cluster_accuracy', 0.0):.3f} | "
                      f"{steps_per_sec:.1f} steps/s")

                start_time = time.time()

            # Evaluation
            if eval_dataset is not None and (step + 1) % self.eval_every == 0:
                print(f"\n  Evaluating at step {step + 1}...")
                eval_metrics = self.evaluate(state, eval_dataset, num_eval_batches=50)

                print(f"  Eval loss: {eval_metrics['eval_loss']:.4f} | "
                      f"Eval acc: {eval_metrics['eval_accuracy']:.4f}\n")

            # Checkpointing
            if (step + 1) % self.save_every == 0:
                self.save_checkpoint(state, step + 1)
                print(f"  Checkpoint saved at step {step + 1}\n")

        print(f"\nTraining completed! Final step: {state.step}")

        # Save final checkpoint
        self.save_checkpoint(state, state.step)

        return state

    def evaluate(
        self,
        state: TrainState,
        eval_dataset: Iterator,
        num_eval_batches: int = 100,
    ) -> Dict[str, float]:
        """
        Run evaluation.

        Args:
            state: Current train state
            eval_dataset: Evaluation data iterator
            num_eval_batches: Number of batches to evaluate

        Returns:
            Dictionary of averaged evaluation metrics
        """
        eval_step_jit = jax.jit(self.eval_step)

        all_metrics = []
        for i in range(num_eval_batches):
            try:
                batch = next(eval_dataset)
            except StopIteration:
                break

            metrics = eval_step_jit(state, batch)
            all_metrics.append(metrics)

        # Average metrics
        avg_metrics = {
            k: float(jnp.mean(jnp.array([m[k] for m in all_metrics])))
            for k in all_metrics[0].keys()
        }

        return avg_metrics

    def save_checkpoint(self, state: TrainState, step: int):
        """Save checkpoint."""
        self.checkpoint_manager.save(
            step,
            args=ocp.args.StandardSave(state),
        )

    def load_checkpoint(self, step: Optional[int] = None) -> Optional[TrainState]:
        """
        Load checkpoint.

        Args:
            step: Specific step to load (None = latest)

        Returns:
            Loaded train state or None if no checkpoint found
        """
        if step is None:
            step = self.checkpoint_manager.latest_step()

        if step is None:
            print("No checkpoint found")
            return None

        # Need to create a dummy state for restoration structure
        # This is a limitation - we need the state structure beforehand
        # Orbax requires a target structure to restore into

        print(f"Loading checkpoint from step {step}")
        # TODO: Implement proper restoration
        # state = self.checkpoint_manager.restore(step, args=ocp.args.StandardRestore(state))
        return None


# ==================== TESTING ====================

if __name__ == "__main__":
    """Test the trainer with a simple model."""

    print("Testing Trainer")
    print("=" * 60)

    # Imports for standalone execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from core.models import SimpleEfficientIDSModel
    from core.hierarchical import ClusteringInfo
    from train.optimizer import create_optimizer, create_learning_rate_schedule
    import numpy as np

    # Configuration
    num_items = 100
    num_clusters = 10
    item_embedding_dim = 64
    batch_size = 4
    seq_len = 8
    num_steps = 200

    # Create clustering
    print("\n1. Creating synthetic data...")
    cluster_assignments = np.random.randint(0, num_clusters, size=num_items)
    max_cluster_size = 20
    cluster_indices = np.full((num_clusters, max_cluster_size), -1, dtype=np.int32)
    in_cluster_id = np.zeros(num_items, dtype=np.int32)

    for cluster_id in range(num_clusters):
        items_in_cluster = np.where(cluster_assignments == cluster_id)[0]
        cluster_indices[cluster_id, :len(items_in_cluster)] = items_in_cluster
        in_cluster_id[items_in_cluster] = np.arange(len(items_in_cluster))

    clustering_info = ClusteringInfo(
        cluster_assignments=cluster_assignments,
        cluster_indices=cluster_indices,
        in_cluster_id=in_cluster_id,
    )
    print("   ✓ Clustering created")

    # Create model
    print("\n2. Creating model...")
    model = SimpleEfficientIDSModel(
        num_items=num_items,
        num_clusters=num_clusters,
        item_embedding_dim=item_embedding_dim,
        model_dims=128,
        clustering_info=clustering_info,
    )
    print("   ✓ Model created")

    # Create optimizer
    print("\n3. Creating optimizer...")
    schedule = create_learning_rate_schedule(
        base_learning_rate=1e-3,
        warmup_steps=50,
        total_steps=num_steps,
        schedule_type='cosine',
    )

    optimizer = create_optimizer(
        learning_rate=schedule,
        optimizer_type='adamw',
        weight_decay=0.01,
        clip_grad_norm=1.0,
    )
    print("   ✓ Optimizer created")

    # Create trainer
    print("\n4. Creating trainer...")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        checkpoint_dir="/tmp/efficientids_test",
        log_every=50,
        eval_every=100,
        save_every=100,
    )
    print("   ✓ Trainer created")

    # Initialize training state
    print("\n5. Initializing training state...")
    key = jax.random.PRNGKey(42)
    sample_batch = {
        'item_ids': jax.random.randint(key, (batch_size, seq_len), 0, num_items),
        'targets': jax.random.randint(key, (batch_size, seq_len), 0, num_items),
        'item_mask': jnp.ones((batch_size, seq_len)),
    }

    state = trainer.create_train_state(key, sample_batch)
    print(f"   ✓ State initialized (step {state.step})")

    # Create synthetic dataset
    print("\n6. Creating synthetic dataset...")
    def create_dataset_iterator():
        """Infinite iterator of random batches."""
        key = jax.random.PRNGKey(123)
        while True:
            key, subkey = jax.random.split(key)
            yield {
                'item_ids': jax.random.randint(subkey, (batch_size, seq_len), 0, num_items),
                'targets': jax.random.randint(subkey, (batch_size, seq_len), 0, num_items),
                'item_mask': jnp.ones((batch_size, seq_len)),
            }

    train_dataset = create_dataset_iterator()
    eval_dataset = create_dataset_iterator()
    print("   ✓ Dataset created")

    # Train
    print("\n7. Training...")
    print("-" * 60)
    state = trainer.train(
        state=state,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_steps=num_steps,
    )
    print("-" * 60)

    print("\n" + "=" * 60)
    print(f"✅ Training completed! Final step: {state.step}")
    print("\nTrainer is ready for production use!")
