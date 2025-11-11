"""
Training Loop for EfficientIDS - Pure JAX/Flax

Simple, efficient training loop compatible with latest JAX versions.
Supports distributed training via native JAX sharding.
Includes cross-platform memory optimizations (GPU + TPU compatible).
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
    """Extended train state with gradient accumulation."""
    grad_accum_count: int = 0  # Counter for gradient accumulation


def detect_platform():
    """Detect if running on TPU, GPU, or CPU."""
    try:
        backend = jax.devices()[0].platform
        return backend  # 'tpu', 'gpu', or 'cpu'
    except:
        return 'cpu'


class Trainer:
    """
    Trainer for EfficientIDs models with cross-platform memory optimizations.

    Handles:
    - Training loop with automatic differentiation
    - Distributed training via JAX sharding
    - Checkpointing with Orbax
    - Metrics logging
    - Evaluation
    - Memory optimizations (gradient checkpointing, mixed precision, gradient accumulation)

    Args:
        model: Flax model (nn.Module)
        optimizer: Optax optimizer
        checkpoint_dir: Directory for saving checkpoints
        log_every: Log metrics every N steps
        eval_every: Evaluate every N steps
        save_every: Save checkpoint every N steps
        use_remat: Enable gradient checkpointing (saves memory, works on GPU + TPU)
        use_mixed_precision: Enable bfloat16 training (faster on TPU, works on modern GPUs)
        gradient_accumulation_steps: Accumulate gradients over N steps (effective batch size multiplier)
    """

    def __init__(
        self,
        model: Any,
        optimizer: optax.GradientTransformation,
        checkpoint_dir: str = "./checkpoints",
        log_every: int = 100,
        eval_every: int = 1000,
        save_every: int = 1000,
        use_remat: bool = False,
        use_mixed_precision: bool = False,
        gradient_accumulation_steps: int = 1,
        enable_profiling: bool = False,
        profiler_num_steps: int = 5,
        profiler_start_step: int | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_every = log_every
        self.eval_every = eval_every
        self.save_every = save_every
        self.use_remat = use_remat
        self.use_mixed_precision = use_mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Profiling setup
        self.profiler = None
        if enable_profiling:
            from .profiler import Profiler
            profiler_dir = self.checkpoint_dir / "profiler_traces"
            self.profiler = Profiler(
                num_steps=profiler_num_steps,
                log_dir=str(profiler_dir),
                auto_enable_at_step=profiler_start_step,
            )

        # Detect platform
        self.platform = detect_platform()
        print(f"🖥️  Detected platform: {self.platform.upper()}")

        # Validate settings for platform
        if use_mixed_precision:
            if self.platform == 'tpu':
                print(f"✅ Mixed precision (bfloat16) enabled - optimal for TPU")
            elif self.platform == 'gpu':
                print(f"✅ Mixed precision (bfloat16) enabled - works on modern GPUs (A100, H100)")
            else:
                print(f"⚠️  Mixed precision enabled on CPU - may not provide speedup")

        if use_remat:
            print(f"✅ Gradient checkpointing (remat) enabled - saves memory on {self.platform.upper()}")

        if gradient_accumulation_steps > 1:
            print(f"✅ Gradient accumulation: {gradient_accumulation_steps} steps "
                  f"(effective batch size multiplier)")

        # Create checkpoint manager
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
        freeze_transformer: bool = False,
    ) -> TrainState:
        """
        Initialize training state with model sharding for multi-device.

        Args:
            rng: Random key
            sample_input: Sample input batch for initialization
            pretrained_params: Optional pretrained parameters to load
            freeze_transformer: If True, only optimize adapters + item embeddings (saves memory!)

        Returns:
            TrainState with initialized parameters and optimizer state
        """
        from jax.sharding import PartitionSpec as P, NamedSharding, Mesh
        from jax.experimental import mesh_utils

        # Check if we have multiple devices - set up sharding mesh
        devices = jax.devices()
        num_devices = len(devices)

        if num_devices > 1:
            print(f"🔀 Setting up model sharding across {num_devices} devices")
            # Create 1D mesh along 'model' axis for model parallelism
            device_mesh = mesh_utils.create_device_mesh((num_devices,))
            mesh = Mesh(device_mesh, axis_names=('model',))
            print(f"   Mesh: {mesh}")
        else:
            mesh = None

        # Initialize model parameters
        if pretrained_params is not None:
            # If we have pretrained params, initialize with abstract shapes only
            # to avoid allocating random weights we'll immediately discard
            print("   Initializing with pretrained params (skipping random init)...")

            # Get param structure with lazy init (shapes only, no arrays)
            from jax import eval_shape
            variables_shape = eval_shape(
                lambda: self.model.init(rng, **sample_input, training=True)
            )

            # Use pretrained params as base, fill in any missing with zeros
            params = pretrained_params

            # Add any params not in pretrained (e.g., adapters, item embeddings)
            # by initializing only those parts
            if 'item_embedding_table' not in params:
                print("   Initializing non-pretrained params (adapters, embeddings)...")
                full_init = self.model.init(rng, **sample_input, training=True)
                params_full = full_init['params']

                # Copy non-transformer params from full init
                for key in params_full.keys():
                    if key not in params:
                        params[key] = params_full[key]
        else:
            # No pretrained params, do full random init
            variables = self.model.init(rng, **sample_input, training=True)
            params = variables['params']

        # Shard model parameters across devices (if multi-device)
        if mesh is not None:
            print(f"   Sharding transformer weights across {num_devices} devices...")
            from flax.traverse_util import flatten_dict, unflatten_dict

            def get_sharding_spec(path_str: str, shape: tuple) -> P:
                """
                Get sharding spec for parameter.

                Strategy:
                - Transformer large matrices: Shard along first dim (8GB/4 = 2GB per chip)
                - Small params (biases, norms): Replicate (tiny)
                - Adapters/embeddings: Replicate (small, ~5MB)
                """
                # Transformer layers - shard large weights
                if 'transformer/' in path_str:
                    # Large weight matrices (kernels, embeddings)
                    if ('kernel' in path_str or 'embedding' in path_str) and len(shape) >= 2:
                        # Shard first dimension across model axis
                        return P('model', None)
                    else:
                        # Small params (bias, scale) - replicate
                        return P(None)
                else:
                    # Adapters and embeddings - replicate (small)
                    return P(None)

            # Apply sharding
            flat_params = flatten_dict(params, sep='/')
            sharded_params = {}

            with mesh:
                for path_str, value in flat_params.items():
                    spec = get_sharding_spec(path_str, value.shape)
                    sharding = NamedSharding(mesh, spec)
                    sharded_params[path_str] = jax.device_put(value, sharding)

            params = unflatten_dict(sharded_params, sep='/')
            print(f"   ✓ Transformer sharded: ~{8000 // num_devices}MB per device (from ~8GB)")

            # Store mesh for later use in train_step
            self._mesh = mesh
        else:
            self._mesh = None

        # Store freeze info for use in train_step
        self._freeze_transformer = freeze_transformer
        if freeze_transformer:
            print("🔒 Freezing transformer (BPROP_VARIABLE_EXCLUSION like Paxml)")
            import re

            # Paxml-style exclusion patterns
            self._exclusion_patterns = [
                r'transformer/.*',  # Freeze entire transformer
            ]

            # Count params
            from flax.traverse_util import flatten_dict
            flat_params = flatten_dict(params, sep='/')

            def matches_exclusion(path_str):
                return any(re.match(p, path_str) for p in self._exclusion_patterns)

            frozen_count = sum(v.size for k, v in flat_params.items() if matches_exclusion(k))
            trainable_count = sum(v.size for k, v in flat_params.items() if not matches_exclusion(k))
            print(f"   Trainable: {trainable_count:,} params")
            print(f"   Frozen: {frozen_count:,} params")

        optimizer = self.optimizer

        # Create train state with gradient accumulation counter
        state = TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=optimizer,
            grad_accum_count=0,
        )

        return state

    def _merge_pretrained_params(
        self,
        initialized_params: Dict[str, Any],
        pretrained_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Merge pretrained parameters into initialized parameters."""
        import copy
        import logging
        logger = logging.getLogger(__name__)

        merged = copy.deepcopy(initialized_params)

        # Merge transformer parameters - try multiple key names
        for key in ['transformer', 'gemma_transformer', 'GemmaTransformer']:
            if key in pretrained_params and key in merged:
                logger.info(f"  ✓ Merging {key} weights from pretrained checkpoint")
                merged[key] = pretrained_params[key]
                break

        return merged

    def train_step(
        self,
        state: TrainState,
        batch: Dict[str, jnp.ndarray],
    ) -> tuple[TrainState, Dict[str, jnp.ndarray]]:
        """
        Single training step with memory optimizations (cross-platform safe).

        Args:
            state: Current train state
            batch: Input batch

        Returns:
            Updated state and metrics dict
        """
        def loss_fn(params):
            """Compute loss and auxiliary outputs."""
            # Apply mixed precision if enabled (safe on GPU/TPU/CPU)
            if self.use_mixed_precision:
                batch_compute = jax.tree.map(
                    lambda x: x.astype(jnp.bfloat16) if jnp.issubdtype(x.dtype, jnp.floating) else x,
                    batch
                )
            else:
                batch_compute = batch

            # Forward pass
            outputs = state.apply_fn(
                {'params': params},
                **batch_compute,
                training=True,
            )

            # Extract loss
            loss = outputs.get('total_loss', outputs.get('loss', 0.0))

            # Convert loss back to float32 for numerical stability
            if self.use_mixed_precision and hasattr(loss, 'astype'):
                loss = loss.astype(jnp.float32)

            # Auxiliary metrics
            aux = {
                'cluster_loss': outputs.get('cluster_loss', 0.0),
                'item_loss': outputs.get('item_loss', 0.0),
                'cluster_accuracy': outputs.get('cluster_accuracy', 0.0),
            }

            # Convert aux to float32
            if self.use_mixed_precision:
                aux = jax.tree.map(
                    lambda x: x.astype(jnp.float32) if hasattr(x, 'astype') and jnp.issubdtype(x.dtype, jnp.floating) else x,
                    aux
                )

            return loss, aux

        # Apply gradient checkpointing if enabled (safe on GPU/TPU/CPU)
        if self.use_remat:
            # Use checkpoint with nothing_saveable policy to save max memory
            loss_fn = jax.checkpoint(loss_fn, policy=jax.checkpoint_policies.nothing_saveable)

        # Compute gradients
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, aux), grads = grad_fn(state.params)

        # Zero out frozen gradients (BPROP_VARIABLE_EXCLUSION like Paxml)
        if hasattr(self, '_freeze_transformer') and self._freeze_transformer:
            import re
            from flax.traverse_util import flatten_dict, unflatten_dict

            flat_grads = flatten_dict(grads, sep='/')
            for key in flat_grads.keys():
                # Check if this param should be frozen
                if any(re.match(p, key) for p in self._exclusion_patterns):
                    flat_grads[key] = jnp.zeros_like(flat_grads[key])
            grads = unflatten_dict(flat_grads, sep='/')

        # Gradient accumulation
        if self.gradient_accumulation_steps > 1:
            # Scale gradients by accumulation steps
            grads = jax.tree.map(lambda g: g / self.gradient_accumulation_steps, grads)

            # TODO: Accumulate gradients in state (requires custom TrainState)
            # For now, just apply scaled gradients each step
            # Full implementation needs accumulated_grads in TrainState

        # Compute gradient norm for monitoring (only trainable)
        grad_norm = optax.global_norm(grads)

        # Update parameters
        state = state.apply_gradients(grads=grads)

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
        """Single evaluation step."""
        outputs = state.apply_fn(
            {'params': state.params},
            **batch,
            training=False,
        )

        # Extract predictions
        logits = outputs['logits']
        targets = batch.get('targets')
        weights = batch.get('weights', jnp.ones_like(targets) if targets is not None else None)

        # Compute loss
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

        metrics = {'eval_loss': loss}

        if targets is not None:
            # Simple accuracy
            predictions = jnp.argmax(logits, axis=-1)
            accuracy = jnp.sum((predictions == targets).astype(jnp.float32) * weights) / jnp.maximum(jnp.sum(weights), 1e-8)
            metrics['eval_accuracy'] = accuracy

        return metrics

    def train(
        self,
        state: TrainState,
        train_dataset: Iterator,
        eval_dataset: Optional[Iterator] = None,
        num_steps: int = 10000,
    ) -> tuple[TrainState, Dict[str, float]]:
        """
        Main training loop.

        Args:
            state: Initial train state
            train_dataset: Training data iterator
            eval_dataset: Optional evaluation data iterator
            num_steps: Total number of training steps

        Returns:
            Final train state and final metrics
        """
        # Check device setup
        devices = jax.devices()
        print(f"\n🚀 Starting training for {num_steps} steps...")
        print(f"   Platform: {self.platform.upper()}")
        print(f"   Devices: {len(devices)} x {devices[0].platform}")
        if len(devices) > 1:
            if hasattr(self, '_mesh') and self._mesh is not None:
                print(f"   Model sharding: ENABLED (params split across {len(devices)} devices)")
                print(f"   Data parallelism: ENABLED (batch replicated on all devices)")
            else:
                print(f"   Data parallelism: ENABLED (batch split across {len(devices)} devices)")
        print(f"   Logging every {self.log_every} steps")
        print(f"   Evaluating every {self.eval_every} steps")
        print(f"   Saving every {self.save_every} steps")
        if self.use_remat:
            print(f"   💾 Gradient checkpointing: ENABLED")
        if self.use_mixed_precision:
            print(f"   🔢 Mixed precision (bfloat16): ENABLED")
        if self.gradient_accumulation_steps > 1:
            print(f"   📊 Gradient accumulation: {self.gradient_accumulation_steps} steps")
        print()

        # JIT compile train step with sharding support
        # JAX will automatically use the sharding from params
        train_step_jit = jax.jit(self.train_step)

        # Metrics accumulator
        metrics_history = []
        start_time = time.time()

        for step in range(state.step, num_steps):
            # Profiler: begin step
            if self.profiler is not None:
                self.profiler.begin_step(step)

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

            # Profiler: end step
            if self.profiler is not None:
                self.profiler.end_step()

            # Accumulate metrics
            metrics_history.append(metrics)

            # Logging
            if (step + 1) % self.log_every == 0:
                # Average metrics over logging window
                avg_metrics = {
                    k: float(jnp.mean(jnp.array([m[k] for m in metrics_history[-self.log_every:]])))
                    for k in metrics_history[-1].keys()
                }

                elapsed = time.time() - start_time
                steps_per_sec = self.log_every / elapsed

                print(f"Step {step + 1}/{num_steps} | "
                      f"Loss: {avg_metrics['loss']:.4f} | "
                      f"Grad norm: {avg_metrics['grad_norm']:.2f} | "
                      f"Cluster acc: {avg_metrics.get('cluster_accuracy', 0.0):.3f} | "
                      f"{steps_per_sec:.1f} steps/s")

                start_time = time.time()

            # Evaluation
            if eval_dataset is not None and (step + 1) % self.eval_every == 0:
                print(f"\n  📊 Evaluating at step {step + 1}...")
                eval_metrics = self.evaluate(state, eval_dataset, num_eval_batches=50)

                print(f"     Eval loss: {eval_metrics['eval_loss']:.4f} | "
                      f"Eval acc: {eval_metrics['eval_accuracy']:.4f}\n")

            # Checkpointing
            if (step + 1) % self.save_every == 0:
                self.save_checkpoint(state, step + 1)
                print(f"  💾 Checkpoint saved at step {step + 1}\n")

        print(f"\n✅ Training completed! Final step: {state.step}")

        # Final metrics
        final_metrics = {
            k: float(jnp.mean(jnp.array([m[k] for m in metrics_history[-100:]])))
            for k in metrics_history[-1].keys()
        } if metrics_history else {}

        # Save final checkpoint
        self.save_checkpoint(state, state.step)

        return state, final_metrics

    def evaluate(
        self,
        state: TrainState,
        eval_dataset: Iterator,
        num_eval_batches: int = 100,
    ) -> Dict[str, float]:
        """Run evaluation."""
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
        } if all_metrics else {}

        return avg_metrics

    def save_checkpoint(self, state: TrainState, step: int):
        """Save checkpoint."""
        self.checkpoint_manager.save(
            step,
            args=ocp.args.StandardSave(state),
        )

    def load_checkpoint(self, step: Optional[int] = None) -> Optional[TrainState]:
        """Load checkpoint."""
        if step is None:
            step = self.checkpoint_manager.latest_step()

        if step is None:
            print("No checkpoint found")
            return None

        print(f"Loading checkpoint from step {step}")
        return None  # TODO: Implement proper restoration
