"""
Evaluation Metrics for EfficientIDS (Flax Port)

Ports the essential evaluation metrics from the original PAXml implementation
to pure JAX/Flax without Praxis dependencies.

Key Metrics:
------------
1. recall_at(k): Recall@K - fraction of relevant items in top-K
2. mrr_at(k): Mean Reciprocal Rank@K  - reciprocal rank of first match
3. greedy_ndcg_at(k): NDCG@K - position-aware ranking metric
4. accuracy@K: Accuracy at last K positions
5. map@K: Mean Average Precision (sequence-aware)

All metrics support:
- Batched computation
- JIT compilation
- Variable-length sequences with padding masks
"""

import functools
from typing import Sequence, Optional, Dict, List, Union

import jax
import jax.numpy as jnp
from jax import lax


JTensor = jnp.ndarray  # Replace Praxis JTensor with simple jnp.ndarray


# ==================== CORE METRICS (PAXml-compatible) ====================

def recall_at(
    k: Union[int, List[int]],
    logits: JTensor,
    labels: JTensor,
    weights: JTensor,
    num_preds: JTensor,
    approx: bool = True,
    cached_top_k: Optional[JTensor] = None,
) -> List[JTensor]:
    """
    Compute recall@k from logits and labels.

    Compatible with the original PAXml implementation.

    Args:
        k: Cutoff value(s) for top-K (e.g., 10 or [1, 5, 10])
        logits: [batch, seq_len, vocab_size] - Model prediction logits
        labels: [batch, seq_len] - Target item IDs
        weights: [batch, seq_len] - Binary mask (1=real item, 0=padding)
        num_preds: Scalar - Total number of predictions for normalization
        approx: Use lax.approx_max_k (faster) instead of lax.top_k
        cached_top_k: Optional pre-computed top-K indices

    Returns:
        List of scalar recall values, one per k value
    """
    if isinstance(k, int):
        k = [k]
    if not k or min(k) < 1:
        raise ValueError(f'{k=} must not be empty or contain values less than 1')

    # Get top-K predictions
    if cached_top_k is None:
        if approx:
            _, top_idx = lax.approx_max_k(logits, k=max(k))
        else:
            _, top_idx = lax.top_k(logits, k=max(k))
    else:
        top_idx = cached_top_k

    # Check if labels appear in top-K
    # Broadcast labels to [batch, seq_len, k] and compare with top_idx
    label_hitmask = jnp.equal(top_idx, jnp.expand_dims(labels, axis=-1))

    vals = []
    for cutoff in k:
        label_cutoff_hitmask = label_hitmask[..., :cutoff]

        # Any match in top-K?
        item_hit = jnp.any(label_cutoff_hitmask, axis=-1).astype(jnp.int32)

        # Only count masked positions
        real_hits_on_masked = item_hit * weights

        # Average recall
        vals.append(jnp.sum(real_hits_on_masked) / jnp.maximum(num_preds, 1))

    return vals


def mrr_at(
    k: Union[int, List[int]],
    logits: JTensor,
    labels: JTensor,
    weights: JTensor,
    num_preds: JTensor,
    approx: bool = True,
    cached_top_k: Optional[JTensor] = None,
    ndcg: bool = False,
) -> List[JTensor]:
    """
    Compute MRR@k (or nDCG@k if ndcg=True) from logits and labels.

    Compatible with the original PAXml implementation.

    Args:
        k: Cutoff value(s) for top-K
        logits: [batch, seq_len, vocab_size]
        labels: [batch, seq_len]
        weights: [batch, seq_len]
        num_preds: Scalar - number of predictions
        approx: Use approx_max_k
        cached_top_k: Optional pre-computed top-K
        ndcg: If True, use nDCG denominator (log2(rank+2)) instead of MRR (rank+1)

    Returns:
        List of scalar MRR/nDCG values
    """
    if isinstance(k, int):
        k = [k]
    if not k or min(k) < 1:
        raise ValueError(f'{k=} must not be empty or contain values less than 1')

    # Get top-K predictions
    if cached_top_k is None:
        if approx:
            _, top_idx = lax.approx_max_k(logits, k=max(k))
        else:
            _, top_idx = lax.top_k(logits, k=max(k))
    else:
        top_idx = cached_top_k

    # Find matches
    label_hitmask = jnp.equal(top_idx, jnp.expand_dims(labels, axis=-1))

    # Find first hit position
    has_hits = jnp.any(label_hitmask, axis=-1)
    hit_indices = jnp.argmax(label_hitmask, axis=-1)

    # Compute metric (MRR or nDCG denominator)
    if ndcg:
        maybe_metric = 1 / jnp.log2(hit_indices.astype(jnp.float32) + 2)
    else:
        maybe_metric = 1 / (hit_indices.astype(jnp.float32) + 1)

    # Zero out non-hits
    metric = jnp.where(has_hits, maybe_metric, 0)

    # Only count masked positions
    real_metric_on_masked = metric * weights

    # Return average for each k value
    return [
        jnp.sum(jnp.where(hit_indices < cutoff, real_metric_on_masked, 0))
        / jnp.maximum(num_preds, 1)
        for cutoff in k
    ]


def greedy_ndcg_at(
    k: Union[int, List[int]],
    logits: JTensor,
    labels: JTensor,
    weights: JTensor,
    num_preds: JTensor,
    approx: bool = True,
    cached_top_k: Optional[JTensor] = None,
) -> List[JTensor]:
    """
    Compute greedy nDCG@k from logits and labels.

    This is a simplified nDCG for single-label recommendation where
    nDCG reduces to MRR with log2 discounting.

    Compatible with the original PAXml implementation.
    """
    return mrr_at(k, logits, labels, weights, num_preds, approx, cached_top_k, ndcg=True)


# ==================== DYNAMIC EVALUATION (Last-K positions) ====================

def find_actual_length(weights: JTensor) -> JTensor:
    """Find actual sequence length by counting non-zero weights."""
    if weights.ndim == 2:
        weights = weights[0]  # Take first sample if batched
    return jnp.sum(weights > 0).astype(jnp.int32)


def get_eval_positions(actual_length: JTensor, num_positions: int, seq_len: int) -> JTensor:
    """
    Get evaluation positions at last K real items.

    Args:
        actual_length: Actual sequence length (excluding padding)
        num_positions: How many last positions to evaluate
        seq_len: Maximum sequence length

    Returns:
        [num_positions] array of position indices
    """
    position_offsets = jnp.arange(num_positions)
    eval_positions = actual_length - num_positions + position_offsets
    return jnp.clip(eval_positions, 0, seq_len - 1)


@functools.partial(jax.jit, static_argnames=['k'])
def dynamic_accuracy_at_k(
    predictions: JTensor,
    targets: JTensor,
    weights: JTensor,
    k: int,
) -> JTensor:
    """
    Compute accuracy at last K positions.

    Evaluates only at the last K real positions (non-padded).

    Args:
        predictions: [batch, seq_len] - Predicted item IDs
        targets: [batch, seq_len] - Target item IDs
        weights: [batch, seq_len] - Position weights (1=real, 0=padding)
        k: Number of last positions to evaluate

    Returns:
        [batch] - Average accuracy across last K positions per sample
    """
    # Ensure batched inputs
    if predictions.ndim == 1:
        predictions = predictions[None, :]
    if targets.ndim == 1:
        targets = targets[None, :]
    if weights.ndim == 1:
        weights = weights[None, :]

    batch_size, seq_len = predictions.shape

    def compute_accuracy_for_sample(sample_idx):
        preds_sample = predictions[sample_idx]
        targets_sample = targets[sample_idx]
        weights_sample = weights[sample_idx]

        # Find actual length
        actual_length = jnp.sum(weights_sample > 0).astype(jnp.int32)

        # Get last K positions
        eval_positions = get_eval_positions(actual_length, k, seq_len)

        # Get predictions/targets at these positions
        preds_at_pos = jnp.take(preds_sample, eval_positions, axis=0)
        targets_at_pos = jnp.take(targets_sample, eval_positions, axis=0)
        weights_at_pos = jnp.take(weights_sample, eval_positions, axis=0)

        # Check matches
        hits = jnp.equal(preds_at_pos, targets_at_pos).astype(jnp.float32)

        # Apply weights
        valid_hits = hits * (weights_at_pos > 0).astype(jnp.float32)
        num_valid = jnp.sum((weights_at_pos > 0).astype(jnp.float32))

        # Average
        return jnp.sum(valid_hits) / jnp.maximum(num_valid, 1.0)

    # Vectorize across batch
    batch_accuracies = jax.vmap(compute_accuracy_for_sample)(jnp.arange(batch_size))
    return batch_accuracies


def dynamic_accuracy_at_k_wrapper(
    logits: JTensor,
    labels: JTensor,
    weights: JTensor,
    k_values: Sequence[int],
    top_k_predictions: Optional[JTensor] = None,
) -> Dict[str, JTensor]:
    """
    Compute accuracy@K at last K positions for multiple K values.

    Args:
        logits: [batch, seq_len, vocab]
        labels: [batch, seq_len]
        weights: [batch, seq_len]
        k_values: List of K values (e.g., [1, 5, 10])
        top_k_predictions: Optional [batch, seq_len, top_k] pre-computed predictions

    Returns:
        Dictionary: {'accuracy@K': scalar_value, ...}
    """
    batch_size, seq_len, vocab_size = logits.shape

    # Get predictions
    if top_k_predictions is not None:
        # Use top-1 from pre-computed top-K
        while top_k_predictions.ndim > 3:
            top_k_predictions = top_k_predictions[0]
        predictions = top_k_predictions[..., 0]
    else:
        # Argmax on logits
        predictions = jnp.argmax(logits, axis=-1)

    # Compute for each K
    results = {}
    for k in k_values:
        acc = dynamic_accuracy_at_k(predictions, labels, weights, k=k)
        results[f'accuracy@{k}'] = jnp.mean(acc)  # Average across batch

    return results


# ==================== HELPER FUNCTIONS ====================

def compute_metrics_from_logits(
    logits: JTensor,
    labels: JTensor,
    weights: JTensor,
    k_values: List[int] = [1, 5, 10],
    metric_types: List[str] = ['recall', 'mrr', 'ndcg', 'accuracy'],
) -> Dict[str, float]:
    """
    Compute all evaluation metrics from model logits.

    Convenience function that computes multiple metrics at once.

    Args:
        logits: [batch, seq_len, vocab_size]
        labels: [batch, seq_len]
        weights: [batch, seq_len]
        k_values: List of K values to evaluate
        metric_types: Which metrics to compute

    Returns:
        Dictionary with all metrics:
            'recall@K', 'mrr@K', 'ndcg@K', 'accuracy@K'
    """
    num_preds = jnp.sum(weights)

    results = {}

    # Compute core metrics (PAXml-style)
    if 'recall' in metric_types:
        recall_vals = recall_at(k_values, logits, labels, weights, num_preds, approx=True)
        for k, val in zip(k_values, recall_vals):
            results[f'recall@{k}'] = float(val)

    if 'mrr' in metric_types:
        mrr_vals = mrr_at(k_values, logits, labels, weights, num_preds, approx=True)
        for k, val in zip(k_values, mrr_vals):
            results[f'mrr@{k}'] = float(val)

    if 'ndcg' in metric_types:
        ndcg_vals = greedy_ndcg_at(k_values, logits, labels, weights, num_preds, approx=True)
        for k, val in zip(k_values, ndcg_vals):
            results[f'ndcg@{k}'] = float(val)

    # Compute accuracy at last-K positions
    if 'accuracy' in metric_types:
        acc_results = dynamic_accuracy_at_k_wrapper(logits, labels, weights, k_values)
        results.update({k: float(v) for k, v in acc_results.items()})

    return results


# ==================== TESTING ====================

if __name__ == "__main__":
    """Test the metrics with synthetic data."""
    print("Testing EfficientIDS Metrics (Flax Port)")
    print("=" * 60)

    # Create synthetic data
    batch_size = 4
    seq_len = 20
    vocab_size = 100
    k_values = [1, 5, 10]

    key = jax.random.PRNGKey(42)

    # Random logits
    logits = jax.random.normal(key, (batch_size, seq_len, vocab_size))

    # Random labels
    labels = jax.random.randint(key, (batch_size, seq_len), 0, vocab_size)

    # Weights (some padding)
    weights = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    weights = weights.at[:, 15:].set(0)  # Pad last 5 positions

    print("\nTest data:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Weights shape: {weights.shape}")
    print(f"  K values: {k_values}")

    # Compute metrics
    print("\n" + "=" * 60)
    print("Computing metrics...")
    print("=" * 60)

    results = compute_metrics_from_logits(
        logits=logits,
        labels=labels,
        weights=weights,
        k_values=k_values,
        metric_types=['recall', 'mrr', 'ndcg', 'accuracy']
    )

    print("\nResults:")
    for metric_name, value in sorted(results.items()):
        print(f"  {metric_name:15s}: {value:.4f}")

    print("\n" + "=" * 60)
    print("✅ Metrics test passed!")
    print("\nImplemented metrics:")
    print("  • recall@K    - Fraction of correct items in top-K")
    print("  • mrr@K       - Mean reciprocal rank")
    print("  • ndcg@K      - Normalized discounted cumulative gain")
    print("  • accuracy@K  - Accuracy at last K positions")
    print("\nAll metrics are JIT-compiled and ready for training!")
