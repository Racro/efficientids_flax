"""
Configuration System for EfficientIDS (Flax Port)

Provides dataclass-based configs that replace PAXml experiment configs.
Compatible with the original PAXml configuration parameters.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Literal
import jax.numpy as jnp


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    # Item vocabulary
    num_items: int  # Total number of items (e.g., 3261 for ML-1M)
    num_clusters: int  # Number of clusters for hierarchical softmax (e.g., 100)
    item_embedding_dim: int = 384  # Item embedding dimension

    # Model dimensions
    model_dims: int = 512  # Hidden dimension for internal processing
    hidden_dims: Optional[int] = None  # If None, defaults to model_dims * 4

    # Hierarchical softmax
    use_hierarchical_softmax: bool = True  # Use hierarchical (True) or full softmax (False)
    use_correction: bool = True  # Apply correction term in hierarchical softmax

    # Item embeddings
    trainable_item_embeddings: bool = True
    trainable_cluster_embeddings: bool = True

    # Optional: Pretrained LM
    pretrained_lm_name: Optional[str] = None  # e.g., "Qwen/Qwen2.5-0.5B"
    freeze_lm: bool = True  # Freeze pretrained LM weights


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Training steps
    max_steps: int = 10000
    warmup_steps: int = 1000

    # Batch size
    batch_size: int = 16  # Per-device batch size
    eval_batch_size: int = 32

    # Sequence length
    max_seq_len: int = 128  # Maximum sequence length

    # Learning rate
    learning_rate: float = 1e-4
    schedule_type: Literal['constant', 'linear', 'cosine'] = 'cosine'

    # Optimizer
    optimizer_type: Literal['adam', 'adamw', 'sgd'] = 'adamw'
    weight_decay: float = 0.01
    clip_grad_norm: float = 1.0

    # Adam/AdamW parameters
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    # Logging and evaluation
    log_every: int = 100
    eval_every: int = 1000
    save_every: int = 1000

    # Random seed
    seed: int = 42


@dataclass
class DataConfig:
    """Data configuration."""

    # Data directory
    data_dir: str = "./data/ml1m_processed/processed"

    # Training mode
    mode: Literal['id_only', 'text_metadata'] = 'id_only'

    # Splits
    train_split: str = "train"
    val_split: str = "val"  # Changed from "validation" to match file names
    test_split: str = "test"

    # Limits (for debugging)
    max_train_sequences: Optional[int] = None
    max_val_sequences: Optional[int] = None
    max_test_sequences: Optional[int] = None

    # Embedding initialization
    embedding_init_method: Literal['random', 'metadata', 'wals'] = 'metadata'


@dataclass
class EvalConfig:
    """Evaluation configuration."""

    # Metrics to compute
    k_values: List[int] = field(default_factory=lambda: [1, 5, 10])
    metric_types: List[str] = field(default_factory=lambda: ['recall', 'mrr', 'ndcg', 'accuracy'])

    # Evaluation settings
    num_eval_batches: int = 100  # Number of batches for periodic evaluation
    compute_full_metrics: bool = True  # Compute all metrics during evaluation


@dataclass
class EfficientIDSConfig:
    """Complete EfficientIDS configuration.

    Combines model, training, data, and eval configs.
    """

    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    eval: EvalConfig

    # Checkpoint and logging
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    experiment_name: str = "efficientids"

    def __post_init__(self):
        """Validate configuration."""
        # Ensure consistency
        assert self.model.num_items > 0, "num_items must be positive"
        assert self.model.num_clusters > 0, "num_clusters must be positive"
        assert self.training.max_steps > 0, "max_steps must be positive"
        assert self.training.batch_size > 0, "batch_size must be positive"


# ==================== PRESET CONFIGURATIONS ====================

def get_qwen_config(
    num_items: int = 3261,
    num_clusters: int = 100,
    max_seq_len: int = 128,
    batch_size: int = 16,
    max_steps: int = 10000,
) -> EfficientIDSConfig:
    """
    Configuration for Qwen 0.6B model (similar to original PAXml config).

    Args:
        num_items: Number of items in catalog
        num_clusters: Number of clusters
        max_seq_len: Maximum sequence length
        batch_size: Batch size
        max_steps: Training steps

    Returns:
        Complete configuration
    """
    return EfficientIDSConfig(
        model=ModelConfig(
            num_items=num_items,
            num_clusters=num_clusters,
            item_embedding_dim=384,
            model_dims=512,
            use_hierarchical_softmax=True,
            use_correction=True,
            pretrained_lm_name=None,  # Start without pretrained LM
        ),
        training=TrainingConfig(
            max_steps=max_steps,
            warmup_steps=1000,
            batch_size=batch_size,
            eval_batch_size=32,
            max_seq_len=max_seq_len,
            learning_rate=1e-4,
            schedule_type='cosine',
            optimizer_type='adamw',
            weight_decay=0.01,
            clip_grad_norm=1.0,
            log_every=100,
            eval_every=1000,
            save_every=1000,
        ),
        data=DataConfig(
            data_dir="./data/ml1m_processed/processed",
            mode='id_only',
            embedding_init_method='metadata',
        ),
        eval=EvalConfig(
            k_values=[1, 5, 10],
            metric_types=['recall', 'mrr', 'accuracy'],
            num_eval_batches=100,
            compute_full_metrics=True,
        ),
        checkpoint_dir="./checkpoints/qwen",
        log_dir="./logs/qwen",
        experiment_name="qwen_128",
    )


def get_llama_config(
    num_items: int = 3261,
    num_clusters: int = 100,
    max_seq_len: int = 128,
    batch_size: int = 16,
    max_steps: int = 15000,
) -> EfficientIDSConfig:
    """
    Configuration for Llama 1B model (similar to original PAXml config).
    """
    return EfficientIDSConfig(
        model=ModelConfig(
            num_items=num_items,
            num_clusters=num_clusters,
            item_embedding_dim=512,
            model_dims=2048,  # Llama 1B hidden size
            use_hierarchical_softmax=True,
            use_correction=True,
            pretrained_lm_name=None,  # Add Llama integration later
            freeze_lm=True,
        ),
        training=TrainingConfig(
            max_steps=max_steps,
            warmup_steps=1500,
            batch_size=batch_size,
            eval_batch_size=32,
            max_seq_len=max_seq_len,
            learning_rate=1e-4,
            schedule_type='cosine',
            optimizer_type='adamw',
            weight_decay=0.01,
            clip_grad_norm=1.0,
        ),
        data=DataConfig(
            data_dir="./data/ml1m_processed/processed",
            mode='id_only',
            embedding_init_method='wals',  # WALS often works best for Llama
        ),
        eval=EvalConfig(
            k_values=[1, 5, 10],
            metric_types=['recall', 'mrr', 'accuracy'],
            num_eval_batches=100,
        ),
        checkpoint_dir="./checkpoints/llama",
        log_dir="./logs/llama",
        experiment_name="llama_128",
    )


def get_gemma_config(
    num_items: int = 3261,
    num_clusters: int = 100,
    max_seq_len: int = 256,
    batch_size: int = 8,
    max_steps: int = 10000,
) -> EfficientIDSConfig:
    """
    Configuration for Gemma 2B pretrained model.

    Gemma 2B specs:
    - hidden_size: 2048
    - num_layers: 18
    - num_heads: 8
    """
    return EfficientIDSConfig(
        model=ModelConfig(
            num_items=num_items,
            num_clusters=num_clusters,
            item_embedding_dim=384,
            model_dims=2048,  # Gemma 2B hidden size
            use_hierarchical_softmax=True,
            pretrained_lm_name="gemma-2b",
            freeze_lm=True,
        ),
        training=TrainingConfig(
            max_steps=max_steps,
            warmup_steps=1000,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            learning_rate=5e-5,  # Lower LR for pretrained
            schedule_type='cosine',
            optimizer_type='adamw',
            weight_decay=0.01,
            clip_grad_norm=1.0,
            log_every=50,
            eval_every=500,
            save_every=1000,
        ),
        data=DataConfig(
            data_dir="./data/ml1m_processed/processed",
            mode='id_only',
            embedding_init_method='metadata',
        ),
        eval=EvalConfig(
            k_values=[1, 5, 10],
            metric_types=['recall', 'mrr', 'ndcg', 'accuracy'],
            num_eval_batches=100,
        ),
        checkpoint_dir="./checkpoints/gemma",
        log_dir="./logs/gemma",
        experiment_name="gemma_2b",
    )


def get_debug_config() -> EfficientIDSConfig:
    """
    Minimal configuration for debugging and testing.
    """
    return EfficientIDSConfig(
        model=ModelConfig(
            num_items=3261,  # Use real num_items to match data
            num_clusters=100,  # Use real num_clusters to match data
            item_embedding_dim=64,
            model_dims=128,
        ),
        training=TrainingConfig(
            max_steps=200,
            warmup_steps=50,
            batch_size=4,
            max_seq_len=32,
            learning_rate=1e-3,
            log_every=20,
            eval_every=50,
            save_every=999999,  # Disable checkpointing for debugging
        ),
        data=DataConfig(
            data_dir="./data/ml1m_processed/processed",
            max_train_sequences=100,
            max_val_sequences=20,
        ),
        eval=EvalConfig(
            k_values=[1, 5],
            num_eval_batches=5,
        ),
        checkpoint_dir="/tmp/efficientids_debug",
        log_dir="/tmp/efficientids_debug/logs",
        experiment_name="debug",
    )


# ==================== TESTING ====================

if __name__ == "__main__":
    """Test configurations."""
    print("Testing EfficientIDS Configurations")
    print("=" * 60)

    # Test Qwen config
    print("\n1. Qwen Configuration:")
    qwen_cfg = get_qwen_config()
    print(f"  Model: {qwen_cfg.model.num_items} items, {qwen_cfg.model.num_clusters} clusters")
    print(f"  Training: {qwen_cfg.training.max_steps} steps, batch size {qwen_cfg.training.batch_size}")
    print(f"  Data: {qwen_cfg.data.mode} mode, {qwen_cfg.data.embedding_init_method} init")
    print(f"  Eval: k={qwen_cfg.eval.k_values}")

    # Test Llama config
    print("\n2. Llama Configuration:")
    llama_cfg = get_llama_config()
    print(f"  Model: {llama_cfg.model.num_items} items, {llama_cfg.model.model_dims} dims")
    print(f"  Training: {llama_cfg.training.max_steps} steps")

    # Test debug config
    print("\n3. Debug Configuration:")
    debug_cfg = get_debug_config()
    print(f"  Model: {debug_cfg.model.num_items} items (debug size)")
    print(f"  Training: {debug_cfg.training.max_steps} steps (quick test)")

    print("\n" + "=" * 60)
    print("✅ All configurations created successfully!")
    print("\nConfigurations are compatible with original PAXml parameters:")
    print("  • MODEL_DIMS → model.model_dims")
    print("  • PERCORE_BATCH_SIZE → training.batch_size")
    print("  • MAX_STEPS → training.max_steps")
    print("  • NUM_ITEM_CLUSTERS → model.num_clusters")
    print("  • ITEM_EMBEDDING_SIZE → model.item_embedding_dim")
