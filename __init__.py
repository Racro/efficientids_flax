"""
EfficientIDS - Pure JAX/Flax Implementation

Minimal, production-ready EfficientIDS for newer JAX versions and GPU generations.
"""

__version__ = "0.1.0"

from .core.embeddings import (
    ItemEmbedding,
    ItemInputAdapter,
    ItemOutputAdapter,
    create_embedding_initializer,
)

from .core.hierarchical import (
    HierarchicalSoftmax,
    ClusteringInfo,
)

from .core.models import (
    EfficientIDSModel,
    SimpleEfficientIDSModel,
)

from .train.optimizer import (
    create_optimizer,
    create_learning_rate_schedule,
)

from .train.trainer import (
    Trainer,
    TrainState,
)

__all__ = [
    # Core components
    "ItemEmbedding",
    "ItemInputAdapter",
    "ItemOutputAdapter",
    "create_embedding_initializer",
    "HierarchicalSoftmax",
    "ClusteringInfo",
    "EfficientIDSModel",
    "SimpleEfficientIDSModel",
    # Training
    "Trainer",
    "TrainState",
    "create_optimizer",
    "create_learning_rate_schedule",
]
