"""Data loading for EfficientIDS Flax."""

# Import from new dataset module (primary)
from .dataset import (
    MovieLensDataset,
    ClusteringInfo,
    create_dataloaders,
)

# Keep old imports for backward compatibility (if movielens_loader exists)
try:
    from .movielens_loader import (
        MovieLensDataLoader,
        create_data_iterator,
    )
except ImportError:
    pass

__all__ = [
    'MovieLensDataset',
    'ClusteringInfo',
    'create_dataloaders',
    # Backward compatibility
    'MovieLensDataLoader',
    'create_data_iterator',
]
