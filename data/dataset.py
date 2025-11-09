"""
Pure JAX/NumPy Data Pipeline for EfficientIDS (Flax Port)

Replaces SeqIO with pure JAX/NumPy data loading for MovieLens.
Compatible with the Flax training loop.

Key Features:
- No SeqIO dependency (pure Python + NumPy)
- Compatible with JAX data loading patterns
- Supports both id_only and text_metadata modes
- Efficient batching and padding
"""

import numpy as np
import joblib
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ClusteringInfo:
    """Clustering information for hierarchical softmax."""
    cluster_assignments: np.ndarray  # [num_items] - item_id → cluster_id
    cluster_indices: np.ndarray      # [num_clusters, max_cluster_size] - cluster → items (padded with -1)
    in_cluster_id: np.ndarray        # [num_items] - item_id → position_in_cluster
    cluster_centers: Optional[np.ndarray] = None  # [num_clusters, embed_dim] - optional

    @classmethod
    def from_pickle(cls, path: str) -> 'ClusteringInfo':
        """Load clustering from pickle file (4 arrays format)."""
        with open(path, 'rb') as f:
            cluster_assignments = pickle.load(f)
            cluster_indices = pickle.load(f)
            in_cluster_id = pickle.load(f)
            cluster_centers = pickle.load(f)

        return cls(
            cluster_assignments=cluster_assignments,
            cluster_indices=cluster_indices,
            in_cluster_id=in_cluster_id,
            cluster_centers=cluster_centers
        )


class MovieLensDataset:
    """
    Pure NumPy/JAX dataset for MovieLens sequences.

    Replaces SeqIO with simple Python iterators and NumPy arrays.
    """

    def __init__(
        self,
        data_dir: str = "./data/ml1m_processed/processed",
        split: str = "train",
        max_seq_len: int = 128,
        batch_size: int = 16,
        shuffle: bool = True,
        mode: str = "id_only",  # or "text_metadata"
        max_sequences: Optional[int] = None,
    ):
        """
        Initialize MovieLens dataset.

        Args:
            data_dir: Path to processed data directory
            split: 'train', 'validation', or 'test'
            max_seq_len: Maximum sequence length
            batch_size: Batch size
            shuffle: Whether to shuffle data
            mode: 'id_only' or 'text_metadata'
            max_sequences: Limit number of sequences (for debugging)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mode = mode
        self.max_sequences = max_sequences

        # Load metadata
        self._load_metadata()

        # Load sequences
        self._load_sequences()

        logger.info(f"✅ Initialized {split} dataset: {len(self.sequences)} sequences")

    def _load_metadata(self):
        """Load preprocessing metadata and clustering."""
        # Load basic metadata
        self.preprocessing_info = joblib.load(self.data_dir / "preprocessing_info.pkl")
        self.dataset_stats = joblib.load(self.data_dir / "dataset_stats.pkl")

        # Load clustering
        self.clustering_info = ClusteringInfo.from_pickle(
            str(self.data_dir / "clustering.pkl")
        )

        # Load item embeddings (for initialization)
        embedding_methods = ['metadata', 'wals', 'random']
        self.item_embeddings = None
        for method in embedding_methods:
            emb_path = self.data_dir / f"item_embeddings_{method}.npy"
            if emb_path.exists():
                self.item_embeddings = np.load(emb_path)
                logger.info(f"Loaded {method} embeddings: {self.item_embeddings.shape}")
                break

        if self.item_embeddings is None:
            # Try default
            default_path = self.data_dir / "item_embeddings.npy"
            if default_path.exists():
                self.item_embeddings = np.load(default_path)

        self.num_items = self.preprocessing_info['num_items']
        self.num_clusters = len(self.clustering_info.cluster_indices)

        logger.info(f"Metadata: {self.num_items} items, {self.num_clusters} clusters")

    def _load_sequences(self):
        """Load user sequences for the split."""
        seq_file = self.data_dir / f"{self.split}_sequences.pkl"

        if not seq_file.exists():
            raise FileNotFoundError(f"Sequences not found: {seq_file}")

        self.sequences = joblib.load(seq_file)

        # Limit sequences if requested
        if self.max_sequences is not None:
            self.sequences = self.sequences[:self.max_sequences]
            logger.info(f"Limited to {len(self.sequences)} sequences")

        logger.info(f"Loaded {len(self.sequences)} sequences from {seq_file}")

    def __len__(self) -> int:
        """Number of batches per epoch."""
        return (len(self.sequences) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        """Iterate over batches (infinite iterator for training)."""
        while True:  # Infinite loop for multiple epochs
            # Shuffle if requested
            indices = np.arange(len(self.sequences))
            if self.shuffle:
                np.random.shuffle(indices)

            # Batch iteration
            for start_idx in range(0, len(self.sequences), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(self.sequences))
                batch_indices = indices[start_idx:end_idx]

                # Collect batch sequences
                batch_sequences = [self.sequences[i] for i in batch_indices]

                # Convert to tensors
                batch = self._collate_batch(batch_sequences)

                yield batch

    def _collate_batch(self, sequences: List[Dict]) -> Dict[str, np.ndarray]:
        """
        Convert list of sequences to batched tensors.

        Args:
            sequences: List of sequence dicts with 'items' field

        Returns:
            Dictionary with batched arrays:
                - item_ids: [batch, seq_len] - Item IDs (padded with 0)
                - targets: [batch, seq_len] - Target items (shifted by 1)
                - weights: [batch, seq_len] - 1.0 for real items, 0.0 for padding
                - cluster_ids: [batch, seq_len] - Cluster IDs for items
                - in_cluster_ids: [batch, seq_len] - Position within cluster
        """
        batch_size = len(sequences)

        # Initialize arrays
        item_ids = np.zeros((batch_size, self.max_seq_len), dtype=np.int32)
        targets = np.zeros((batch_size, self.max_seq_len), dtype=np.int32)
        weights = np.zeros((batch_size, self.max_seq_len), dtype=np.float32)
        cluster_ids = np.zeros((batch_size, self.max_seq_len), dtype=np.int32)
        in_cluster_ids = np.zeros((batch_size, self.max_seq_len), dtype=np.int32)

        for i, seq in enumerate(sequences):
            items = seq['item_ids']  # List of item IDs
            seq_len = min(len(items), self.max_seq_len)

            # Input: items[:-1] (all but last)
            # Target: items[1:] (all but first)
            # This creates the next-item prediction task

            if seq_len > 1:
                # Input items (all but last)
                input_items = items[:seq_len-1]
                item_ids[i, :len(input_items)] = input_items

                # Target items (all but first)
                target_items = items[1:seq_len]
                targets[i, :len(target_items)] = target_items

                # Weights (1.0 for real positions)
                weights[i, :len(target_items)] = 1.0

                # Cluster information
                for j, item_id in enumerate(input_items):
                    cluster_ids[i, j] = self.clustering_info.cluster_assignments[item_id]
                    in_cluster_ids[i, j] = self.clustering_info.in_cluster_id[item_id]

        return {
            'item_ids': item_ids,
            'targets': targets,
            'weights': weights,
            'cluster_ids': cluster_ids,
            'in_cluster_ids': in_cluster_ids,
        }

    def get_clustering_info(self) -> ClusteringInfo:
        """Get clustering information for model."""
        return self.clustering_info

    def get_item_embeddings(self) -> Optional[np.ndarray]:
        """Get item embeddings for initialization."""
        return self.item_embeddings


# ==================== HELPER FUNCTIONS ====================

def create_dataloaders(
    data_dir: str,
    batch_size: int,
    max_seq_len: int = 128,
    mode: str = "id_only",
    max_sequences: Optional[Dict[str, int]] = None,
) -> Tuple[MovieLensDataset, MovieLensDataset, MovieLensDataset]:
    """
    Create train/val/test dataloaders.

    Args:
        data_dir: Path to processed data
        batch_size: Batch size
        max_seq_len: Maximum sequence length
        mode: 'id_only' or 'text_metadata'
        max_sequences: Optional dict with limits per split

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    if max_sequences is None:
        max_sequences = {}

    train_dataset = MovieLensDataset(
        data_dir=data_dir,
        split="train",
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        shuffle=True,
        mode=mode,
        max_sequences=max_sequences.get('train'),
    )

    val_dataset = MovieLensDataset(
        data_dir=data_dir,
        split="val",
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        shuffle=False,
        mode=mode,
        max_sequences=max_sequences.get('validation'),
    )

    test_dataset = MovieLensDataset(
        data_dir=data_dir,
        split="test",
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        shuffle=False,
        mode=mode,
        max_sequences=max_sequences.get('test'),
    )

    return train_dataset, val_dataset, test_dataset


# ==================== TESTING ====================

if __name__ == "__main__":
    """Test the dataset."""
    logging.basicConfig(level=logging.INFO)

    print("Testing MovieLensDataset")
    print("=" * 60)

    # Create dataset
    dataset = MovieLensDataset(
        data_dir="../efficientids/data/ml1m_processed/processed",
        split="train",
        max_seq_len=128,
        batch_size=4,
        shuffle=True,
        max_sequences=100,  # Small for testing
    )

    print(f"\nDataset info:")
    print(f"  Sequences: {len(dataset.sequences)}")
    print(f"  Num items: {dataset.num_items}")
    print(f"  Num clusters: {dataset.num_clusters}")
    print(f"  Batches per epoch: {len(dataset)}")

    # Test iteration
    print(f"\nTesting batch iteration...")
    for i, batch in enumerate(dataset):
        if i >= 2:  # Only show 2 batches
            break

        print(f"\nBatch {i}:")
        print(f"  item_ids shape: {batch['item_ids'].shape}")
        print(f"  targets shape: {batch['targets'].shape}")
        print(f"  weights shape: {batch['weights'].shape}")
        print(f"  First sequence (first 10 items): {batch['item_ids'][0, :10]}")
        print(f"  First targets (first 10): {batch['targets'][0, :10]}")
        print(f"  First weights (first 10): {batch['weights'][0, :10]}")

    print("\n" + "=" * 60)
    print("✅ Dataset test passed!")
