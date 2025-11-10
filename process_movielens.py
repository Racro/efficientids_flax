#!/usr/bin/env python3
"""
MovieLens Data Processing Pipeline for EfficientIDS

This script downloads, processes, and prepares MovieLens datasets (ML-1M or ML-20M)
for training with the EfficientIDS recommendation system.

Main Components:
----------------
1. MovieLensProcessor: Main data processing class
   - download_data(): Downloads and extracts MovieLens data
   - load_raw_data(): Loads ratings, movies, tags
   - filter_data(): Filters by minimum interactions
   - create_item_embeddings(): Generates embeddings from metadata (genres, titles)
   - create_clusters(): K-means/spectral/hierarchical clustering
   - create_sequences(): Builds user interaction sequences
   - split_sequences(): Train/val/test splitting with data leakage prevention
   - save_processed_data(): Saves all outputs as .pkl files

Processing Pipeline:
--------------------
1. Download → 2. Load Raw Data → 3. Filter → 4. Generate Embeddings
   → 5. Cluster Items → 6. Create Sequences → 7. Split → 8. Save

Dataset Support:
----------------
- ML-1M: ~1M ratings, 3,260 movies, 6,040 users
- ML-20M: ~20M ratings, 27K movies, 138K users
(Auto-detected from --output_dir name: "ml1m" → ML-1M, else ML-20M)

Output Files (in <output_dir>/processed/):
-------------------------------------------
- train_sequences.pkl, val_sequences.pkl, test_sequences.pkl
- item_embeddings.npy (128d vectors from genres + titles)
- clustering.pkl (4-part format: assignments, indices, in_cluster_id, centers)
- preprocessing_info.pkl (metadata, mappings, params)
- dataset_stats.pkl (statistics)

Clustering Methods:
-------------------
- kmeans: Fast, content-based (default)
- spectral: Best quality, slower
- hierarchical: Tree structure
- frequency: Popularity-based

Data Split Strategy (Leave-One-Out):
------------------------------------
- Train: Items [0...N-12] with sliding windows (stride=64) for sequences > 128
- Val/Test: Last 128 items for sequences >= 128, full sequence for < 128 (Val == Test)
- Holds out last 12 items from training to prevent leakage
- Single mode (no mode-specific files)

Usage Examples:
---------------
    # ML-1M with 100 clusters (recommended)
    python process_movielens.py \\
        --output_dir ./data/ml1m_processed \\
        --num_clusters 100 \\
        --clustering_method kmeans

    # ML-20M with spectral clustering
    python process_movielens.py \\
        --output_dir ./data/ml20m_processed \\
        --num_clusters 256 \\
        --clustering_method spectral

Arguments:
----------
--output_dir: Output directory path (auto-detects ML-1M vs ML-20M)
--num_clusters: Number of item clusters (default: 100)
--embedding_dim: Embedding dimension (default: 128)
--clustering_method: kmeans/spectral/hierarchical/frequency (default: kmeans)
--min_user_interactions: Min interactions per user (default: 20 for ML-20M, 10 for ML-1M)
--min_item_interactions: Min interactions per item (default: 10 for ML-20M, 5 for ML-1M)

Note: Uses leave-one-out split strategy (single mode, no mode-specific files)
"""

import argparse
import sys
import zipfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.preprocessing import MultiLabelBinarizer
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MovieLensProcessor:
    """Main processor for MovieLens 20M dataset."""

    def __init__(self, output_dir: str = "./data/ml20m_processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Auto-detect dataset type from output directory name
        if "ml1m" in str(output_dir).lower():
            self.dataset_type = "ml-1m"
            self.dataset_url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
            self.dataset_size_mb = 6  # ML-1M is ~6MB
        else:
            self.dataset_type = "ml-20m"
            self.dataset_url = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"
            self.dataset_size_mb = 240  # ML-20M is ~240MB

        # Processing parameters - adjust for dataset size
        if self.dataset_type == "ml-1m":
            # ML-1M is smaller, use lower thresholds
            self.min_user_interactions = 10
            self.min_item_interactions = 5
        else:
            # ML-20M is larger, use higher thresholds
            self.min_user_interactions = 20
            self.min_item_interactions = 10
        self.max_sequence_length = 128
        self.test_split_ratio = 0.1
        self.val_split_ratio = 0.1

        # Paths
        self.raw_data_dir = self.output_dir / "raw"
        self.processed_data_dir = self.output_dir / "processed"
        self.raw_data_dir.mkdir(exist_ok=True)
        self.processed_data_dir.mkdir(exist_ok=True)

    def download_data(self):
        """Download MovieLens dataset."""
        zip_filename = f"{self.dataset_type}.zip"
        zip_path = self.raw_data_dir / zip_filename

        if zip_path.exists():
            logger.info(f"📁 MovieLens {self.dataset_type.upper()} already downloaded")
            return

        logger.info(f"⬇️ Downloading MovieLens {self.dataset_type.upper()} dataset...")
        logger.info(f"   URL: {self.dataset_url}")
        logger.info(f"   This may take a few minutes (~{self.dataset_size_mb}MB)...")

        try:
            urllib.request.urlretrieve(self.dataset_url, zip_path)
            logger.info("✅ Download completed successfully")

            # Extract the zip file
            logger.info("📂 Extracting dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.raw_data_dir)
            logger.info("✅ Extraction completed")

        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            if zip_path.exists():
                zip_path.unlink()
            raise

    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load raw MovieLens data files."""
        ml_dir = self.raw_data_dir / self.dataset_type

        logger.info("📊 Loading raw data files...")

        # Load ratings - different file formats for ML-1M vs ML-20M
        if self.dataset_type == "ml-1m":
            # ML-1M uses :: separator and different column names
            ratings_path = ml_dir / "ratings.dat"
            ratings = pd.read_csv(ratings_path, sep='::', engine='python',
                                names=['userId', 'movieId', 'rating', 'timestamp'])

            # Load movies
            movies_path = ml_dir / "movies.dat"
            movies = pd.read_csv(movies_path, sep='::', engine='python',
                               names=['movieId', 'title', 'genres'], encoding='latin-1')

            # Load users (optional for ML-1M)
            users_path = ml_dir / "users.dat"
            tags = pd.read_csv(users_path, sep='::', engine='python',
                              names=['userId', 'gender', 'age', 'occupation', 'zipCode']) if users_path.exists() else pd.DataFrame()
        else:
            # ML-20M uses CSV format
            ratings_path = ml_dir / "ratings.csv"
            ratings = pd.read_csv(ratings_path)

            movies_path = ml_dir / "movies.csv"
            movies = pd.read_csv(movies_path)

            # Load tags (optional)
            tags_path = ml_dir / "tags.csv"
            tags = pd.read_csv(tags_path) if tags_path.exists() else pd.DataFrame()

        logger.info(f"   Loaded {len(ratings):,} ratings")
        logger.info(f"   Loaded {len(movies):,} movies")
        logger.info(f"   Loaded {len(tags):,} additional records")

        return ratings, movies, tags

    def filter_data(self, ratings: pd.DataFrame, movies: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """Filter data based on minimum interaction thresholds."""
        logger.info("🔍 Filtering data...")

        original_users = ratings['userId'].nunique()
        original_items = ratings['movieId'].nunique()
        original_ratings = len(ratings)

        # Filter users with minimum interactions
        user_counts = ratings['userId'].value_counts()
        valid_users = user_counts[user_counts >= self.min_user_interactions].index
        ratings_filtered = ratings[ratings['userId'].isin(valid_users)]

        # Filter items with minimum interactions
        item_counts = ratings_filtered['movieId'].value_counts()
        valid_items = item_counts[item_counts >= self.min_item_interactions].index
        ratings_filtered = ratings_filtered[ratings_filtered['movieId'].isin(valid_items)]

        # Filter movies to match valid items
        movies_filtered = movies[movies['movieId'].isin(valid_items)]

        # Create ID mappings (0-indexed)
        unique_users = sorted(ratings_filtered['userId'].unique())
        unique_items = sorted(ratings_filtered['movieId'].unique())

        user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
        item_to_idx = {item_id: idx for idx, item_id in enumerate(unique_items)}
        idx_to_user = {idx: user_id for user_id, idx in user_to_idx.items()}
        idx_to_item = {idx: item_id for item_id, idx in item_to_idx.items()}

        # Apply mappings
        ratings_filtered['user_idx'] = ratings_filtered['userId'].map(user_to_idx)
        ratings_filtered['item_idx'] = ratings_filtered['movieId'].map(item_to_idx)
        movies_filtered['item_idx'] = movies_filtered['movieId'].map(item_to_idx)

        filter_stats = {
            'original_users': original_users,
            'original_items': original_items,
            'original_ratings': original_ratings,
            'filtered_users': len(unique_users),
            'filtered_items': len(unique_items),
            'filtered_ratings': len(ratings_filtered),
            'user_to_idx': user_to_idx,
            'item_to_idx': item_to_idx,
            'idx_to_user': idx_to_user,
            'idx_to_item': idx_to_item,
        }

        logger.info(f"   Users: {original_users:,} → {len(unique_users):,}")
        logger.info(f"   Items: {original_items:,} → {len(unique_items):,}")
        logger.info(f"   Ratings: {original_ratings:,} → {len(ratings_filtered):,}")

        return ratings_filtered, movies_filtered, filter_stats

    def create_item_embeddings(self, movies: pd.DataFrame, embedding_dim: int = 128) -> np.ndarray:
        """Create item embeddings from movie metadata."""
        logger.info("🎬 Creating item embeddings...")

        # Parse genres
        movies['genres_list'] = movies['genres'].str.split('|')

        # Genre encoding
        mlb = MultiLabelBinarizer()
        genre_features = mlb.fit_transform(movies['genres_list'])
        logger.info(f"   Genre features: {genre_features.shape[1]} dimensions")

        # Title TF-IDF features
        tfidf = TfidfVectorizer(max_features=100, stop_words='english', lowercase=True)
        title_features = tfidf.fit_transform(movies['title'].fillna('')).toarray()
        logger.info(f"   Title features: {title_features.shape[1]} dimensions")

        # Combine features
        combined_features = np.hstack([genre_features, title_features])
        logger.info(f"   Combined features: {combined_features.shape[1]} dimensions")

        # Reduce dimensionality if needed
        if combined_features.shape[1] > embedding_dim:
            svd = TruncatedSVD(n_components=embedding_dim, random_state=42)
            item_embeddings = svd.fit_transform(combined_features)
            logger.info(f"   Reduced to {embedding_dim} dimensions via SVD")
        else:
            item_embeddings = combined_features

        # Normalize embeddings
        from sklearn.preprocessing import normalize
        item_embeddings = normalize(item_embeddings, norm='l2')

        logger.info(f"✅ Created metadata embeddings: {item_embeddings.shape}")
        return item_embeddings.astype(np.float32)

    def create_random_embeddings(self, num_items: int, embedding_dim: int = 128) -> np.ndarray:
        """Create random item embeddings (Xavier/Glorot initialization).

        Args:
            num_items: Number of items
            embedding_dim: Embedding dimensionality

        Returns:
            Random embeddings normalized to unit length
        """
        logger.info("🎲 Creating random embeddings (Xavier initialization)...")

        # Xavier/Glorot initialization
        limit = np.sqrt(6.0 / (num_items + embedding_dim))
        embeddings = np.random.uniform(-limit, limit, size=(num_items, embedding_dim))

        # Normalize to unit length
        from sklearn.preprocessing import normalize
        embeddings = normalize(embeddings, norm='l2', axis=1)

        logger.info(f"✅ Created random embeddings: {embeddings.shape}")
        return embeddings.astype(np.float32)

    def create_wals_embeddings(self, ratings: pd.DataFrame, num_items: int,
                               embedding_dim: int = 128) -> np.ndarray:
        """Create WALS (Weighted Alternating Least Squares) embeddings from user-item interactions.

        Uses matrix factorization (SVD) on the user-item interaction matrix to create
        collaborative filtering embeddings that capture behavioral patterns.

        Args:
            ratings: DataFrame with 'user_idx' and 'item_idx' columns
            num_items: Number of items
            embedding_dim: Embedding dimensionality

        Returns:
            WALS embeddings from collaborative filtering
        """
        logger.info("🤝 Creating WALS embeddings (collaborative filtering)...")

        # Build user-item interaction matrix
        logger.info("   Building interaction matrix...")
        unique_users = ratings['user_idx'].unique()
        num_users = len(unique_users)

        row_indices = ratings['user_idx'].values
        col_indices = ratings['item_idx'].values
        data = np.ones(len(ratings))

        from scipy import sparse
        interaction_matrix = sparse.csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(num_users, num_items)
        )

        logger.info(f"   Interaction matrix: {interaction_matrix.shape}, "
                   f"density={interaction_matrix.nnz / (num_users * num_items):.4f}")

        # Apply SVD (matrix factorization)
        logger.info(f"   Applying SVD for {embedding_dim} components...")
        from scipy.sparse.linalg import svds

        k = min(embedding_dim, min(interaction_matrix.shape) - 1)
        try:
            U, sigma, Vt = svds(interaction_matrix, k=k)

            # Item embeddings are rows of V (transpose of Vt), scaled by singular values
            item_embeddings = Vt.T * sigma

            # Pad or truncate to exact embedding_dim
            if item_embeddings.shape[1] < embedding_dim:
                padding = np.zeros((item_embeddings.shape[0], embedding_dim - item_embeddings.shape[1]))
                item_embeddings = np.hstack([item_embeddings, padding])
            else:
                item_embeddings = item_embeddings[:, :embedding_dim]

            # Normalize to unit length
            from sklearn.preprocessing import normalize
            item_embeddings = normalize(item_embeddings, norm='l2', axis=1)

            logger.info(f"✅ Created WALS embeddings: {item_embeddings.shape}")

        except Exception as e:
            logger.error(f"   ❌ SVD failed: {e}")
            logger.warning("   Falling back to random embeddings for WALS")
            item_embeddings = self.create_random_embeddings(num_items, embedding_dim)

        return item_embeddings.astype(np.float32)

    def create_clusters(self, item_embeddings: np.ndarray, num_clusters: int = 256,
                       clustering_method: str = 'kmeans', item_popularity: Optional[np.ndarray] = None) -> Dict:
        """Create hierarchical clusters of items.

        Args:
            item_embeddings: Item embeddings [num_items, embedding_dim]
            num_clusters: Number of clusters to create
            clustering_method: One of 'kmeans', 'spectral', 'hierarchical', 'frequency'
            item_popularity: Item popularity counts (required for 'frequency' method)

        Returns:
            Dictionary with clustering information
        """
        logger.info(f"🎯 Creating {num_clusters} item clusters using {clustering_method.upper()} method...")

        num_items = len(item_embeddings)
        cluster_centers = None

        if clustering_method == 'kmeans':
            # K-means clustering (default, fast and scalable)
            logger.info("   Using K-Means clustering on content embeddings")
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(item_embeddings)
            cluster_centers = kmeans.cluster_centers_

        elif clustering_method == 'spectral':
            # Spectral clustering (best quality, slower)
            logger.info("   Using Spectral clustering (high quality, may take a few minutes...)")
            logger.info(f"   Computing on {num_items} items with {item_embeddings.shape[1]}d embeddings")
            spectral = SpectralClustering(
                n_clusters=num_clusters,
                affinity='nearest_neighbors',  # More scalable than 'rbf'
                n_neighbors=min(50, num_items // 10),  # Adaptive neighbors
                random_state=42,
                n_jobs=-1  # Use all CPU cores
            )
            cluster_labels = spectral.fit_predict(item_embeddings)
            # Spectral clustering doesn't provide cluster centers, compute them manually
            cluster_centers = np.array([
                item_embeddings[cluster_labels == i].mean(axis=0)
                for i in range(num_clusters)
            ])

        elif clustering_method == 'hierarchical':
            # Hierarchical/Agglomerative clustering (creates tree structure)
            logger.info("   Using Hierarchical (Agglomerative) clustering")
            logger.info(f"   Computing on {num_items} items (may take a few minutes...)")
            hierarchical = AgglomerativeClustering(
                n_clusters=num_clusters,
                linkage='ward'  # Minimizes variance (good for embeddings)
            )
            cluster_labels = hierarchical.fit_predict(item_embeddings)
            # Compute cluster centers manually
            cluster_centers = np.array([
                item_embeddings[cluster_labels == i].mean(axis=0)
                for i in range(num_clusters)
            ])

        elif clustering_method == 'frequency':
            # Frequency-based clustering (popular items in same clusters)
            logger.info("   Using Frequency-based clustering (popularity-based)")
            if item_popularity is None:
                raise ValueError("item_popularity required for frequency-based clustering")

            # Sort items by popularity (most popular first)
            popularity_order = np.argsort(item_popularity)[::-1]

            # Assign items to clusters in round-robin fashion
            # This ensures each cluster has mix of popularity levels
            cluster_labels = np.zeros(num_items, dtype=np.int32)
            for idx, item_idx in enumerate(popularity_order):
                cluster_labels[item_idx] = idx % num_clusters

            logger.info(f"   Items sorted by popularity: top item has {item_popularity.max()} interactions")

            # Compute cluster centers
            cluster_centers = np.array([
                item_embeddings[cluster_labels == i].mean(axis=0)
                for i in range(num_clusters)
            ])

        else:
            raise ValueError(f"Unknown clustering method: {clustering_method}. "
                           f"Choose from: kmeans, spectral, hierarchical, frequency")

        # Create cluster mappings
        item_clusters = {item_idx: int(cluster_id) for item_idx, cluster_id in enumerate(cluster_labels)}

        cluster_items = {}
        for item_idx, cluster_id in item_clusters.items():
            if cluster_id not in cluster_items:
                cluster_items[cluster_id] = []
            cluster_items[cluster_id].append(item_idx)

        # Compute cluster statistics
        cluster_sizes = [len(items) for items in cluster_items.values()]

        clustering_data = {
            'item_clusters': item_clusters,
            'cluster_items': cluster_items,
            'cluster_centers': cluster_centers,
            'num_clusters': num_clusters,
            'clustering_method': clustering_method,
            'cluster_sizes': cluster_sizes,
            'avg_cluster_size': np.mean(cluster_sizes),
            'min_cluster_size': np.min(cluster_sizes),
            'max_cluster_size': np.max(cluster_sizes),
        }

        logger.info(f"✅ Clustering completed with {clustering_method.upper()}")
        logger.info(f"   Average cluster size: {clustering_data['avg_cluster_size']:.1f}")
        logger.info(f"   Cluster size range: {clustering_data['min_cluster_size']}-{clustering_data['max_cluster_size']}")
        logger.info(f"   Cluster size std: {np.std(cluster_sizes):.1f}")

        return clustering_data

    def create_sequences(self, ratings: pd.DataFrame) -> List[Dict]:
        """Create user interaction sequences."""
        logger.info("📝 Creating user sequences...")

        # Sort by user and timestamp
        ratings_sorted = ratings.sort_values(['user_idx', 'timestamp'])

        sequences = []
        for user_idx in ratings_sorted['user_idx'].unique():
            user_ratings = ratings_sorted[ratings_sorted['user_idx'] == user_idx]

            item_sequence = user_ratings['item_idx'].tolist()
            timestamp_sequence = user_ratings['timestamp'].tolist()

            # Split long sequences into overlapping windows
            if len(item_sequence) > self.max_sequence_length:
                for start_idx in range(0, len(item_sequence) - self.max_sequence_length + 1,
                                     self.max_sequence_length // 2):
                    end_idx = start_idx + self.max_sequence_length
                    sequences.append({
                        'user_id': user_idx,
                        'item_ids': item_sequence[start_idx:end_idx],
                        'timestamps': timestamp_sequence[start_idx:end_idx],
                        'sequence_length': end_idx - start_idx,
                    })
            else:
                sequences.append({
                    'user_id': user_idx,
                    'item_ids': item_sequence,
                    'timestamps': timestamp_sequence,
                    'sequence_length': len(item_sequence),
                })

        logger.info(f"✅ Created {len(sequences):,} sequences")
        return sequences

    def split_sequences(self, sequences: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split sequences into train/validation/test sets using leave-one-out strategy.

        Args:
            sequences: List of user sequences to split
        """
        return self._split_simple(sequences)

    def _split_simple(self, sequences: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Leave-one-out split with held-out items to prevent data leakage.

        Train:
        - Items [0...N-12] (no skip at start, hold out last 12)
        - For sequences < 128: use as-is
        - For sequences >= 128: create sliding windows with stride 64

        Val/Test (identical):
        - For sequences < 128: use full sequence
        - For sequences >= 128: use last 128 items (most recent context)
        """
        logger.info("✂️ Splitting with LEAVE-ONE-OUT strategy...")
        logger.info("   Train: Items [0...N-12], sliding windows for >128")
        logger.info("   Val/Test: Last 128 items if >=128, else full (Val == Test)")

        # Group sequences by user and take the LONGEST sequence per user
        user_to_longest_seq = {}
        for seq in sequences:
            user_id = seq['user_id']
            if user_id not in user_to_longest_seq:
                user_to_longest_seq[user_id] = seq
            else:
                # Keep the longest sequence
                if len(seq['item_ids']) > len(user_to_longest_seq[user_id]['item_ids']):
                    user_to_longest_seq[user_id] = seq

        train_sequences = []
        val_sequences = []
        test_sequences = []

        skipped_users = 0

        for user_id, seq in user_to_longest_seq.items():
            item_ids = seq['item_ids']

            # Filter out any padding (item_id = 0)
            item_ids_no_padding = [item_id for item_id in item_ids if item_id != 0]
            n_items = len(item_ids_no_padding)

            # Need at least 38 items: 26 (min train) + 12 (holdout)
            if n_items < 38:
                skipped_users += 1
                continue

            # Train: Remove last 12 (no skip at start)
            train_items = item_ids_no_padding[:-12]

            # Create sliding windows for train if sequence is long
            if len(train_items) > self.max_sequence_length:
                # Create overlapping windows
                for start_idx in range(0, len(train_items) - self.max_sequence_length + 1,
                                     self.max_sequence_length // 2):  # stride = 64
                    end_idx = min(start_idx + self.max_sequence_length, len(train_items))
                    train_seq = seq.copy()
                    train_seq['item_ids'] = train_items[start_idx:end_idx]
                    train_seq['sequence_length'] = len(train_items[start_idx:end_idx])
                    train_sequences.append(train_seq)
            else:
                # Short sequence, use as is
                train_seq = seq.copy()
                train_seq['item_ids'] = train_items
                train_seq['sequence_length'] = len(train_items)
                train_sequences.append(train_seq)

            # Val: Same as test
            # For sequences >= 128, use last 128 items (most recent context)
            # For sequences < 128, use full sequence
            if len(item_ids_no_padding) >= self.max_sequence_length:
                test_items = item_ids_no_padding[-self.max_sequence_length:]
            else:
                test_items = item_ids_no_padding

            test_seq = seq.copy()
            test_seq['item_ids'] = test_items
            test_seq['sequence_length'] = len(test_items)
            val_sequences.append(test_seq)
            test_sequences.append(test_seq)

        logger.info(f"   Train: {len(train_sequences):,} sequences (items [0...N-12])")
        logger.info(f"   Validation: {len(val_sequences):,} sequences (last 128 items if >128, else full)")
        logger.info(f"   Test: {len(test_sequences):,} sequences (last 128 items if >128, else full)")
        logger.info(f"   Val == Test: {val_sequences == test_sequences}")
        logger.info(f"   Skipped {skipped_users:,} users with < 38 items")
        logger.info("   ✅ Last 12+ items held out from training!")

        # Validate the splits
        self._validate_splits(train_sequences, val_sequences, test_sequences, user_to_longest_seq)

        return train_sequences, val_sequences, test_sequences

    def _validate_splits(self, train_sequences: List[Dict], val_sequences: List[Dict],
                        test_sequences: List[Dict], user_to_longest_seq: Dict):
        """Comprehensive validation of train/val/test splits."""
        logger.info("🔍 Validating splits...")

        issues = []
        warnings = []

        # Check 1: Val and Test should be identical
        if len(val_sequences) != len(test_sequences):
            issues.append(f"Val and Test have different lengths: {len(val_sequences)} vs {len(test_sequences)}")
        else:
            for i, (val_seq, test_seq) in enumerate(zip(val_sequences, test_sequences)):
                if val_seq['item_ids'] != test_seq['item_ids']:
                    issues.append(f"Sequence {i}: Val != Test for user {val_seq['user_id']}")
                    break

        # Check 2: Group train sequences by user
        train_by_user = {}
        for seq in train_sequences:
            user_id = seq['user_id']
            if user_id not in train_by_user:
                train_by_user[user_id] = []
            train_by_user[user_id].append(seq)

        # Check 3: Val/Test sequences by user
        val_by_user = {seq['user_id']: seq for seq in val_sequences}
        test_by_user = {seq['user_id']: seq for seq in test_sequences}

        # Check 4: Verify each user's sequences
        for user_id, original_seq in user_to_longest_seq.items():
            if user_id not in train_by_user:
                continue  # User was skipped (< 38 items)

            original_items = [x for x in original_seq['item_ids'] if x != 0]
            n_items = len(original_items)

            # Get train sequences for this user
            user_train_seqs = train_by_user.get(user_id, [])
            user_test_seq = test_by_user.get(user_id)

            if not user_test_seq:
                issues.append(f"User {user_id}: No test sequence found")
                continue

            # Check 4a: No padding in sequences
            for seq in user_train_seqs:
                if 0 in seq['item_ids']:
                    issues.append(f"User {user_id}: Train sequence contains padding (0)")
                    break

            if 0 in user_test_seq['item_ids']:
                issues.append(f"User {user_id}: Test sequence contains padding (0)")

            # Check 4b: Test sequence length
            test_items = user_test_seq['item_ids']
            expected_test_len = min(n_items, self.max_sequence_length)
            if len(test_items) != expected_test_len:
                issues.append(f"User {user_id}: Test length {len(test_items)}, expected {expected_test_len}")

            # Check 4c: Test uses last items (for sequences >= 128)
            if n_items >= self.max_sequence_length:
                expected_test_items = original_items[-self.max_sequence_length:]
                if test_items != expected_test_items:
                    issues.append(f"User {user_id}: Test doesn't use last {self.max_sequence_length} items")
            else:
                if test_items != original_items:
                    issues.append(f"User {user_id}: Test doesn't match full sequence")

            # Check 4d: Train sequences don't contain last 12 items
            last_12_items = set(original_items[-12:])
            for seq_idx, seq in enumerate(user_train_seqs):
                train_items_set = set(seq['item_ids'])
                overlap = train_items_set & last_12_items
                if overlap:
                    issues.append(f"User {user_id}, train seq {seq_idx}: Contains {len(overlap)} items from last 12")
                    break

            # Check 4e: Train sequences use items [0:-12]
            expected_train_items = original_items[:-12]

            # If sequence > 128, should have multiple train sequences
            if len(expected_train_items) > self.max_sequence_length:
                # Should have sliding windows
                expected_num_windows = 1 + (len(expected_train_items) - self.max_sequence_length) // (self.max_sequence_length // 2)
                if len(user_train_seqs) < expected_num_windows:
                    warnings.append(f"User {user_id}: Expected ~{expected_num_windows} train windows, got {len(user_train_seqs)}")

                # All train items should come from expected_train_items
                all_train_items = []
                for seq in user_train_seqs:
                    all_train_items.extend(seq['item_ids'])

                unique_train_items = set(all_train_items)
                expected_train_set = set(expected_train_items)

                # Check all train items are from expected range
                invalid_items = unique_train_items - expected_train_set
                if invalid_items:
                    issues.append(f"User {user_id}: Train contains {len(invalid_items)} items not in [0:-12]")
            else:
                # Should have exactly 1 train sequence
                if len(user_train_seqs) != 1:
                    issues.append(f"User {user_id}: Expected 1 train sequence, got {len(user_train_seqs)}")
                else:
                    if user_train_seqs[0]['item_ids'] != expected_train_items:
                        issues.append(f"User {user_id}: Train items don't match expected [0:-12]")

            # Check 4f: Sequence lengths match item_ids length
            for seq in user_train_seqs:
                if seq['sequence_length'] != len(seq['item_ids']):
                    issues.append(f"User {user_id}: Train sequence_length mismatch")
                    break

            if user_test_seq['sequence_length'] != len(user_test_seq['item_ids']):
                issues.append(f"User {user_id}: Test sequence_length mismatch")

        # Check 5: Overall statistics
        logger.info(f"   Total users in splits: {len(val_by_user)}")
        logger.info(f"   Users with multiple train sequences: {sum(1 for seqs in train_by_user.values() if len(seqs) > 1)}")
        logger.info(f"   Average train sequences per user: {len(train_sequences) / len(train_by_user):.2f}")

        train_lengths = [seq['sequence_length'] for seq in train_sequences]
        test_lengths = [seq['sequence_length'] for seq in test_sequences]
        logger.info(f"   Train sequence lengths: min={min(train_lengths)}, max={max(train_lengths)}, avg={np.mean(train_lengths):.1f}")
        logger.info(f"   Test sequence lengths: min={min(test_lengths)}, max={max(test_lengths)}, avg={np.mean(test_lengths):.1f}")

        # Report results
        if issues:
            logger.error(f"❌ Found {len(issues)} CRITICAL issues:")
            for issue in issues[:10]:  # Show first 10
                logger.error(f"   - {issue}")
            if len(issues) > 10:
                logger.error(f"   ... and {len(issues) - 10} more issues")
            raise ValueError(f"Split validation failed with {len(issues)} issues")

        if warnings:
            logger.warning(f"⚠️  Found {len(warnings)} warnings:")
            for warning in warnings[:5]:
                logger.warning(f"   - {warning}")

        logger.info("✅ Split validation passed!")


    def save_processed_data(self, ratings: pd.DataFrame, movies: pd.DataFrame,
                           item_embeddings_dict: Dict[str, np.ndarray], clustering_data: Dict,
                           train_sequences: List[Dict], val_sequences: List[Dict],
                           test_sequences: List[Dict], filter_stats: Dict):
        """Save all processed data.

        Args:
            item_embeddings_dict: Dictionary with keys 'random', 'metadata', 'wals'
                                 mapping to embedding arrays
        """
        logger.info("💾 Saving processed data...")

        # Save sequences (single mode, no mode-specific files)
        joblib.dump(train_sequences, self.processed_data_dir / "train_sequences.pkl")
        joblib.dump(val_sequences, self.processed_data_dir / "val_sequences.pkl")
        joblib.dump(test_sequences, self.processed_data_dir / "test_sequences.pkl")
        logger.info(f"   ✅ Saved train sequences: {len(train_sequences):,}")
        logger.info(f"   ✅ Saved val sequences: {len(val_sequences):,}")
        logger.info(f"   ✅ Saved test sequences: {len(test_sequences):,}")

        # Save all 3 embedding types
        for method, embeddings in item_embeddings_dict.items():
            embedding_path = self.processed_data_dir / f"item_embeddings_{method}.npy"
            np.save(embedding_path, embeddings)
            logger.info(f"   ✅ Saved {method} embeddings: {embedding_path}")

        # Backward compatibility: also save metadata as default
        if 'metadata' in item_embeddings_dict:
            np.save(self.processed_data_dir / "item_embeddings.npy", item_embeddings_dict['metadata'])
            logger.info("   ✅ Saved default embeddings (metadata) for backward compatibility")

        # Save clustering data in the format expected by the training code
        # The training code expects 4 separate pickle objects loaded sequentially:
        # 1. cluster_assignments: item_idx -> cluster_id mapping
        # 2. cluster_indices: cluster_id -> list of item_indices
        # 3. in_cluster_id: item_idx -> position within cluster
        # 4. cluster_means: cluster centers

        import pickle

        # Convert our clustering data to the expected format
        # Create array of size 3261 where index = item_id (direct mapping)
        # Index 0: padding token (maps to cluster 0, always masked)
        # Index 1-3260: real MovieLens items
        cluster_assignments_base = np.array([clustering_data['item_clusters'][i] for i in range(len(clustering_data['item_clusters']))])

        # Extend to size 3261 by prepending padding entry
        cluster_assignments = np.zeros(3261, dtype=cluster_assignments_base.dtype)
        cluster_assignments[0] = 0  # Padding maps to cluster 0 (arbitrary, always masked)
        cluster_assignments[1:] = cluster_assignments_base  # Items 1-3260

        print("✅ Cluster assignments created")
        print(f"   Length: {len(cluster_assignments)} (indices 0-3260)")
        print("   Index 0: Padding token (cluster 0, always masked)")
        print("   Index 1-3260: MovieLens items (direct mapping)")
        print(f"   Number of clusters: {clustering_data['num_clusters']}")
        print("   ⚠️  Set ITEM_VOCAB_SIZE = 3261 in config")
        print(f"   ⚠️  Set NUM_ITEM_CLUSTERS = {clustering_data['num_clusters']} in config")

        # Create cluster_indices as a 2D array [num_clusters, max_cluster_size] padded with -1
        max_cluster_size = max(len(items) for items in clustering_data['cluster_items'].values()) + 10  # Add buffer
        cluster_indices = np.full((clustering_data['num_clusters'], max_cluster_size), -1, dtype=np.int32)

        # Create in_cluster_id mapping (position of each item within its cluster)
        # Size 3261 to match cluster_assignments (index = item_id)
        in_cluster_id = np.zeros(3261, dtype=np.int32)

        # Fill both cluster_indices and in_cluster_id
        # Use direct mapping: item_id as index (1-3260)
        for cluster_id, items in clustering_data['cluster_items'].items():
            for pos, ml_item_0indexed in enumerate(items):
                ml_item_id = ml_item_0indexed + 1  # MovieLens uses 1-indexed IDs
                cluster_indices[cluster_id, pos] = ml_item_id  # Store 1-indexed ID
                in_cluster_id[ml_item_id] = pos  # Direct indexing: item_id → position

        # Index 0 (padding) in in_cluster_id remains 0 (doesn't matter, always masked)

        with open(self.processed_data_dir / "clustering.pkl", 'wb') as f:
            pickle.dump(cluster_assignments, f)                     # cluster_assignments
            pickle.dump(cluster_indices, f)                         # cluster_indices
            pickle.dump(in_cluster_id, f)                          # in_cluster_id
            pickle.dump(clustering_data['cluster_centers'], f)      # cluster_means

        # Save preprocessing info
        # Get embedding dim from any of the embeddings (they're all the same size)
        embedding_dim = list(item_embeddings_dict.values())[0].shape[1]
        preprocessing_info = {
            'num_users': len(filter_stats['user_to_idx']),
            'num_items': len(filter_stats['item_to_idx']),
            'embedding_dim': embedding_dim,
            'num_clusters': clustering_data['num_clusters'],
            'max_sequence_length': self.max_sequence_length,
            'min_user_interactions': self.min_user_interactions,
            'min_item_interactions': self.min_item_interactions,
            'processing_date': datetime.now().isoformat(),
            # Add mappings for metadata loading
            'item_to_idx': filter_stats['item_to_idx'],  # Original movie ID -> internal item index
            'user_to_idx': filter_stats['user_to_idx'],  # Original user ID -> internal user index
        }
        joblib.dump(preprocessing_info, self.processed_data_dir / "preprocessing_info.pkl")

        # Save dataset statistics
        dataset_stats = {
            'num_users': preprocessing_info['num_users'],
            'num_items': preprocessing_info['num_items'],
            'num_ratings': len(ratings),
            'num_clusters': clustering_data['num_clusters'],
            'train_sequences': len(train_sequences),
            'val_sequences': len(val_sequences),
            'test_sequences': len(test_sequences),
            'avg_sequence_length': np.mean([seq['sequence_length'] for seq in train_sequences + val_sequences + test_sequences]),
            'avg_cluster_size': clustering_data['avg_cluster_size'],
        }
        joblib.dump(dataset_stats, self.processed_data_dir / "dataset_stats.pkl")

        logger.info("✅ All data saved successfully")
        logger.info(f"   Output directory: {self.processed_data_dir}")

    def process(self, num_clusters: int = 256, embedding_dim: int = 128,
               clustering_method: str = 'kmeans'):
        """Run the complete processing pipeline with leave-one-out split strategy."""
        logger.info("🚀 Starting MovieLens 20M processing pipeline...")
        logger.info("   Split strategy: leave-one-out (single mode)")
        logger.info(f"   Clustering method: {clustering_method}")

        # Step 1: Download data
        self.download_data()

        # Step 2: Load raw data
        ratings, movies, tags = self.load_raw_data()

        # Step 3: Filter data
        ratings_filtered, movies_filtered, filter_stats = self.filter_data(ratings, movies)

        # Step 4: Create all 3 types of item embeddings
        logger.info("🎨 Creating item embeddings using 3 methods...")
        num_items = len(filter_stats['item_to_idx'])

        metadata_embeddings = self.create_item_embeddings(movies_filtered, embedding_dim)
        random_embeddings = self.create_random_embeddings(num_items, embedding_dim)
        wals_embeddings = self.create_wals_embeddings(ratings_filtered, num_items, embedding_dim)

        item_embeddings_dict = {
            'metadata': metadata_embeddings,
            'random': random_embeddings,
            'wals': wals_embeddings
        }
        logger.info(f"✅ Generated all 3 embedding types (each: {metadata_embeddings.shape})")

        # Step 5: Compute item popularity (for frequency clustering)
        item_popularity = None
        if clustering_method == 'frequency':
            logger.info("📊 Computing item popularity for frequency clustering...")
            item_counts = ratings_filtered.groupby('item_idx').size().values
            item_popularity = np.zeros(len(filter_stats['item_to_idx']))
            for item_idx in range(len(filter_stats['item_to_idx'])):
                if item_idx < len(item_counts):
                    item_popularity[item_idx] = item_counts[item_idx]
            logger.info(f"   Popularity range: {item_popularity.min():.0f} to {item_popularity.max():.0f}")

        # Step 6: Create clusters (use metadata embeddings as reference)
        clustering_data = self.create_clusters(
            metadata_embeddings, num_clusters,
            clustering_method=clustering_method,
            item_popularity=item_popularity
        )

        # Step 7: Create sequences
        sequences = self.create_sequences(ratings_filtered)

        # Step 8: Split sequences
        train_sequences, val_sequences, test_sequences = self.split_sequences(sequences)

        # Step 9: Save processed data with all 3 embedding types
        self.save_processed_data(
            ratings_filtered, movies_filtered, item_embeddings_dict, clustering_data,
            train_sequences, val_sequences, test_sequences, filter_stats
        )

        logger.info("🎉 Processing pipeline completed successfully!")
        return True

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Process MovieLens 20M dataset for EfficientIDS")
    parser.add_argument("--output_dir", default="./data/ml20m_processed",
                       help="Output directory for processed data")
    parser.add_argument("--num_clusters", type=int, default=100,
                       help="Number of item clusters to create (default: 100)")
    parser.add_argument("--embedding_dim", type=int, default=128,
                       help="Dimension of item embeddings (default: 128)")
    parser.add_argument("--clustering_method", type=str, default='kmeans',
                       choices=['kmeans', 'spectral', 'hierarchical', 'frequency'],
                       help="Clustering method: kmeans (fast, content-based), "
                            "spectral (best quality, slower), "
                            "hierarchical (tree structure, slow), "
                            "frequency (popularity-based)")
    parser.add_argument("--min_user_interactions", type=int, default=20,
                       help="Minimum interactions per user")
    parser.add_argument("--min_item_interactions", type=int, default=10,
                       help="Minimum interactions per item")

    args = parser.parse_args()

    # Create processor
    processor = MovieLensProcessor(args.output_dir)
    processor.min_user_interactions = args.min_user_interactions
    processor.min_item_interactions = args.min_item_interactions

    # Run processing
    try:
        success = processor.process(
            args.num_clusters,
            args.embedding_dim,
            args.clustering_method
        )
        if success:
            print("\n🎉 SUCCESS!")
            print(f"📁 Processed data saved to: {args.output_dir}/processed/")
            print("\nNext steps:")
            print("1. python -m paxml.main --exp=movielens_config.MovieLensDebug  # Quick test")
            print("2. python -m paxml.main --exp=movielens_config.MovieLensSmall  # Full training")
        else:
            print("❌ Processing failed!")
            sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Processing failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

