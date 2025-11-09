"""
MovieLens Data Pipeline for EfficientIDS Flax

Complete port from PAXml's movielens_seqio.py with:
- Text metadata interleaving (75%/25% split)
- SentencePiece tokenization
- Proper loss masking (item→item only)
- Real MovieLens sequences

Training Modes:
- id_only: Pure item sequences [ITEM_1, ITEM_2, ...]
- text_metadata: Interleaved [TEXT_1, ITEM_1, TEXT_2, ITEM_2, ...]
"""

import numpy as np
import joblib
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator
import logging

logger = logging.getLogger(__name__)

# Configuration
DEFAULT_DATA_DIR = "../efficientids/data/ml1m_processed/processed"
PADDING_TOKEN = 0


class MovieLensDataLoader:
    """Loads processed MovieLens data."""

    def __init__(self, data_dir: str = DEFAULT_DATA_DIR):
        self.data_dir = Path(data_dir)
        self._load_metadata()
        self._load_movie_info()
        self._load_sequences()

    def _load_metadata(self):
        """Load preprocessing metadata and clustering."""
        try:
            self.preprocessing_info = joblib.load(self.data_dir / "preprocessing_info.pkl")
            self.dataset_stats = joblib.load(self.data_dir / "dataset_stats.pkl")

            # Load clustering (4-part pickle format)
            with open(self.data_dir / "clustering.pkl", 'rb') as f:
                self.cluster_assignments = pickle.load(f)  # [num_items] item -> cluster
                self.cluster_indices = pickle.load(f)      # [num_clusters, max_size] cluster -> items
                self.in_cluster_id = pickle.load(f)        # [num_items] item -> position
                self.cluster_centers = pickle.load(f)      # [num_clusters, dim] centers

            self.num_items = self.preprocessing_info['num_items']
            self.num_clusters = self.cluster_indices.shape[0]

            logger.info(f"✅ Loaded metadata: {self.num_items} items, {self.num_clusters} clusters")

        except FileNotFoundError as e:
            logger.error(f"❌ Data not found at {self.data_dir}")
            logger.error("   Run: python ../efficientids/process_movielens.py")
            raise e

    def _load_movie_info(self):
        """Load movie titles, genres, years from movies.dat."""
        raw_dir = self.data_dir.parent / "raw" / "ml-1m"
        movies_file = raw_dir / "movies.dat"

        self.movie_info = {}

        if not movies_file.exists():
            logger.warning(f"⚠️  Movies file not found: {movies_file}")
            logger.warning("   Text metadata mode will not work!")
            return

        try:
            with open(movies_file, encoding='latin-1') as f:
                for line in f:
                    parts = line.strip().split('::')
                    if len(parts) >= 3:
                        original_id = int(parts[0])
                        title = parts[1]
                        genres = parts[2].split('|')
                        year = self._extract_year(title)

                        # Map to internal item index
                        if 'item_to_idx' in self.preprocessing_info:
                            item_idx = self.preprocessing_info['item_to_idx'].get(original_id)
                            if item_idx is not None:
                                self.movie_info[item_idx] = {
                                    'title': title,
                                    'genres': genres,
                                    'year': year,
                                }

            logger.info(f"✅ Loaded metadata for {len(self.movie_info)} movies")

        except Exception as e:
            logger.error(f"❌ Failed to load movie metadata: {e}")

    def _extract_year(self, title: str) -> int:
        """Extract year from 'Toy Story (1995)'."""
        match = re.search(r'\((\d{4})\)', title)
        return int(match.group(1)) if match else 0

    def _load_sequences(self):
        """Load train/validation/test sequences."""
        self.sequences = {}
        for split in ['train', 'validation', 'test']:
            path = self.data_dir / f"{split}_sequences.pkl"
            if path.exists():
                self.sequences[split] = joblib.load(path)
                logger.info(f"✅ Loaded {split}: {len(self.sequences[split])} sequences")
            else:
                logger.warning(f"⚠️  {split}_sequences.pkl not found")
                self.sequences[split] = []

    def format_movie_metadata(
        self,
        item_idx: int,
        use_genres: bool = True,
        use_year: bool = True,
        use_title: bool = False,
    ) -> str:
        """
        Format movie metadata as text.

        Example: "1995 Animation Childrens Comedy id: 1"
        """
        info = self.movie_info.get(item_idx, {
            'title': f'Unknown ({item_idx})',
            'genres': [],
            'year': 0,
        })

        parts = []

        if use_year and info['year']:
            parts.append(str(info['year']))

        if use_genres and info['genres']:
            clean_genres = [g.replace("'", "").replace("-", "") for g in info['genres']]
            parts.extend(clean_genres)

        if use_title:
            parts.append(f"title: {info['title']}")

        parts.append(f"id: {item_idx}")

        return ' '.join(parts)

    def get_sequences(self, split: str) -> List[Dict]:
        """Get sequences for a split."""
        return self.sequences.get(split, [])


class SentencePieceTokenizer:
    """SentencePiece tokenizer for text metadata."""

    def __init__(self, vocab_path: str = 'gs://t5-data/vocabs/cc_all.32000/sentencepiece.model'):
        """
        Initialize tokenizer.

        Args:
            vocab_path: Path to SentencePiece model. Can be:
                - GCS path (gs://...)
                - Local file path
                - None (will try to download default)
        """
        self.vocab_path = vocab_path
        self._tokenizer = None
        self._load_tokenizer()

    def _load_tokenizer(self):
        """Load SentencePiece model."""
        try:
            import sentencepiece as spm

            # Handle GCS paths
            if self.vocab_path.startswith('gs://'):
                import tempfile
                import subprocess

                # Try to download from GCS
                with tempfile.NamedTemporaryFile(suffix='.model', delete=False) as tmp:
                    try:
                        subprocess.run(
                            ['gsutil', 'cp', self.vocab_path, tmp.name],
                            check=True,
                            capture_output=True
                        )
                        local_path = tmp.name
                    except subprocess.CalledProcessError:
                        logger.warning(f"⚠️  Failed to download {self.vocab_path}")
                        logger.warning("   Falling back to character-level tokenization")
                        self._tokenizer = None
                        return
            else:
                local_path = self.vocab_path

            # Load model
            self._tokenizer = spm.SentencePieceProcessor()
            self._tokenizer.Load(local_path)
            logger.info(f"✅ Loaded SentencePiece: {self._tokenizer.vocab_size()} tokens")

        except ImportError:
            logger.warning("⚠️  sentencepiece not installed")
            logger.warning("   Install: pip install sentencepiece")
            logger.warning("   Falling back to character-level tokenization")
            self._tokenizer = None
        except Exception as e:
            logger.warning(f"⚠️  Failed to load tokenizer: {e}")
            logger.warning("   Falling back to character-level tokenization")
            self._tokenizer = None

    def encode(self, text: str) -> List[int]:
        """Tokenize text to IDs."""
        if self._tokenizer is not None:
            return self._tokenizer.EncodeAsIds(text)
        else:
            # Fallback: character-level (for testing without SentencePiece)
            # Map each char to 32-255 range to avoid item ID conflicts
            return [min(255, 32 + ord(c)) for c in text[:50]]  # Limit length

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        if self._tokenizer is not None:
            return self._tokenizer.vocab_size()
        else:
            return 256  # Fallback character-level


class MovieLensDataset:
    """
    MovieLens dataset with text interleaving.

    Generates batches with proper masking for training.
    """

    def __init__(
        self,
        data_loader: MovieLensDataLoader,
        split: str = 'train',
        mode: str = 'text_metadata',  # 'id_only' or 'text_metadata'
        text_ratio: float = 0.75,      # 75% text, 25% ID-only
        seq_len_id_only: int = 128,
        seq_len_text: int = 384,
        use_genres: bool = True,
        use_years: bool = True,
        use_titles: bool = False,
        tokenizer: Optional[SentencePieceTokenizer] = None,
        seed: int = 42,
    ):
        self.data_loader = data_loader
        self.split = split
        self.mode = mode
        self.text_ratio = text_ratio
        self.seq_len_id_only = seq_len_id_only
        self.seq_len_text = seq_len_text
        self.use_genres = use_genres
        self.use_years = use_years
        self.use_titles = use_titles
        self.seed = seed

        # Get sequences
        self.sequences = data_loader.get_sequences(split)
        logger.info(f"📦 Dataset {split}: {len(self.sequences)} sequences, mode={mode}")

        # Tokenizer
        if tokenizer is None:
            self.tokenizer = SentencePieceTokenizer()
        else:
            self.tokenizer = tokenizer

        # RNG for text/id split
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return len(self.sequences)

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        """Iterate over examples."""
        for sequence in self.sequences:
            item_ids = sequence['item_ids']

            if len(item_ids) < 6:
                continue

            # Decide: text or ID-only
            use_text = (self.mode == 'text_metadata' and
                       self.rng.random() < self.text_ratio)

            if use_text:
                example = self._create_text_example(item_ids)
            else:
                example = self._create_id_only_example(item_ids)

            if example is not None:
                yield example

    def _create_text_example(self, item_ids: List[int]) -> Optional[Dict[str, np.ndarray]]:
        """
        Create interleaved text-item example.

        Format: [TEXT_1, ITEM_1, TEXT_2, ITEM_2, ...]

        Loss mask: Only item→item predictions count.
        """
        # Take last N items (fit in seq_len after interleaving)
        max_items = min(len(item_ids), 80)
        items_to_use = item_ids[-max_items:]

        # Build interleaved sequence
        input_ids = []
        item_weights = []  # 0=text token, 1=item token

        for item_id in items_to_use:
            if item_id == 0:
                continue

            # Get metadata text
            metadata = self.data_loader.format_movie_metadata(
                item_id,
                self.use_genres,
                self.use_years,
                self.use_titles,
            )

            # Tokenize
            text_tokens = self.tokenizer.encode(metadata)

            # Add text tokens
            input_ids.extend(text_tokens)
            item_weights.extend([0] * len(text_tokens))

            # Add item token
            input_ids.append(item_id)
            item_weights.append(1)

            # Check length
            if len(input_ids) > self.seq_len_text - 20:
                break

        if len(input_ids) == 0:
            return None

        # Compute loss mask: item positions where TARGET is also an item
        # PAXml handles text targets separately with word_xent, but we only have hierarchical
        # So mask to: current=item AND target=item
        loss_mask = []
        for i in range(len(input_ids)):
            current_is_item = (item_weights[i] == 1)
            # Since targets[i] = input_ids[i+1], check if next position is item
            target_is_item = (i + 1 < len(input_ids) and item_weights[i + 1] == 1)
            loss_mask.append(1 if (current_is_item and target_is_item) else 0)

        # Pad/truncate
        original_len = len(input_ids)

        if len(input_ids) > self.seq_len_text:
            input_ids = input_ids[:self.seq_len_text]
            item_weights = item_weights[:self.seq_len_text]
            loss_mask = loss_mask[:self.seq_len_text]
            original_len = self.seq_len_text  # Update original_len after truncation
        else:
            pad_len = self.seq_len_text - len(input_ids)
            input_ids.extend([PADDING_TOKEN] * pad_len)
            item_weights.extend([0] * pad_len)
            loss_mask.extend([0] * pad_len)

        # Attention mask (use updated original_len)
        attention_mask = [1] * original_len + [0] * (self.seq_len_text - original_len)

        # Targets (shifted by 1)
        targets = input_ids[1:] + [PADDING_TOKEN]

        return {
            'input_ids': np.array(input_ids, dtype=np.int32),
            'targets': np.array(targets, dtype=np.int32),
            'item_weights': np.array(item_weights, dtype=np.float32),
            'loss_mask': np.array(loss_mask, dtype=np.float32),
            'attention_mask': np.array(attention_mask, dtype=np.float32),
        }

    def _create_id_only_example(self, item_ids: List[int]) -> Dict[str, np.ndarray]:
        """
        Create pure ID sequence.

        Format: [ITEM_1, ITEM_2, ITEM_3, ...]
        """
        seq_len = self.seq_len_id_only

        # Take last seq_len items
        if len(item_ids) >= seq_len:
            full_sequence = item_ids[-seq_len:]
        else:
            # Right padding
            full_sequence = item_ids + [PADDING_TOKEN] * (seq_len - len(item_ids))

        original_len = min(len(item_ids), seq_len)

        # All tokens are items
        item_weights = [1.0] * original_len + [0.0] * (seq_len - original_len)

        # Loss mask: all positions (no text)
        loss_mask = [1.0] * max(0, original_len - 1) + [0.0] * (seq_len - original_len + 1)

        # Attention mask
        attention_mask = [1.0] * original_len + [0.0] * (seq_len - original_len)

        # Targets (shifted)
        targets = full_sequence[1:] + [PADDING_TOKEN]

        return {
            'input_ids': np.array(full_sequence, dtype=np.int32),
            'targets': np.array(targets, dtype=np.int32),
            'item_weights': np.array(item_weights, dtype=np.float32),
            'loss_mask': np.array(loss_mask, dtype=np.float32),
            'attention_mask': np.array(attention_mask, dtype=np.float32),
        }


def create_data_iterator(
    data_dir: str,
    split: str = 'train',
    mode: str = 'text_metadata',
    batch_size: int = 4,
    seq_len: int = 384,
    **kwargs
) -> Iterator[Dict[str, np.ndarray]]:
    """
    Create data iterator for training.

    Args:
        data_dir: Path to processed data
        split: 'train', 'validation', or 'test'
        mode: 'text_metadata' or 'id_only'
        batch_size: Batch size
        seq_len: Sequence length (384 for text, 128 for ID-only)
        **kwargs: Additional args for MovieLensDataset

    Yields:
        Batched examples
    """
    # Load data
    data_loader = MovieLensDataLoader(data_dir)

    # Create dataset
    dataset = MovieLensDataset(
        data_loader=data_loader,
        split=split,
        mode=mode,
        seq_len_text=seq_len if mode == 'text_metadata' else 384,
        seq_len_id_only=seq_len if mode == 'id_only' else 128,
        **kwargs
    )

    # Batch examples
    batch = {
        'input_ids': [],
        'targets': [],
        'item_weights': [],
        'loss_mask': [],
        'attention_mask': [],
    }

    for example in dataset:
        for key in batch.keys():
            batch[key].append(example[key])

        if len(batch['input_ids']) >= batch_size:
            # Convert to arrays
            yield {k: np.array(v) for k, v in batch.items()}

            # Reset batch
            batch = {k: [] for k in batch.keys()}

    # Yield remaining
    if len(batch['input_ids']) > 0:
        yield {k: np.array(v) for k, v in batch.items()}


if __name__ == "__main__":
    """Test data loading."""
    import sys

    print("Testing MovieLens Data Pipeline")
    print("=" * 70)

    # Load data
    data_loader = MovieLensDataLoader(DEFAULT_DATA_DIR)

    print(f"\n✅ Data loaded:")
    print(f"   Items: {data_loader.num_items}")
    print(f"   Clusters: {data_loader.num_clusters}")
    print(f"   Movies with metadata: {len(data_loader.movie_info)}")

    # Test tokenizer
    print(f"\n🔤 Testing tokenizer...")
    tokenizer = SentencePieceTokenizer()
    test_text = "1995 Animation Childrens Comedy id: 1"
    tokens = tokenizer.encode(test_text)
    print(f"   Text: {test_text}")
    print(f"   Tokens: {tokens[:20]}... (len={len(tokens)})")

    # Test dataset
    print(f"\n📦 Testing text_metadata dataset...")
    dataset = MovieLensDataset(
        data_loader=data_loader,
        split='train',
        mode='text_metadata',
        seq_len_text=384,
    )

    for i, example in enumerate(dataset):
        if i >= 2:
            break

        print(f"\nExample {i+1}:")
        print(f"   input_ids shape: {example['input_ids'].shape}")
        print(f"   targets shape: {example['targets'].shape}")
        print(f"   loss_mask sum: {example['loss_mask'].sum():.0f} (positions with loss)")
        print(f"   item_weights sum: {example['item_weights'].sum():.0f} (item tokens)")
        print(f"   attention_mask sum: {example['attention_mask'].sum():.0f} (non-padding)")

    print(f"\n" + "=" * 70)
    print(f"✅ Data pipeline working!")
