
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it."""
    return SentenceTransformer(MODEL_NAME, device="cpu")


class EmbeddingService:
    """Convert text into normalized numerical vectors."""

    def __init__(self):
        self.model = get_model()

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate one embedding vector."""
        if not text or not text.strip():
            raise ValueError("Text cannot be empty.")

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        return np.asarray(embedding, dtype=np.float32)

    def generate_embeddings(self, texts: list[str]) -> np.ndarray:
        """Generate embedding vectors for multiple texts."""
        if not texts:
            return np.empty(
                (0, EMBEDDING_DIMENSION),
                dtype=np.float32
            )

        if any(not text or not text.strip() for text in texts):
            raise ValueError("Text entries must not be empty.")

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        return np.asarray(embeddings, dtype=np.float32)
