from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it."""
    return SentenceTransformer(MODEL_NAME, device="cpu")


def generate_embeddings(texts: list[str]) -> np.ndarray:
    """Convert a list of text strings into normalized embedding vectors."""
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    if any(not text.strip() for text in texts):
        raise ValueError("Text entries must not be empty.")

    model = get_model()

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return embeddings.astype(np.float32)


if __name__ == "__main__":
    sample_texts = [
        "Alice followed the rabbit down the hole.",
        "Alice went into a rabbit hole.",
        "The weather is sunny today."
    ]

    vectors = generate_embeddings(sample_texts)

    print("Number of sentences:", len(sample_texts))
    print("Embedding shape:", vectors.shape)
    print("Vector dimensions:", vectors.shape[1])
    print("Vector data type:", vectors.dtype)

    similarity = float(np.dot(vectors[0], vectors[1]))
    print("Similarity between the first two sentences:", round(similarity, 4))
