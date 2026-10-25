
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class KeywordSearchService:
    """Find document chunks using NLP keyword matching with TF-IDF."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b[a-zA-Z0-9_]+\b",
        )

    def search(
        self,
        question: str,
        chunks: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """Rank chunks by keyword similarity to the question."""

        if not question or not question.strip() or not chunks:
            return []

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        valid_chunks = [
            chunk for chunk in chunks
            if chunk.get("text", "").strip()
        ]

        if not valid_chunks:
            return []

        texts = [chunk["text"] for chunk in valid_chunks]

        # Fit TF-IDF on the chunks and transform the question
        # into the same vector space.
        try:
            chunk_vectors = self.vectorizer.fit_transform(texts)
            question_vector = self.vectorizer.transform([question])
        except ValueError:
            # This can happen if all text contains only stop words.
            return []

        scores = cosine_similarity(
            question_vector,
            chunk_vectors,
        ).flatten()

        ranked_positions = np.argsort(scores)[::-1][:top_k]

        results = []

        for position in ranked_positions:
            score = float(scores[position])

            # Do not return chunks with no keyword overlap.
            if score <= 0:
                continue

            chunk = valid_chunks[int(position)]

            results.append({
                **chunk,
                "keyword_score": score,
            })

        return results
