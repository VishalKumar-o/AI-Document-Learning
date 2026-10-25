
import faiss
import numpy as np

from app.services.embedding_service import (
    EmbeddingService,
    EMBEDDING_DIMENSION,
)
from app.services.keyword_search_service import KeywordSearchService


class RetrievalService:
    """Retrieve document chunks using hybrid semantic and keyword search."""

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.keyword_search_service = KeywordSearchService()
        self.documents = {}

    def add_document(
        self,
        document_id: str,
        chunks: list[dict],
    ) -> None:
        """Create a FAISS index for one document."""

        if not document_id.strip():
            raise ValueError("Document ID cannot be empty.")

        if document_id in self.documents:
            raise ValueError(
                f"Document '{document_id}' is already indexed."
            )

        if not chunks:
            raise ValueError("Cannot index an empty document.")

        for chunk in chunks:
            if not chunk.get("text", "").strip():
                raise ValueError("Every chunk must contain text.")

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_service.generate_embeddings(texts)

        if embeddings.shape != (len(chunks), EMBEDDING_DIMENSION):
            raise ValueError("Unexpected embedding dimensions.")

        # Normalized embeddings allow inner product to measure
        # cosine similarity.
        index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
        index.add(embeddings)

        self.documents[document_id] = {
            "index": index,
            "chunks": chunks,
        }

    def search(
        self,
        question: str,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> list[dict]:
        """Combine semantic and keyword rankings using weighted RRF."""

        if not question or not question.strip():
            return []

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if document_id is not None:
            if document_id not in self.documents:
                return []

            selected_documents = [
                (document_id, self.documents[document_id])
            ]
        else:
            selected_documents = list(self.documents.items())

        if not selected_documents:
            return []

        query_vector = self.embedding_service.generate_embedding(
            question.strip()
        )
        query_vector = np.asarray(
            [query_vector],
            dtype=np.float32,
        )

        rrf_scores = {}
        result_details = {}

        rrf_constant = 60
        semantic_weight = 0.4
        keyword_weight = 0.6

        for current_document_id, document_data in selected_documents:
            chunks = document_data["chunks"]
            index = document_data["index"]

            # 1. Semantic search using FAISS.
            result_count = min(max(top_k * 2, 10), index.ntotal)

            semantic_scores, positions = index.search(
                query_vector,
                result_count,
            )

            semantic_results = []

            for score, position in zip(
                semantic_scores[0],
                positions[0],
            ):
                if position < 0:
                    continue

                chunk = chunks[int(position)]

                result_key = (
                    current_document_id,
                    chunk["chunk_id"],
                )

                semantic_results.append(
                    (result_key, float(score))
                )

                if result_key not in result_details:
                    result_details[result_key] = {
                        **chunk,
                        "document_id": current_document_id,
                    }

                result_details[result_key]["semantic_score"] = float(score)

            # 2. Keyword search using TF-IDF.
            keyword_results = self.keyword_search_service.search(
                question=question,
                chunks=chunks,
                top_k=min(max(top_k * 2, 10), len(chunks)),
            )

            # 3. Add semantic ranking contributions.
            for rank, (result_key, score) in enumerate(
                semantic_results,
                start=1,
            ):
                rrf_scores[result_key] = (
                    rrf_scores.get(result_key, 0.0)
                    + semantic_weight / (rrf_constant + rank)
                )

            # 4. Add keyword ranking contributions.
            for rank, chunk in enumerate(
                keyword_results,
                start=1,
            ):
                result_key = (
                    current_document_id,
                    chunk["chunk_id"],
                )

                if result_key not in result_details:
                    result_details[result_key] = {
                        **chunk,
                        "document_id": current_document_id,
                    }

                rrf_scores[result_key] = (
                    rrf_scores.get(result_key, 0.0)
                    + keyword_weight / (rrf_constant + rank)
                )

                result_details[result_key]["keyword_score"] = (
                    chunk["keyword_score"]
                )

        # 5. Sort by the combined ranking score.
        ranked_keys = sorted(
            rrf_scores,
            key=lambda key: rrf_scores[key],
            reverse=True,
        )

        results = []

        for result_key in ranked_keys[:top_k]:
            result = dict(result_details[result_key])

            result["hybrid_score"] = rrf_scores[result_key]

            # Preserve compatibility with existing tests and API clients.
            result["score"] = result.get(
                "semantic_score",
                result["hybrid_score"],
            )

            results.append(result)

        return results

    def remove_document(self, document_id: str) -> bool:
        """Remove a document and its in-memory search index."""

        return self.documents.pop(document_id, None) is not None
