
from app.services.embedding_service import EmbeddingService


def main():
    embedding_service = EmbeddingService()

    text1 = "Alice drank a potion and became smaller."
    text2 = "Alice's size decreased after drinking the liquid."
    text3 = "The database uses tables to store records."

    embeddings = embedding_service.generate_embeddings(
        [text1, text2, text3]
    )

    print("Number of embeddings:", len(embeddings))
    print("Vector dimensions:", len(embeddings[0]))

    similarity_12 = sum(
        a * b for a, b in zip(embeddings[0], embeddings[1])
    )

    similarity_13 = sum(
        a * b for a, b in zip(embeddings[0], embeddings[2])
    )

    print(f"Similarity between related sentences: {similarity_12:.4f}")
    print(f"Similarity between unrelated sentences: {similarity_13:.4f}")


if __name__ == "__main__":
    main()
