
from app.services.retrieval_service import RetrievalService


def main():
    retrieval_service = RetrievalService()

    sample_chunks = [
        {
            "chunk_id": 1,
            "page_number": 10,
            "chunk_number": 1,
            "text": (
                "Database normalization organizes relational tables "
                "to reduce data redundancy and improve data integrity."
            ),
        },
        {
            "chunk_id": 2,
            "page_number": 25,
            "chunk_number": 1,
            "text": (
                "A primary key uniquely identifies each record in a table."
            ),
        },
        {
            "chunk_id": 3,
            "page_number": 40,
            "chunk_number": 1,
            "text": (
                "Photosynthesis is the process plants use to convert "
                "light energy into chemical energy."
            ),
        },
    ]

    retrieval_service.add_document(
        document_id="dbms-book",
        chunks=sample_chunks,
    )

    question = "Why is database normalization used?"

    results = retrieval_service.search(
        question=question,
        top_k=2,
        document_id="dbms-book",
    )

    print("\nQuestion:", question)
    print("\nRetrieved chunks:")

    for result in results:
        print("\nPage:", result["page_number"])
        print("Chunk ID:", result["chunk_id"])
        print("Similarity:", round(result["score"], 4))
        print("Text:", result["text"])

    print(
        "\nDocument removed:",
        retrieval_service.remove_document("dbms-book"),
    )


if __name__ == "__main__":
    main()
