
from app.services.keyword_search_service import KeywordSearchService


service = KeywordSearchService()

chunks = [
    {
        "chunk_id": 1,
        "page_number": 5,
        "text": "Alice followed the White Rabbit down the rabbit hole."
    },
    {
        "chunk_id": 2,
        "page_number": 12,
        "text": "The Mad Hatter hosted a tea party with the March Hare."
    },
    {
        "chunk_id": 3,
        "page_number": 20,
        "text": "The Queen of Hearts ordered the execution of the prisoners."
    },
]

question = "Who did Alice follow?"

results = service.search(
    question=question,
    chunks=chunks,
    top_k=3,
)

print("Question:", question)
print("\nSearch results:")

for result in results:
    print(
        f"Page {result['page_number']} | "
        f"Score: {result['keyword_score']:.4f}"
    )
    print(result["text"])
    print()
