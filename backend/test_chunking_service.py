
from app.services.chunking_service import ChunkingService


def main():
    chunker = ChunkingService(
        chunk_size=1000,
        chunk_overlap=200
    )

    sample_text = (
        "Natural Language Processing helps computers understand "
        "human language. " * 100
    )

    chunks = chunker.split_text(sample_text)

    print("Original text length:", len(sample_text))
    print("Number of chunks:", len(chunks))

    for index, chunk in enumerate(chunks, start=1):
        print(f"\nChunk {index}")
        print("Length:", len(chunk))
        print("Beginning:", chunk[:80])


if __name__ == "__main__":
    main()
