
from app.services.document_processing_service import DocumentProcessingService


def main():
    processor = DocumentProcessingService()

    pdf_path = input("Enter the PDF file path: ").strip()

    chunks = processor.process_pdf(pdf_path)

    print(f"\nTotal chunks created: {len(chunks)}")

    if chunks:
        first_chunk = chunks[0]

        print("\nFirst chunk ID:", first_chunk["chunk_id"])
        print("Source page:", first_chunk["page_number"])
        print("Chunk number on page:", first_chunk["chunk_number"])
        print("\nFirst 500 characters:")
        print(first_chunk["text"][:500])


if __name__ == "__main__":
    main()
