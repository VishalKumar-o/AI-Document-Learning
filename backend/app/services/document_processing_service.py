
from app.services.pdf_service import PDFService
from app.services.text_processing_service import TextProcessingService
from app.services.chunking_service import ChunkingService


class DocumentProcessingService:
    """Extract, clean, and split PDF text into searchable chunks."""

    def __init__(self):
        self.pdf_service = PDFService()
        self.text_processor = TextProcessingService()
        self.chunking_service = ChunkingService(
            chunk_size=1000,
            chunk_overlap=200
        )

    def process_pdf(self, pdf_path: str) -> list[dict]:
        extracted_pages = self.pdf_service.extract_text(pdf_path)

        processed_pages = []

        for page in extracted_pages:
            cleaned_text = self.text_processor.clean_text(page["text"])

            if cleaned_text:
                processed_pages.append({
                    "page_number": page["page_number"],
                    "text": cleaned_text
                })

        chunks = self.chunking_service.create_page_chunks(processed_pages)

        for chunk_id, chunk in enumerate(chunks, start=1):
            chunk["chunk_id"] = chunk_id

        return chunks
