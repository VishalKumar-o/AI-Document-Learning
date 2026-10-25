
import pymupdf


class PDFService:
    """Extract text from PDF documents, page by page."""

    def extract_text(self, pdf_path: str) -> list[dict]:
        pages = []

        with pymupdf.open(pdf_path) as document:
            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text").strip()

                if text:
                    pages.append({
                        "page_number": page_number,
                        "text": text
                    })

        return pages
