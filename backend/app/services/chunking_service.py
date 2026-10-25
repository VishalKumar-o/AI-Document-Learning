
class ChunkingService:
    """Split document text into overlapping chunks."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        if chunk_size <= 0:
            raise ValueError("Chunk size must be greater than zero.")

        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(
                "Chunk overlap must be non-negative and smaller than chunk size."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        text = text.strip()

        if not text:
            return []

        chunks = []
        start = 0
        step = self.chunk_size - self.chunk_overlap

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            start += step

        return chunks

    def create_page_chunks(self, pages: list[dict]) -> list[dict]:
        all_chunks = []

        for page in pages:
            page_chunks = self.split_text(page["text"])

            for chunk_number, chunk_text in enumerate(
                page_chunks,
                start=1
            ):
                all_chunks.append({
                    "page_number": page["page_number"],
                    "chunk_number": chunk_number,
                    "text": chunk_text
                })

        return all_chunks
