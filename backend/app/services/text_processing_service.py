
import re


class TextProcessingService:
    """Clean extracted document text before chunking and retrieval."""

    def clean_text(self, text: str) -> str:
        if not text:
            return ""

        # Normalize different line endings.
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Replace tabs and repeated spaces with a single space.
        text = re.sub(r"[ \t]+", " ", text)

        # Remove spaces at the beginning and end of each line.
        lines = [line.strip() for line in text.split("\n")]

        # Keep at most one blank line between blocks of text.
        text = "\n".join(lines)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def normalize_for_search(self, text: str) -> str:
        """Create a lowercase version for simple text matching."""
        return self.clean_text(text).lower()
