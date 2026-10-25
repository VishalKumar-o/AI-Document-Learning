
from app.services.text_processing_service import TextProcessingService


def main():
    processor = TextProcessingService()

    sample_text = """
        Artificial   Intelligence


        NLP helps computers understand language.
        It can support document question answering.
    """

    cleaned_text = processor.clean_text(sample_text)
    normalized_text = processor.normalize_for_search(sample_text)

    print("Original text:")
    print(repr(sample_text))

    print("\nCleaned text:")
    print(repr(cleaned_text))

    print("\nNormalized text:")
    print(repr(normalized_text))


if __name__ == "__main__":
    main()
