from app.services.pdf_service import PDFService


def main():
    pdf_service = PDFService()

    pdf_path = input("Enter the PDF file path: ").strip()

    pages = pdf_service.extract_text(pdf_path)

    print(f"\nTotal pages with extracted text: {len(pages)}")

    if pages:
        print("\nFirst page number:", pages[0]["page_number"])
        print("\nFirst 500 characters:")
        print(pages[0]["text"][:500])
    else:
        print("No extractable text was found.")


if __name__ == "__main__":
    main()
