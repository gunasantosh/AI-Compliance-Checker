import os
from langchain_community.document_loaders import PyPDFDirectoryLoader


def extract_text_from_pdf():
    """
    Extract text from all PDFs stored in the 'docs' folder.
    :return: Combined extracted text from all PDFs as a single string.
    """
    # Hardcoded relative path to the 'docs' directory
    docs_directory = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "docs")
    )

    # Ensure the directory exists
    if not os.path.exists(docs_directory):
        raise FileNotFoundError(f"Directory '{docs_directory}' does not exist.")

    # Load and process PDFs in the directory
    file_loader = PyPDFDirectoryLoader(docs_directory)
    docs = file_loader.load()

    # Combine text content from all documents
    return docs
