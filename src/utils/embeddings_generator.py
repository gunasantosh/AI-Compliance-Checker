from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from src.config.settings import settings


def generate_embeddings(split_docs):
    """
    Generate embeddings for the provided documents using Hugging Face Inference API.

    :param split_docs: List of split documents (LangChain Document objects or plain text strings).
    :return: List of embeddings for the split documents.
    """
    # Initialize the Hugging Face Inference API embeddings model
    embeddings_model = HuggingFaceInferenceAPIEmbeddings(
        api_key=settings.HUGGINGFACE_API_KEY,
        model_name="sentence-transformers/all-MiniLM-l6-v2",
    )

    # Check if input documents are plain text strings or LangChain documents
    texts = [
        doc.page_content if hasattr(doc, "page_content") else str(doc)
        for doc in split_docs
    ]

    # Generate embeddings for the texts
    try:
        embeddings = embeddings_model.embed_documents(texts)
        return embeddings
    except Exception as e:
        raise ValueError(f"Failed to generate embeddings: {str(e)}")
