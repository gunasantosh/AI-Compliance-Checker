from langchain_pinecone import PineconeVectorStore
from src.config.settings import settings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

from pinecone import Pinecone


def create_vectorstore(split_docs, index_name):
    """
    Creates a Pinecone vector store from a list of split documents.

    :param split_docs: List of documents to store as vectors.
    :param embeddings: Embedding model instance.
    :param index_name: Name of the Pinecone index.
    :return: PineconeVectorStore instance.
    """
    # Initialize Pinecone with API key
    # pinecone = Pinecone(api_key=settings.PINECONE_API_KEY)

    # Load documents into PineconeVectorStore

    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=settings.HUGGINGFACE_API_KEY,
        model_name="sentence-transformers/all-MiniLM-l6-v2",
    )

    vector_store = PineconeVectorStore.from_documents(
        split_docs,
        embedding=embeddings,
        index_name=index_name,
        pinecone_api_key=settings.PINECONE_API_KEY,
    )
    return vector_store
