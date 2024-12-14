# AI Compliance Checker

AI Compliance Checker is a tool designed to process and analyze PDFs for compliance checks. It extracts text, splits the content into chunks, generates embeddings using Hugging Face models, and stores the embeddings in a persistent FAISS index for efficient search and retrieval.

## Features
- **PDF Upload**: Upload PDFs for analysis.
- **Text Extraction**: Extracts text from PDF files.
- **Chunking**: Splits extracted text into smaller chunks for better processing.
- **Embeddings Generation**: Generates embeddings for document chunks using Hugging Face models.
- **Local FAISS Storage**: Stores embeddings in a local FAISS index for quick retrieval.
- **Pinecone Integration**: Optionally stores embeddings in Pinecone for scalable search.

