from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_docs(docs, chunk_size=800, chunk_overlap=50):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    split_docs = text_splitter.split_documents(docs)

    return split_docs
