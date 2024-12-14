import faiss
import numpy as np
import os


def store_in_faiss(embeddings, index_save_path):
    """
    Creates and stores a FAISS vector index using precomputed embeddings.

    :param embeddings: List of embeddings (list of lists or NumPy array).
    :param index_save_path: Full file path to save the FAISS index.
    """
    try:
        # Ensure the directory exists for the FAISS index file
        index_dir = os.path.dirname(index_save_path)
        os.makedirs(index_dir, exist_ok=True)

        # Convert embeddings to a NumPy array
        embedding_matrix = np.array(embeddings, dtype="float32")

        # Create a FAISS index
        dimension = embedding_matrix.shape[1]  # Dimension of embeddings
        index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity
        index.add(embedding_matrix)  # Add embeddings to the FAISS index

        # Save the FAISS index to disk
        faiss.write_index(index, index_save_path)
        print(f"FAISS index saved to {index_save_path}")

    except Exception as e:
        raise ValueError(f"Failed to create and store FAISS index: {str(e)}")
