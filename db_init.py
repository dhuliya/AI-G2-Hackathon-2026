import os
import chromadb
from chromadb.utils import embedding_functions

def init_rag_db(rag_store: str = "./rag_store"):
    """
    Initializes the persistent ChromaDB client and creates or retrieves collections
    using the specified embedding function.
    """
    # 1. Initialize persistent Chroma client
    client = chromadb.PersistentClient(path=rag_store)
    
    # 2. Set up the SentenceTransformer embedding function
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="google/embeddinggemma-300m"
    )
    
    # 3. Define target collections
    collection_names = [
        "audios",
        "audio_segments",
        "transcribe_segments",
        "audio_attributes"
    ]
    
    # 4. Create or retrieve collections safely
    collections = {}
    for name in collection_names:
        collections[name] = client.get_or_create_collection(
            name=name, 
            embedding_function=sentence_transformer_ef
        )
        print(f"Collection '{name}' ready.")
        
    return client, collections


def main():
    print("Initializing ChromaDB collections...")
    client, collections = init_rag_db()
    
    # Verification step
    print("\nInitialization Complete. Active collections in database:")
    for name in client.list_collections():
        print(f" - {name.name}")


# Runnable execution block for Jupyter / IPython
if __name__ == "__main__" or "__file__" not in globals():
    main()