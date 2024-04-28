import pandas as pd
import cohere
import os
import chromadb
from typing import List
from chromadb import PersistentClient
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

cohere_api_key = os.environ['COHERE_API_KEY']
cohere_client = cohere.Client(cohere_api_key)

CHROMA_CLIENT = 'chroma_vectorstore'
chroma_client = PersistentClient(path=CHROMA_CLIENT)


CSV_PATH = 'data/sub_chunk_kb_acl-100k.csv'



def main():
  
  # Load documents
  df = pd.read_csv(CSV_PATH)
  df = df.head()
  
  # Chunk documents - the loaded document was already split into 100-word chunks
  def chunk():
    pass
  
  df_texts = df['text'].tolist()
  # Create embeddings for documents
  def embed_and_index_documents(
    # df_texts: List[str],
    cohere_model: str = 'embed-english-v3.0',
    collection_name: str = "rag_collection"
):
    """
    Embeds text using a CoHere model, creates a ChromaDB collection, and adds documents to it.

    Args:
        df_texts (List[str]): List of texts to embed and add to the collection.
        cohere_model (str): CoHere model name to use for embedding. Default is 'embed-english-v3.0'.
        chroma_path (str): Path to the ChromaDB storage. Default is "chroma_vectorstore".
        collection_name (str): Name of the collection to create. Default is "rag_collection".
    """
    # Embed texts using CoHere
    document_embeddings = cohere_client.embed(texts=df_texts, model=cohere_model, input_type='search_document')

    # Generate document IDs
    ids = [str(i) for i in range(len(document_embeddings.embeddings))]

    documents = df_texts

    # Initialize ChromaDB client and create a collection
    collection = chroma_client.create_collection(name=collection_name, get_or_create=True)

    # Add documents to the collection
    collection.add(
        embeddings=document_embeddings.embeddings,
        ids=ids,
        documents=documents
    )
  
  embed_and_index_documents(cohere_model='embed-english-v3.0', collection_name='rag_collection')
  
if __name__ == "__main__":
  main()
