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
  df = df.head(len(df) // 2000)
  
  # Chunk documents - the loaded document was already split into 100-word chunks
  def chunk():
    pass
  
  df_texts = df['text'].tolist()
  # Create embeddings for documents
  def embed_documents(cohere_model: str = 'embed-english-v3.0',):
    # Embed texts using CoHere
    document_embeddings = cohere_client.embed(texts=df_texts, model=cohere_model, input_type='search_document')
    return document_embeddings.embeddings

  def index_documents(collection_name, document_embeddings):
    # Initialize ChromaDB client and create a collection
    collection = chroma_client.create_collection(name=collection_name, get_or_create=True)

    # Generate document IDs
    ids = [str(i) for i in range(len(document_embeddings))]

    documents = df_texts
    # Add documents to the collection
    collection.add(
        embeddings=document_embeddings,
        ids=ids,
        documents=documents
    )
  
  document_embeddings = embed_documents(cohere_model='embed-english-v3.0')
  index_documents(collection_name='rag_collection', document_embeddings=document_embeddings)
  
if __name__ == "__main__":
  main()
