from ingest_c import cohere_client, chroma_client
from chromadb import PersistentClient

def retrieve_context(query: str, n_docs: int):
  """
  Retrieves contextually relevant docs based on user query.
  Args:
    query: User query as a string.
    n_docs: Number of relevant docs to retrieve.
  Returns:
    List of relevant docs.
  """
  collection = chroma_client.get_collection('rag_collection')
  query_embeddings = cohere_client.embed(texts=[query], model='embed-english-v3.0', input_type='search_query')

  retrieved_docs = collection.query(
    query_embeddings=query_embeddings.embeddings,
    n_results= n_docs,
  )
  return retrieved_docs['documents'][0]

def classify_intent():
  pass


# retrieved_docs = retrieve_context("What is tvshow", 5)
# print(retrieved_docs[0][:len(retrieved_docs[0]) + 1])

