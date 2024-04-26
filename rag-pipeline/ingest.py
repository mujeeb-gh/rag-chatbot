from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS


model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

csv_path = "data/sub_chunk_kb_acl-100k.csv"

loader=CSVLoader(csv_path)
docs = loader.load()
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
final_documents=text_splitter.split_documents(docs[:50])
# find out if langchain's implementation of faiss uses HNSW
index=FAISS.from_documents(final_documents,embeddings)

index.save_local("faiss_index")

print("Pre-built index created and saved successfully!")