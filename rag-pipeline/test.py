# ------- THIS SCRIPT IS FOR TESTING MODIFICATIONS ON app.py -------

import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
from dotenv import load_dotenv

load_dotenv('myenv.env')

## load the Groq API key
groq_api_key=os.getenv('GROQ_API_KEY')

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

index_path= 'faiss_index'

if "vector" not in st.session_state: 
    st.session_state.index = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    
st.title("RAG Chatbot Demo")
llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="mixtral-8x7b-32768")

<<<<<<< HEAD
def needs_retrieval(prompt, llm):
  """
  Classifies user intent and determines if retrieval is needed.
  Args:
    prompt: User query as a string.
    llm: LangChain LLM object (e.g., ChatGroq).
  Returns:
    True if retrieval is needed, False otherwise.
  """
  intent_classes = ["informational", "open ended", "factual"]  # Example classes
  classification_prompt = ChatPromptTemplate.from_template(f"""
                                                           Classify the intent of the query, return one of these intent classes only: {intent_classes}. Query: {prompt}
                                                           """)
  
  chain = create_stuff_documents_chain(llm, classification_prompt)
  
  classification_response = chain.invoke({'input':prompt})
  predicted_intent = classification_response["answer"].lower()  # Extract intent

  # Retrieval decision based on intent
  if predicted_intent in ["informational", "factual"]:
    return True  # Retrieval needed for informative or factual queries
  else:
    return False  # No retrieval for open ended queries
    

=======
>>>>>>> 4750db6592cde0a09e06425c56fcaa5fe8ce31be
prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.index.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

prompt=st.text_input("Input you prompt here")

if prompt:
<<<<<<< HEAD
    retrieve = needs_retrieval(prompt, llm)
    if retrieve:
        start=time.process_time()
        response=retrieval_chain.invoke({"input":prompt})
        print("Response time :",time.process_time()-start)
        st.write(response['answer'])

        # With a streamlit expander
        with st.expander("Document Similarity Search"):
            # Find the relevant chunks
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
    else:        
        response = document_chain.invoke({'input': {prompt}})
        st.write(response['answer'])

        
        
        
        
=======
    start=time.process_time()
    response=retrieval_chain.invoke({"input":prompt})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
>>>>>>> 4750db6592cde0a09e06425c56fcaa5fe8ce31be
