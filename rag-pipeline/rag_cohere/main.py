from openai import OpenAI
import streamlit as st
import os
import cohere
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
st.title("ChatGPT-like clone")

# client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

cohere_api_key = os.environ['COHERE_API_KEY']
client = cohere.Client(cohere_api_key)



if "openai_model" not in st.session_state:
    st.session_state["cohere_model"] = "command"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["text"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "USER", "text": prompt})
    with st.chat_message("USER"):
        st.markdown(prompt)

    with st.chat_message("CHATBOT"):
        response = client.chat(
            model=st.session_state["cohere_model"],
            message=prompt,
            chat_history=[{"role": msg['role'], "text": msg['text']} for msg in st.session_state.messages]
        )
        st.write(response.text)
    st.session_state.messages.append({"role": "CHATBOT", "text": response.text})
