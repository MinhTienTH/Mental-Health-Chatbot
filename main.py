import streamlit as st
from app import create_chain, create_vector_db

st.title("Mental Health Chatbot")

button = st.button("Create a knowledge base")
if button:
    pass

question = st.text_input("Enter your question:")

if question:
    chain = create_chain()
    response = chain(question)
    st.header("Answer: ")
    st.write(response["result"])


