import streamlit as st
from src.multi_hop_rag import multi_hop_rag


st.title("Multi-hop RAG QA")

question = st.text_input("Ask a question")

if question:

    answer = multi_hop_rag(question)

    st.write(answer)