import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import streamlit as st
from multi_hop_rag import multi_hop_rag


st.title("Multi-hop RAG QA")

question = st.text_input("Ask a question")

if question:

    answer = multi_hop_rag(question)

    st.write(answer)