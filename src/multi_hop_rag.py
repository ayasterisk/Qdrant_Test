from src.retriever import retrieve
from src.llm import ask_llm


def multi_hop_rag(question):

    docs1 = retrieve(question)

    context1 = "\n".join([d["text"] for d in docs1])

    sub_prompt = f"""
Question:
{question}

Context:
{context1}

Generate a follow-up question.
"""

    sub_question = ask_llm(sub_prompt)

    docs2 = retrieve(sub_question)

    context2 = "\n".join([d["text"] for d in docs2])

    final_prompt = f"""
Answer the question.

Question:
{question}

Context:
{context1}
{context2}
"""

    answer = ask_llm(final_prompt)

    return answer