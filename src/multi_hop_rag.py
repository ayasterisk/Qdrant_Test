from retriever import retrieve
from llm import ask_llm


def multi_hop_rag(question):

    # Hop 1
    docs1 = retrieve(question)

    context1 = "\n".join([d["text"] for d in docs1])

    sub_prompt = f"""
Given the question:

{question}

Context:
{context1}

Generate a follow-up question needed to answer the original question.
"""

    sub_question = ask_llm(sub_prompt)

    print("Sub-question:", sub_question)

    # Hop 2
    docs2 = retrieve(sub_question)

    context2 = "\n".join([d["text"] for d in docs2])

    final_prompt = f"""
Answer the question using the context.

Question:
{question}

Context:
{context1}
{context2}
"""

    answer = ask_llm(final_prompt)

    return answer