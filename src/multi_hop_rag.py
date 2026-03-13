from retriever import retrieve
from llm import ask_llm


def multi_hop_rag(question):

    docs1 = retrieve(question)

    context1 = " ".join([d["text"] for d in docs1])

    query2 = question + " " + context1[:200]

    docs2 = retrieve(query2)

    context2 = " ".join([d["text"] for d in docs2])

    context = context1 + " " + context2

    answer = ask_llm(question, context)

    return answer