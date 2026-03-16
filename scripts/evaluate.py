import os
import sys
import re
from collections import Counter
from datasets import load_from_disk

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from multi_hop_rag import multi_hop_rag
from retriever import retrieve


def normalize_answer(s):

    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'[^\w\s]', '', s)

    return " ".join(s.split())


def exact_match(pred, gt):

    return normalize_answer(pred) == normalize_answer(gt)


def f1_score(pred, gt):

    pred_tokens = normalize_answer(pred).split()
    gt_tokens = normalize_answer(gt).split()

    common = Counter(pred_tokens) & Counter(gt_tokens)

    num_same = sum(common.values())

    if num_same == 0:
        return 0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)

    return 2 * precision * recall / (precision + recall)


def supporting_fact_accuracy(retrieved_docs, gold_supports):

    retrieved_titles = [doc["title"] for doc in retrieved_docs]

    correct = 0

    for title, _ in gold_supports:
        if title in retrieved_titles:
            correct += 1

    return correct / len(gold_supports)


def evaluate(dataset_path="hotpot_mini_1k", num_samples=100):

    dataset = load_from_disk(dataset_path)

    em_total = 0
    f1_total = 0
    sp_total = 0

    for i, sample in enumerate(dataset):

        if i >= num_samples:
            break

        question = sample["question"]
        gt_answer = sample["answer"]
        supporting_facts = sample["supporting_facts"]

        # retrieve documents
        docs = retrieve(question)

        # RAG answer
        pred_answer = multi_hop_rag(question)

        em = exact_match(pred_answer, gt_answer)
        f1 = f1_score(pred_answer, gt_answer)
        sp = supporting_fact_accuracy(docs, supporting_facts)

        em_total += em
        f1_total += f1
        sp_total += sp

        print("\nQuestion:", question)
        print("GT:", gt_answer)
        print("Pred:", pred_answer)
        print("EM:", em, "F1:", f1, "SP:", sp)

    n = num_samples

    print("\n===== FINAL RESULTS =====")

    print("Exact Match:", em_total / n)
    print("F1 Score:", f1_total / n)
    print("Supporting Fact Accuracy:", sp_total / n)


if __name__ == "__main__":

    evaluate()