import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

def ask_llm(question, context):

    prompt = f"""
You are a multi-hop QA system.

Use the context to answer the question.

Instructions:
1. Look for relevant facts in the context
2. Combine information if needed
3. If the answer is not in the context, say:
"I do not have enough knowledge to answer this question."

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content