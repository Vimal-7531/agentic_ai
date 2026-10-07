import os
from dotenv import load_dotenv
from anthropic import Anthropic

from router import route_question
from policy_agent import answer_policy_question
from network_agent import answer_network_question
from billing_agent import answer_billing_question


load_dotenv()

client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)


def generate_final_answer(question, result):
    prompt = f"""
You are a support assistant for Prodapt.

Answer the user's question using ONLY the retrieved information below.

Do not invent facts.
Do not add information that is not present in the retrieved information.
Keep the answer concise and direct.
If the retrieved information does not contain the answer, say that the information is not available.

User question:
{question}

Retrieved information:
{result}

Final answer:
"""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.content[0].text.strip()


def answer_question(question):
    routing = route_question(question)
    route = routing["route"]

    if route == "rag":
        result = answer_policy_question(question)
        return generate_final_answer(question, result)

    if route == "database":
        reason = routing.get("reason", "").lower()

        if "network" in reason:
            result = answer_network_question(question)
            return generate_final_answer(question, result)

        if "customer" in reason or "billing" in reason:
            result = answer_billing_question(question)
            return generate_final_answer(question, result)

        result = answer_billing_question(question)
        return generate_final_answer(question, result)

    return "I could not determine how to handle this question."


if __name__ == "__main__":
    questions = [
        "What is the roaming policy for Japan?",
        "Is tower TX-512 operational?",
        "Why are there 5G session drops at TX-512?",
        "What is the balance of CUST-10002?",
        "Are there any billing disputes for CUST-10002?"
    ]

    for question in questions:
        print("\nQuestion:", question)
        print("Answer:")
        print(answer_question(question))