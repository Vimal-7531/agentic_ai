
from typing import TypedDict, Any

from langgraph.graph import StateGraph, START, END

from router import route_question
from policy_agent import answer_policy_question
from network_agent import answer_network_question
from billing_agent import answer_billing_question


class SupervisorState(TypedDict, total=False):
    question: str
    route: str
    reason: str
    answer: Any


def classify_request(state: SupervisorState):
    question = state["question"]

    routing = route_question(question)

    return {
        "route": routing["route"],
        "reason": routing.get("reason", "")
    }


def policy_worker(state: SupervisorState):
    question = state["question"]

    result = answer_policy_question(question)

    return {
        "answer": result
    }


def network_worker(state: SupervisorState):
    question = state["question"]

    result = answer_network_question(question)

    return {
        "answer": result
    }


def billing_worker(state: SupervisorState):
    question = state["question"]

    result = answer_billing_question(question)

    return {
        "answer": result
    }


def route_to_worker(state: SupervisorState):
    route = state["route"]
    reason = state.get("reason", "").lower()

    if route == "rag":
        return "policy"

    if route == "database":

        if "network" in reason:
            return "network"

        if "customer" in reason or "billing" in reason:
            return "billing"

        return "billing"

    return "billing"


builder = StateGraph(SupervisorState)

builder.add_node("classify", classify_request)
builder.add_node("policy", policy_worker)
builder.add_node("network", network_worker)
builder.add_node("billing", billing_worker)

builder.add_edge(START, "classify")

builder.add_conditional_edges(
    "classify",
    route_to_worker,
    {
        "policy": "policy",
        "network": "network",
        "billing": "billing"
    }
)

builder.add_edge("policy", END)
builder.add_edge("network", END)
builder.add_edge("billing", END)

supervisor = builder.compile()


def answer_question(question: str):
    result = supervisor.invoke({
        "question": question
    })

    return result["answer"]


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

        try:
            print(answer_question(question))
        except Exception as e:
            print("Error:", e)