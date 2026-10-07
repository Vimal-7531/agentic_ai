import re


def classify_question(question):
    text = question.lower()

    policy_terms = [
        "policy",
        "rule",
        "rules",
        "procedure",
        "travel pass",
        "roaming",
        "zone a",
        "zone b",
        "zone c",
        "zone d"
    ]

    if any(term in text for term in policy_terms):
        return "policy"

    network_terms = [
        "tower",
        "site",
        "cell",
        "incident",
        "outage",
        "packet loss",
        "signal",
        "throughput",
        "5g",
        "4g"
    ]

    if any(term in text for term in network_terms):
        return "network"

    billing_terms = [
        "customer",
        "account",
        "balance",
        "bill",
        "billing",
        "charge",
        "credit",
        "dispute",
        "subscription"
    ]

    if any(term in text for term in billing_terms):
        return "billing"

    return "unknown"

def route_question(question):
    category = classify_question(question)

    if category == "network":
        return {
            "route": "database",
            "reason": "Question requires network operational data."
        }

    if category == "billing":
        return {
            "route": "database",
            "reason": "Question requires customer or billing data."
        }

    if category == "policy":
        return {
            "route": "rag",
            "reason": "Question requires policy information."
        }

    return {
        "route": "unknown",
        "reason": "Question could not be confidently classified."
    }


if __name__ == "__main__":
    questions = [
        "What is the roaming policy for Japan?",
        "Is tower TX-512 operational?",
        "Why are there 5G session drops?",
        "What is the balance of CUST-10002?",
        "What is the billing dispute policy?"
    ]

    for question in questions:
        result = route_question(question)

        print(f"\nQuestion: {question}")
        print(f"Route: {result['route']}")
        print(f"Reason: {result['reason']}")