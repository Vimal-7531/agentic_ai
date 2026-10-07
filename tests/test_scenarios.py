from llamaindex_rag.document_rag import answer_policy_question
from llamaindex_rag.sql_semantic_search import answer_sql_question


def test_policy_rag():
    print("\n==============================")
    print("TEST 1 - POLICY RAG")
    print("==============================")

    question = "What will data cost in Japan?"

    result = answer_policy_question(question)

    print("QUESTION:")
    print(question)

    print("\nRESULT:")
    print(result)


def test_semantic_sql():
    print("\n==============================")
    print("TEST 2 - SEMANTIC SQL")
    print("==============================")

    question = "What is the latest packet loss for TX-512?"

    result = answer_sql_question(question)

    print("QUESTION:")
    print(question)

    print("\nRESULT:")
    print(result)


def test_billing_sql():
    print("\n==============================")
    print("TEST 3 - BILLING SQL")
    print("==============================")

    question = "Does CUST-10002 have any duplicate charges?"

    result = answer_sql_question(question)

    print("QUESTION:")
    print(question)

    print("\nRESULT:")
    print(result)


if __name__ == "__main__":

    print("\nPRODAPT AGENTIC AI - COMPONENT TESTS")

    test_policy_rag()

    test_semantic_sql()

    test_billing_sql()

    print("\n==============================")
    print("TESTING COMPLETE")
    print("==============================")