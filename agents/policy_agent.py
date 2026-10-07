import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


INDEX_PATH = PROJECT_ROOT / "data" / "policy_index"

embedding_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

storage_context = StorageContext.from_defaults(
    persist_dir=str(INDEX_PATH)
)

index = load_index_from_storage(
    storage_context,
    embed_model=embedding_model
)

retriever = index.as_retriever(
    similarity_top_k=3
)


def answer_policy_question(question):
    nodes = retriever.retrieve(question)

    if not nodes:
        return {
            "status": "not_found",
            "answer": "No relevant policy information was found.",
            "context": ""
        }

    sections = []

    for node in nodes:
        content = node.get_content().strip()

        if content:
            sections.append(content)

    if not sections:
        return {
            "status": "not_found",
            "answer": "No relevant policy information was found.",
            "context": ""
        }

    context = "\n\n".join(sections)

    return {
        "status": "success",
        "context": context,
        "sources": len(sections)
    }


if __name__ == "__main__":
    questions = [
        "What will data cost in Japan?",
        "What is the roaming policy for Western Europe?",
        "What does the policy say about Travel Pass?"
    ]

    for question in questions:
        print(f"\nQuestion: {question}")

        result = answer_policy_question(question)

        if result["status"] == "success":
            print("\nRetrieved policy sections:")
            print(result["context"])
        else:
            print(result["answer"])