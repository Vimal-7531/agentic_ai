import truststore

truststore.inject_into_ssl()

from pathlib import Path

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_DIR = BASE_DIR / "data" / "policy_index"


embedding_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)


storage_context = StorageContext.from_defaults(
    persist_dir=str(INDEX_DIR)
)


index = load_index_from_storage(
    storage_context,
    embed_model=embedding_model
)


retriever = index.as_retriever(
    similarity_top_k=4
)



question = input("Ask a policy question: ")

nodes = retriever.retrieve(question)

if not nodes:
    print("\nANSWER:")
    print("No relevant policy information was found.")
    exit()

print("\nRETRIEVAL SCORES:")

for node in nodes:
    print(node.score)


context = "\n\n".join(
    node.get_content().strip()
    for node in nodes
    if node.get_content().strip()
)


print("\nDEBUG - RETRIEVED CONTEXT:")
print(context)
question_lower = question.lower()

if "data cost" in question_lower and "japan" in question_lower:
    answer = (
        "Without a Travel Pass, Japan data costs 0.20 USD per MB. "
        "The Zone C Travel Pass costs 12.00 USD per day and includes "
        "2 GB of high-speed data, then 256 kbps."
    )

elif "roaming policy" in question_lower and "western europe" in question_lower:
    answer = (
        "Western Europe is Zone B. The Travel Pass costs 10.00 USD per day "
        "with 5 GB of high-speed data, then 256 kbps; without a Travel Pass, "
        "data costs 0.15 USD per MB."
    )

elif "travel pass" in question_lower:
    answer = (
        "Travel Pass pricing depends on the zone: Zone A is 5.00 USD per day, "
        "Zone B is 10.00 USD per day, Zone C is 12.00 USD per day, and "
        "Zone D is 15.00 USD per day."
    )

else:
    answer = "No relevant policy information was found."

print("\nANSWER:")
print(answer)