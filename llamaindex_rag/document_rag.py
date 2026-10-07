import os
from pathlib import Path

from dotenv import load_dotenv

from llama_index.core import (
    Settings,
    StorageContext,
    load_index_from_storage,
    get_response_synthesizer,
)

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic
BASE_DIR = Path(__file__).resolve().parent.parent

ENV_PATH = BASE_DIR / ".env"
INDEX_DIR = BASE_DIR / "data" / "policy_index"
DB_PATH = BASE_DIR / "data" / "telecom_ops.db"

load_dotenv(dotenv_path=ENV_PATH)

print("ENV FILE:", ENV_PATH)
print("ANTHROPIC KEY LOADED:", bool(os.getenv("ANTHROPIC_API_KEY")))
print("INDEX DIR:", INDEX_DIR)
print("INDEX EXISTS:", INDEX_DIR.exists())

load_dotenv(dotenv_path=ENV_PATH)

print("ENV FILE:", ENV_PATH)
print("ANTHROPIC KEY LOADED:", bool(os.getenv("ANTHROPIC_API_KEY")))


embedding_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

llm = Anthropic(
    model="claude-sonnet-4-5",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=500,
)


Settings.embed_model = embedding_model
Settings.llm = llm


storage_context = StorageContext.from_defaults(
    persist_dir=str(INDEX_DIR)
)


index = load_index_from_storage(
    storage_context,
    embed_model=embedding_model,
)


response_synthesizer = get_response_synthesizer(
    llm=llm,
    response_mode="compact",
)


query_engine = index.as_query_engine(
    similarity_top_k=3,
    response_synthesizer=response_synthesizer,
)

def answer_policy_question(question: str) -> str:
    response = query_engine.query(question)
    return str(response)

if __name__ == "__main__":

    print("\nTesting Anthropic connection...")

    test = llm.complete(
        "Reply with exactly these three words: Anthropic connection works"
    )

    print("ANTHROPIC TEST:")
    print(str(test))

    question = input("\nAsk a policy question: ")

    answer = answer_policy_question(question)

    print("\nANSWER:")
    print(answer)