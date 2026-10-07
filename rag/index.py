import truststore

truststore.inject_into_ssl()

from pathlib import Path

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
INDEX_DIR = BASE_DIR / "data" / "policy_index"


embedding_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

Settings.embed_model = embedding_model

documents = SimpleDirectoryReader(
    input_dir=str(DOCUMENTS_DIR)
).load_data()


splitter = SentenceSplitter(
    chunk_size=500,
    chunk_overlap=50
)


nodes = splitter.get_nodes_from_documents(documents)


index = VectorStoreIndex(
    nodes,
    embed_model=embedding_model
)


INDEX_DIR.mkdir(
    parents=True,
    exist_ok=True
)

index.storage_context.persist(
    persist_dir=str(INDEX_DIR)
)


print(f"Loaded {len(documents)} documents.")
print(f"Created {len(nodes)} chunks.")
print(f"Policy index created at: {INDEX_DIR}")