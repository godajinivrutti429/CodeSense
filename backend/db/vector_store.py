"""ChromaDB wrapper: one collection per indexed repo (namespacing = per-user/repo isolation)."""
import os
from typing import Dict, List

import chromadb
from chromadb.config import Settings

CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "/data/chroma")

_client = None


def get_client():
    global _client
    if _client is None:
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def get_or_create_collection(name: str):
    client = get_client()
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def add_chunks(collection_name: str, ids: List[str], embeddings: List[List[float]],
               documents: List[str], metadatas: List[Dict]):
    collection = get_or_create_collection(collection_name)
    # Chroma has a batch size limit on some backends; chunk the insert to be safe.
    BATCH = 500
    for i in range(0, len(ids), BATCH):
        collection.add(
            ids=ids[i:i + BATCH],
            embeddings=embeddings[i:i + BATCH],
            documents=documents[i:i + BATCH],
            metadatas=metadatas[i:i + BATCH],
        )


def query_collection(collection_name: str, query_embedding: List[float], top_k: int = 5):
    collection = get_or_create_collection(collection_name)
    return collection.query(query_embeddings=[query_embedding], n_results=top_k)


def delete_collection(collection_name: str):
    client = get_client()
    client.delete_collection(name=collection_name)


def list_collections() -> List[str]:
    client = get_client()
    return [c.name for c in client.list_collections()]
