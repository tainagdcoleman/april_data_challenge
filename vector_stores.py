"""
Tiny vector-store abstraction.

The store implements:

    store = make_store("chroma", collection="ndp_datasets", dim=384)
    store.reset()
    store.add(ids, documents, metadatas, embeddings)
    hits = store.search(query_vector, k=3)
    # hits -> list[{"id": str, "document": str, "metadata": dict, "distance": float}]

The notebook picks a backend via a VECTOR_BACKEND variable; everything
downstream (retrieval, chatbot) is unaware of which backend is active.

Backends:
- ChromaStore — ChromaDB, file-based persistent client.
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol


class VectorStore(Protocol):
    def reset(self) -> None: ...
    def add(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> None: ...
    def search(self, query_vector: List[float], k: int = 3) -> List[Dict[str, Any]]: ...
    def count(self) -> int: ...


# ---------------------------------------------------------------------------
# Chroma
# ---------------------------------------------------------------------------

class ChromaStore:
    def __init__(self, collection: str, dim: int, path: str = "./chroma_db"):
        import chromadb

        self.collection_name = collection
        self.dim = dim
        self.client = chromadb.PersistentClient(path=path)
        self.collection = None  # set in reset()
        self.mode = f"local ({path})"

    def reset(self) -> None:
        existing = [c.name for c in self.client.list_collections()]
        if self.collection_name in existing:
            self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(name=self.collection_name)

    def _ensure_collection(self):
        if self.collection is None:
            self.collection = self.client.get_or_create_collection(self.collection_name)

    def add(self, ids, documents, metadatas, embeddings) -> None:
        self._ensure_collection()
        # Chroma rejects empty-dict metadatas; replace with a placeholder.
        safe_meta = [m if m else {"_": ""} for m in metadatas]
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=safe_meta,
            embeddings=embeddings,
        )

    def search(self, query_vector, k=3):
        self._ensure_collection()
        res = self.collection.query(query_embeddings=[query_vector], n_results=k)
        hits = []
        for doc, meta, _id, dist in zip(
            res["documents"][0], res["metadatas"][0], res["ids"][0], res["distances"][0]
        ):
            hits.append({"id": _id, "document": doc, "metadata": meta or {}, "distance": dist})
        return hits

    def count(self) -> int:
        self._ensure_collection()
        return self.collection.count()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_store(backend: str, collection: str, dim: int) -> VectorStore:
    backend = backend.lower()
    if backend == "chroma":
        return ChromaStore(collection=collection, dim=dim)
    raise ValueError(f"Unknown backend: {backend!r}. Only 'chroma' is supported.")
