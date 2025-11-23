from typing import List, Dict, Any
from qdrant_client import QdrantClient, models
from data_loader import EMBED_DIM


class QdrantStorage:
    def __init__(
        self,
        collection_name: str = "rag_pdf",
        host: str = "localhost",
        port: int = 6333,
    ):
        self.collection_name = collection_name
        self.client = QdrantClient(host=host, port=port)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        collections = self.client.get_collections().collections
        if not any(c.name == self.collection_name for c in collections):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=EMBED_DIM,
                    distance=models.Distance.COSINE,
                ),
            )

    def upsert(
        self,
        ids: List[str],
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
    ) -> None:
        points = [
            models.PointStruct(
                id=ids[i],
                vector=vectors[i],
                payload=payloads[i],
            )
            for i in range(len(ids))
        ]

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

    def search(self, query_vec: List[float], top_k: int = 5) -> Dict[str, List[str]]:
        res = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vec,
            limit=top_k,
            with_payload=True,
        )

        points = res.points

        contexts = [p.payload.get("text", "") for p in points]
        sources = [p.payload.get("source", "") for p in points]

        return {"contexts": contexts, "sources": sources}
