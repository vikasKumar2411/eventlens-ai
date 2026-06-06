from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from app.config.settings import settings


class QdrantService:
    def __init__(self) -> None:
        self.client = QdrantClient(url=settings.qdrant_url)
        self.collection_name = settings.qdrant_collection

    def search(
        self,
        query_vector: list[float],
        case_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        limit = top_k or settings.top_k

        query_filter = None
        if case_id:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="case_id",
                        match=MatchValue(value=case_id),
                    )
                ]
            )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

        results = response.points

        chunks = []

        for item in results:
            payload = item.payload or {}

            chunks.append(
                {
                    "id": str(item.id),
                    "score": item.score,
                    "case_id": payload.get("case_id"),
                    "company_name": payload.get("company_name"),
                    "symbol": payload.get("symbol"),
                    "event_type": payload.get("event_type"),
                    "section_title": (
                        payload.get("section_title")
                        or payload.get("sec_item_title")
                        or payload.get("heading_text")
                    ),
                    "chunk_text": payload.get("chunk_text") or payload.get("text"),
                    "payload": payload,
                }
            )

        return chunks