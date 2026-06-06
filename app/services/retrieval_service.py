from typing import Any, Dict

from app.config.settings import settings
from app.services.embedding_service import EmbeddingService
from app.services.qdrant_service import QdrantService


class RetrievalService:
    def __init__(self) -> None:
        print("[RetrievalService] Initializing...", flush=True)
        self.embedding_service = EmbeddingService()
        self.qdrant_service = QdrantService()
        print("[RetrievalService] Initialized.", flush=True)

    def retrieve_for_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        case_id = plan["case_id"]
        retrieval_queries = plan["retrieval_queries"]

        print(f"[RetrievalService] case_id={case_id}", flush=True)
        print(f"[RetrievalService] num_queries={len(retrieval_queries)}", flush=True)

        retrieval_results = {}

        for field_name, query_text in retrieval_queries.items():
            print(f"[RetrievalService] Embedding field={field_name}", flush=True)
            query_vector = self.embedding_service.embed_text(query_text)
            print(f"[RetrievalService] Embedded field={field_name}", flush=True)

            print(f"[RetrievalService] Searching Qdrant field={field_name}", flush=True)
            chunks = self.qdrant_service.search(
                query_vector=query_vector,
                case_id=case_id,
                top_k=settings.top_k,
            )
            print(
                f"[RetrievalService] Qdrant returned {len(chunks)} chunks for field={field_name}",
                flush=True,
            )

            retrieval_results[field_name] = {
                "query": query_text,
                "chunks": chunks,
            }

        print("[RetrievalService] Retrieval complete.", flush=True)

        return retrieval_results