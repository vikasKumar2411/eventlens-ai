import sys
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from app.services.qdrant_service import QdrantService


def main():
    service = QdrantService()

    points, next_page = service.client.scroll(
        collection_name=service.collection_name,
        limit=500,
        with_payload=True,
        with_vectors=False,
    )

    case_ids = []

    for point in points:
        payload = point.payload or {}

        case_id = (
            payload.get("case_id")
            or payload.get("filing_id")
            or payload.get("document_id")
        )

        event_type = payload.get("event_type")

        if case_id:
            case_ids.append(case_id)

    counts = Counter(case_ids)

    print("\nAvailable case IDs:")
    for case_id, count in counts.most_common(30):
        print(f"{case_id}: {count} chunks")


if __name__ == "__main__":
    main()