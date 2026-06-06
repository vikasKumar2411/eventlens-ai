from sentence_transformers import SentenceTransformer

from app.config.settings import settings


class EmbeddingService:
    _model = None

    def __init__(self) -> None:
        if EmbeddingService._model is None:
            EmbeddingService._model = SentenceTransformer(settings.embedding_model)

        self.model = EmbeddingService._model

    def embed_text(self, text: str) -> list[float]:
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()