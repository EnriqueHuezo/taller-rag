"""Servicios modulares de generación de embeddings (Gemini y OpenAI)."""

import logging
from abc import ABC, abstractmethod
from typing import Sequence

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


class BaseEmbeddingService(ABC):
    """Contrato base para proveedores de embeddings."""

    @abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Genera vectores para un lote de textos."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Genera vector para una consulta única de búsqueda."""
        pass


class GeminiEmbeddingService(BaseEmbeddingService):
    """Generación de embeddings con Google GenAI SDK (gemini-embedding-001 o text-embedding-004)."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-001",
        dimension: int = 768,
    ) -> None:
        if not api_key:
            raise ValueError(
                "Falta la clave GOOGLE_API_KEY para inicializar GeminiEmbeddingService."
            )
        from google import genai

        self.client = genai.Client(api_key=api_key)
        self.model = model.removeprefix("models/")
        self.dimension = dimension

    def embed_texts(self, texts: Sequence[str], batch_size: int = 50) -> list[list[float]]:
        from google.genai import types

        config = (
            types.EmbedContentConfig(output_dimensionality=self.dimension)
            if self.dimension
            else None
        )
        results: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            response = self.client.models.embed_content(
                model=self.model,
                contents=batch,
                config=config,
            )
            for emb in response.embeddings:
                results.append(list(emb.values))
        return results

    def embed_query(self, query: str) -> list[float]:
        from google.genai import types

        config = (
            types.EmbedContentConfig(output_dimensionality=self.dimension)
            if self.dimension
            else None
        )
        response = self.client.models.embed_content(
            model=self.model,
            contents=query,
            config=config,
        )
        return list(response.embeddings[0].values)


class OpenAIEmbeddingService(BaseEmbeddingService):
    """Generación de embeddings con OpenAI (text-embedding-3-small)."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimension: int = 768,
    ) -> None:
        if not api_key:
            raise ValueError(
                "Falta la clave OPENAI_API_KEY para inicializar OpenAIEmbeddingService."
            )
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.dimension = dimension

    def embed_texts(self, texts: Sequence[str], batch_size: int = 100) -> list[list[float]]:
        results: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            kwargs = {"model": self.model, "input": batch}
            if self.dimension and "text-embedding-3" in self.model:
                kwargs["dimensions"] = self.dimension
            response = self.client.embeddings.create(**kwargs)
            for item in response.data:
                results.append(item.embedding)
        return results

    def embed_query(self, query: str) -> list[float]:
        kwargs = {"model": self.model, "input": query}
        if self.dimension and "text-embedding-3" in self.model:
            kwargs["dimensions"] = self.dimension
        response = self.client.embeddings.create(**kwargs)
        return response.data[0].embedding


def get_embedding_service(settings: Settings | None = None) -> BaseEmbeddingService:
    """Fábrica que construye el proveedor de embeddings según la configuración."""
    cfg = settings or get_settings()
    if cfg.embedding_provider == "gemini":
        return GeminiEmbeddingService(
            api_key=cfg.google_api_key or "",
            model=cfg.embedding_model,
            dimension=cfg.embedding_dim,
        )
    elif cfg.embedding_provider == "openai":
        return OpenAIEmbeddingService(
            api_key=cfg.openai_api_key or "",
            model=cfg.embedding_model,
            dimension=cfg.embedding_dim,
        )
    else:
        raise ValueError(f"Proveedor de embedding desconocido: {cfg.embedding_provider}")
