"""Configuración global centralizada mediante Pydantic Settings."""

from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Configuración inmutable de la aplicación.
    Lee automáticamente las variables de entorno o el archivo .env.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Entorno
    environment: str = "development"
    debug: bool = True

    # PostgreSQL & pgvector
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "rag_db"
    postgres_user: str = "postgres"
    postgres_password: str = "postgrespassword"

    # Proveedores
    embedding_provider: Literal["gemini", "openai"] = "gemini"
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 768

    llm_provider: Literal["gemini", "openai"] = "gemini"
    llm_model: str = "gemini-3.5-flash-lite"
    llm_temperature: float = 0.1

    # Credenciales de API
    google_api_key: str | None = None
    openai_api_key: str | None = None

    # Chunking Jerárquico (Parent-Child)
    parent_chunk_size: int = 3500
    parent_chunk_overlap: int = 400
    child_chunk_size: int = 800
    child_chunk_overlap: int = 150

    # Recuperación Híbrida y Reranking
    dense_top_k: int = 15
    sparse_top_k: int = 15
    rrf_k: int = 60
    rerank_top_k: int = 5

    # Guardrails
    guardrail_enabled: bool = True
    system_domain_description: str = (
        "Documentación técnica, manuales de servicio, especificaciones y procedimientos de mantenimiento industrial o HVAC"
    )

    @property
    def postgres_dsn(self) -> str:
        """Data Source Name para conexión a PostgreSQL."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}@"
            f"{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retorna una instancia singleton de Settings."""
    return Settings()
