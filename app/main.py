"""Punto de entrada principal de la aplicación FastAPI."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import get_settings
from app.db.connection import close_connection_pool, init_db_schema

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rag-enterprise")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Ciclo de vida de la aplicación."""
    cfg = get_settings()
    logger.info("Iniciando RAG Enterprise Pipeline en entorno: %s", cfg.environment)

    # Verificar y crear esquema si la BD está accesible
    try:
        init_db_schema()
    except Exception as exc:
        logger.warning(
            "No se pudo inicializar el esquema de BD automáticamente durante el arranque: %s. "
            "(Asegúrate de que PostgreSQL con pgvector esté en ejecución).",
            exc,
        )

    yield

    logger.info("Cerrando recursos y conexiones...")
    close_connection_pool()


app = FastAPI(
    title="RAG Enterprise Pipeline API",
    description=(
        "Pipeline RAG modular de grado de producción: Ingesta -> Parent-Child Chunking -> "
        "Embeddings -> PostgreSQL (pgvector HNSW + tsvector) -> Recuperación Híbrida (RRF) -> "
        "ReRanking Neuronal (Cross-Encoder) -> Generación con Citas y Guardrails de Fuera de Dominio."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registrar rutas
app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
