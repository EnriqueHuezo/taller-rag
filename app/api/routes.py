"""Rutas de la API REST para ingesta y consultas RAG."""

import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.config import get_settings
from app.pipeline.engine import RAGPipeline
from app.schemas.models import (
    DocumentRecord,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)

router = APIRouter()
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    """Singleton lazy-loaded del pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(get_settings())
    return _pipeline


@router.get("/health", tags=["Sistema"])
def health_check() -> dict:
    """Verifica el estado del servicio y configuración."""
    cfg = get_settings()
    return {
        "status": "healthy",
        "environment": cfg.environment,
        "embedding_provider": cfg.embedding_provider,
        "llm_provider": cfg.llm_provider,
        "postgres_host": cfg.postgres_host,
    }


@router.post("/api/chat", response_model=QueryResponse, tags=["RAG"])
def chat_query(request: QueryRequest) -> QueryResponse:
    """
    Ejecuta el flujo completo RAG:
    Guardrails -> Búsqueda Híbrida (HNSW + tsvector + RRF) -> ReRanking Neuronal -> Respuesta con Citas.
    """
    try:
        pipeline = get_pipeline()
        return pipeline.answer_query(request)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error durante el procesamiento RAG: {exc}",
        ) from exc


@router.post("/api/documents/upload", response_model=IngestResponse, tags=["Documentos"])
async def upload_document(file: UploadFile = File(...)) -> IngestResponse:
    """
    Sube un archivo .pdf o .md y ejecuta el pipeline de ingesta jerárquica con embeddings.
    """
    filename = file.filename or "uploaded_file"
    suffix = Path(filename).suffix.lower()

    if suffix not in {".pdf", ".md", ".markdown"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Formato no soportado. Únicamente se permiten archivos PDF o Markdown (.md).",
        )

    # Guardar en archivo temporal para que los loaders puedan leerlo
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        # Renombramos temporalmente para preservar el nombre de archivo real en metadatos
        target_path = tmp_path.parent / filename
        if target_path.exists():
            target_path.unlink()
        tmp_path.rename(target_path)

        pipeline = get_pipeline()
        return pipeline.ingest_file(target_path)
    finally:
        if target_path.exists():
            target_path.unlink()


@router.get("/api/documents", response_model=list[DocumentRecord], tags=["Documentos"])
def list_documents() -> list[DocumentRecord]:
    """Lista todos los documentos indexados en la base de datos."""
    pipeline = get_pipeline()
    return pipeline.storage.list_documents()


@router.delete("/api/documents/{filename}", tags=["Documentos"])
def delete_document(filename: str) -> dict:
    """Elimina un documento y todos sus fragmentos asociados."""
    pipeline = get_pipeline()
    success = pipeline.storage.delete_document(filename)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No se encontró el documento '{filename}'.",
        )
    return {"message": f"Documento '{filename}' eliminado con éxito."}
