"""Esquemas de datos Pydantic para el contrato de la API y el pipeline interno."""

from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field


class DocumentRecord(BaseModel):
    """Representación de un documento maestro indexado."""
    id: UUID
    filename: str
    file_type: str
    total_pages: int = 1
    checksum: str | None = None
    metadata: dict = Field(default_factory=dict)
    created_at: datetime | None = None


class RawDocument(BaseModel):
    """Documento extraído en memoria antes del chunking."""
    filename: str
    file_type: str  # 'pdf' | 'markdown'
    pages: list[tuple[int, str]]  # Lista de (page_number, text)
    metadata: dict = Field(default_factory=dict)


class ParentChunk(BaseModel):
    """Chunk de contexto amplio para síntesis por el LLM."""
    id: UUID | None = None
    document_id: UUID | None = None
    chunk_index: int
    content: str
    page_number: int | None = None
    metadata: dict = Field(default_factory=dict)


class ChildChunk(BaseModel):
    """Sub-chunk de alta resolución para búsqueda vectorial y léxica."""
    id: UUID | None = None
    parent_id: UUID | None = None
    document_id: UUID | None = None
    chunk_index: int
    content: str
    page_number: int | None = None
    embedding: list[float] | None = None
    metadata: dict = Field(default_factory=dict)


class GuardrailResult(BaseModel):
    """Resultado de la verificación de dominio y seguridad."""
    is_allowed: bool
    reason: str
    suggested_response: str | None = None


class RetrievedCandidate(BaseModel):
    """Candidato recuperado tras la búsqueda híbrida y resolución de padre."""
    child_id: UUID
    parent_id: UUID
    document_id: UUID
    filename: str
    page_number: int | None = None
    child_content: str
    parent_content: str
    dense_score: float | None = None
    sparse_score: float | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None


class Citation(BaseModel):
    """Cita formal mapeada a la fuente y página exacta."""
    index: int = Field(description="Número de referencia en el texto [1], [2], etc.")
    filename: str = Field(description="Nombre del documento de procedencia")
    page_number: int | None = Field(default=None, description="Número de página de la evidencia")
    quote_snippet: str = Field(description="Fragmento textual que sustenta la respuesta")


class QueryRequest(BaseModel):
    """Solicitud de consulta RAG."""
    question: str = Field(min_length=2, description="Pregunta del usuario")
    brand_filter: str | None = Field(default=None, description="Filtro opcional por marca o etiqueta")
    top_k: int = Field(default=5, ge=1, le=20, description="Cantidad de fuentes finales para el LLM")


class QueryResponse(BaseModel):
    """Respuesta generada con respaldo documental verificado."""
    question: str
    answer: str
    citations: list[Citation]
    guardrail: GuardrailResult
    retrieved_count: int
    latency_ms: float


class IngestResponse(BaseModel):
    """Resultado del proceso de ingesta de un documento."""
    filename: str
    file_type: str
    total_pages: int
    parent_chunks_count: int
    child_chunks_count: int
    status: str
    message: str
