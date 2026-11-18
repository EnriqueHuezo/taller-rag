"""Motor orquestador del Pipeline RAG Empresarial."""

import logging
from pathlib import Path
from time import perf_counter

from app.core.config import Settings, get_settings
from app.pipeline.chunking import ParentChildChunker
from app.pipeline.embeddings import get_embedding_service
from app.pipeline.generator import RAGGenerator
from app.pipeline.guardrails import DomainGuardrail
from app.pipeline.loaders import load_document
from app.pipeline.reranker import NeuralReranker
from app.pipeline.retrieval import HybridRetriever
from app.pipeline.storage import DocumentStorageService
from app.schemas.models import (
    IngestResponse,
    QueryRequest,
    QueryResponse,
)

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Orquestador principal que integra todos los pasos de la arquitectura:
    Ingesta:
      Carga -> Chunking Jerárquico -> Embeddings -> Persistencia PostgreSQL
    Consulta:
      Guardrail -> Búsqueda Híbrida (HNSW + tsvector + RRF) -> ReRanking Neuronal -> Síntesis con Citas
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.storage = DocumentStorageService()
        self.chunker = ParentChildChunker(
            parent_chunk_size=self.settings.parent_chunk_size,
            parent_chunk_overlap=self.settings.parent_chunk_overlap,
            child_chunk_size=self.settings.child_chunk_size,
            child_chunk_overlap=self.settings.child_chunk_overlap,
        )
        self.embedding_service = get_embedding_service(self.settings)
        self.guardrail = DomainGuardrail(self.settings)
        self.retriever = HybridRetriever(self.embedding_service, self.settings)
        self.reranker = NeuralReranker(self.settings)
        self.generator = RAGGenerator(self.settings)

    def ingest_file(self, file_path: Path) -> IngestResponse:
        """
        Ejecuta el pipeline completo de ingesta para un archivo PDF o Markdown.
        """
        started = perf_counter()
        logger.info("Iniciando ingesta del archivo: %s", file_path.name)

        # 1. Cargar archivo (.pdf o .md)
        raw_doc = load_document(file_path)

        # 2. Chunking Jerárquico (Padres e Hijos)
        parent_chunks, child_chunks = self.chunker.process(raw_doc)

        if not child_chunks:
            raise ValueError(f"No se pudo extraer contenido útil del archivo {file_path.name}")

        # 3. Generar Embeddings para los Chunks Hijos
        child_texts = [c.content for c in child_chunks]
        embeddings = self.embedding_service.embed_texts(child_texts)

        for chunk, emb in zip(child_chunks, embeddings):
            chunk.embedding = emb

        # 4. Almacenar atómicamente en PostgreSQL con pgvector y tsvector
        doc_record = self.storage.save_document_hierarchy(
            raw_doc=raw_doc,
            parent_chunks=parent_chunks,
            child_chunks=child_chunks,
        )

        elapsed = round(perf_counter() - started, 2)
        logger.info(
            "Ingesta completada en %s s: %s (%d padres, %d hijos)",
            elapsed,
            file_path.name,
            len(parent_chunks),
            len(child_chunks),
        )

        return IngestResponse(
            filename=doc_record.filename,
            file_type=doc_record.file_type,
            total_pages=doc_record.total_pages,
            parent_chunks_count=len(parent_chunks),
            child_chunks_count=len(child_chunks),
            status="success",
            message=f"Documento indexado con éxito en {elapsed}s.",
        )

    def answer_query(self, request: QueryRequest) -> QueryResponse:
        """
        Ejecuta el pipeline completo de consulta y generación con citas.
        """
        started = perf_counter()
        logger.info("Procesando consulta: '%s'", request.question)

        # 1. Guardrail de fuera de dominio y seguridad
        guardrail_result = self.guardrail.validate(request.question)
        if not guardrail_result.is_allowed:
            elapsed_ms = round((perf_counter() - started) * 1000, 2)
            return QueryResponse(
                question=request.question,
                answer=guardrail_result.suggested_response or "Consulta rechazada por políticas de dominio.",
                citations=[],
                guardrail=guardrail_result,
                retrieved_count=0,
                latency_ms=elapsed_ms,
            )

        # 2. Recuperación Híbrida (Dense HNSW + Sparse tsvector + RRF)
        candidates = self.retriever.retrieve(
            query=request.question,
            brand_filter=request.brand_filter,
            dense_limit=self.settings.dense_top_k,
            sparse_limit=self.settings.sparse_top_k,
            rrf_k=self.settings.rrf_k,
        )

        # 3. Re-Ranking Neuronal (Cross-Encoder)
        top_candidates = self.reranker.rerank(
            query=request.question,
            candidates=candidates,
            top_k=request.top_k,
        )

        # 4. Generación Aumentada con Citas y Grounding Estricto
        answer, citations = self.generator.generate_response(
            question=request.question,
            candidates=top_candidates,
        )

        elapsed_ms = round((perf_counter() - started) * 1000, 2)
        logger.info(
            "Consulta respondida con éxito en %s ms (%d citas generadas)",
            elapsed_ms,
            len(citations),
        )

        return QueryResponse(
            question=request.question,
            answer=answer,
            citations=citations,
            guardrail=guardrail_result,
            retrieved_count=len(top_candidates),
            latency_ms=elapsed_ms,
        )
