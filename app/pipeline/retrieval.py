"""Recuperación Híbrida: Búsqueda Densa (pgvector HNSW) + Búsqueda Léxica (tsvector) + RRF.

Teoría y Buenas Prácticas:
1. Búsqueda Densa (Vectores): Excelente para sinonimia, conceptos generales y lenguaje natural ("no enfría", "hace ruido extraño").
2. Búsqueda Léxica (Sparse / BM25 / tsvector): Esencial para términos exactos, modelos alfanuméricos, códigos de error ("E6", "FTXS25", "220V").
3. Reciprocal Rank Fusion (RRF):
   - Los puntajes de similitud coseno (ej. 0.72) y los puntajes de ts_rank_cd (ej. 0.08) están en escalas numéricas completamente dispares.
   - RRF normaliza por POSICIÓN en el ranking: RRF_Score = SUM(1 / (k + rank)) con k=60.
   - No requiere calibrar pesos arbitrarios y produce resultados robustos y estables.
4. Resolución de Padres: Los candidatos recuperan el Chunk Padre para entregar contexto completo al LLM.
"""

import logging
from uuid import UUID

from app.core.config import Settings, get_settings
from app.db.connection import get_db_connection
from app.pipeline.embeddings import BaseEmbeddingService
from app.schemas.models import RetrievedCandidate

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Ejecuta búsqueda combinada vectorial y léxica sobre PostgreSQL."""

    def __init__(
        self,
        embedding_service: BaseEmbeddingService,
        settings: Settings | None = None,
    ) -> None:
        self.embedding_service = embedding_service
        self.settings = settings or get_settings()

    def retrieve(
        self,
        query: str,
        brand_filter: str | None = None,
        dense_limit: int | None = None,
        sparse_limit: int | None = None,
        rrf_k: int | None = None,
    ) -> list[RetrievedCandidate]:
        """
        Ejecuta la búsqueda híbrida y aplica fusión RRF.
        """
        d_limit = dense_limit or self.settings.dense_top_k
        s_limit = sparse_limit or self.settings.sparse_top_k
        k = rrf_k or self.settings.rrf_k

        # 1. Generar vector para la búsqueda densa
        query_vector = self.embedding_service.embed_query(query)

        # 2. Ejecutar ambas búsquedas en base de datos
        dense_results = self._dense_search(query_vector, limit=d_limit)
        sparse_results = self._sparse_search(query, limit=s_limit)

        # 3. Aplicar Reciprocal Rank Fusion (RRF)
        # Diccionario clave: child_id -> RetrievedCandidate
        merged: dict[UUID, RetrievedCandidate] = {}

        # Asignar RRF para resultados densos
        for rank, item in enumerate(dense_results, start=1):
            cid = item["child_id"]
            rrf_val = 1.0 / (k + rank)
            merged[cid] = RetrievedCandidate(
                child_id=cid,
                parent_id=item["parent_id"],
                document_id=item["document_id"],
                filename=item["filename"],
                page_number=item["page_number"],
                child_content=item["child_content"],
                parent_content=item["parent_content"],
                dense_score=float(item["dense_score"]),
                sparse_score=None,
                rrf_score=rrf_val,
            )

        # Acumular RRF para resultados léxicos
        for rank, item in enumerate(sparse_results, start=1):
            cid = item["child_id"]
            rrf_val = 1.0 / (k + rank)
            if cid in merged:
                merged[cid].rrf_score += rrf_val
                merged[cid].sparse_score = float(item["sparse_score"])
            else:
                merged[cid] = RetrievedCandidate(
                    child_id=cid,
                    parent_id=item["parent_id"],
                    document_id=item["document_id"],
                    filename=item["filename"],
                    page_number=item["page_number"],
                    child_content=item["child_content"],
                    parent_content=item["parent_content"],
                    dense_score=None,
                    sparse_score=float(item["sparse_score"]),
                    rrf_score=rrf_val,
                )

        # 4. Ordenar candidatos por puntaje RRF descendente
        ranked_candidates = sorted(
            merged.values(), key=lambda c: c.rrf_score, reverse=True
        )

        logger.info(
            "Búsqueda híbrida completada: %d densos, %d léxicos -> %d combinados",
            len(dense_results),
            len(sparse_results),
            len(ranked_candidates),
        )
        return ranked_candidates

    def _dense_search(self, query_vector: list[float], limit: int) -> list[dict]:
        """Consulta vectorial densa con distancia coseno sobre el índice HNSW."""
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 
                        c.id AS child_id,
                        c.parent_id,
                        c.document_id,
                        d.filename,
                        c.page_number,
                        c.content AS child_content,
                        p.content AS parent_content,
                        (1 - (c.embedding <=> %(qvec)s::vector)) AS dense_score
                    FROM child_chunks c
                    JOIN documents d ON d.id = c.document_id
                    JOIN parent_chunks p ON p.id = c.parent_id
                    ORDER BY c.embedding <=> %(qvec)s::vector
                    LIMIT %(limit)s;
                    """,
                    {"qvec": query_vector, "limit": limit},
                )
                return cur.fetchall()

    def _sparse_search(self, query_text: str, limit: int) -> list[dict]:
        """Consulta léxica de texto completo sobre el índice GIN de tsvector."""
        # Limpieza básica para evitar caracteres de control de consulta SQL
        clean_text = " ".join(query_text.split()).strip()
        if not clean_text:
            return []

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Usamos websearch_to_tsquery para admitir sintaxis natural amigable (comillas, operadores)
                cur.execute(
                    """
                    SELECT 
                        c.id AS child_id,
                        c.parent_id,
                        c.document_id,
                        d.filename,
                        c.page_number,
                        c.content AS child_content,
                        p.content AS parent_content,
                        ts_rank_cd(c.tsv, websearch_to_tsquery('spanish', %(qtext)s)) AS sparse_score
                    FROM child_chunks c
                    JOIN documents d ON d.id = c.document_id
                    JOIN parent_chunks p ON p.id = c.parent_id
                    WHERE c.tsv @@ websearch_to_tsquery('spanish', %(qtext)s)
                    ORDER BY sparse_score DESC
                    LIMIT %(limit)s;
                    """,
                    {"qtext": clean_text, "limit": limit},
                )
                return cur.fetchall()
