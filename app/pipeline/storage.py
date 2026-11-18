"""Capa de persistencia e indexación en PostgreSQL + pgvector + tsvector."""

import json
import logging
from uuid import UUID

from app.db.connection import get_db_connection
from app.schemas.models import ChildChunk, DocumentRecord, ParentChunk, RawDocument

logger = logging.getLogger(__name__)


class DocumentStorageService:
    """Gestiona el almacenamiento relacional, vectorial y de texto completo."""

    def save_document_hierarchy(
        self,
        raw_doc: RawDocument,
        parent_chunks: list[ParentChunk],
        child_chunks: list[ChildChunk],
    ) -> DocumentRecord:
        """
        Inserta atómicamente el documento, sus chunks padres y sus chunks hijos con vectores.
        """
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # 1. Insertar Documento Maestro
                cur.execute(
                    """
                    INSERT INTO documents (filename, file_type, total_pages, checksum, metadata)
                    VALUES (%(filename)s, %(file_type)s, %(total_pages)s, %(checksum)s, %(metadata)s)
                    ON CONFLICT (filename) DO UPDATE 
                    SET total_pages = EXCLUDED.total_pages,
                        checksum = EXCLUDED.checksum,
                        metadata = EXCLUDED.metadata
                    RETURNING id, filename, file_type, total_pages, checksum, metadata, created_at;
                    """,
                    {
                        "filename": raw_doc.filename,
                        "file_type": raw_doc.file_type,
                        "total_pages": len(raw_doc.pages),
                        "checksum": raw_doc.metadata.get("checksum"),
                        "metadata": json.dumps(raw_doc.metadata),
                    },
                )
                doc_row = cur.fetchone()
                doc_id: UUID = doc_row["id"]

                # 2. Si ya existía, limpiar chunks anteriores (cascada en BD)
                cur.execute("DELETE FROM parent_chunks WHERE document_id = %(doc_id)s;", {"doc_id": doc_id})

                # 3. Insertar Chunks Padres
                parent_params = [
                    (
                        str(p.id),
                        str(doc_id),
                        p.chunk_index,
                        p.content,
                        p.page_number,
                        json.dumps(p.metadata),
                    )
                    for p in parent_chunks
                ]
                cur.executemany(
                    """
                    INSERT INTO parent_chunks (id, document_id, chunk_index, content, page_number, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s);
                    """,
                    parent_params,
                )

                # 4. Insertar Chunks Hijos con sus Vectores de Embedding
                child_params = [
                    (
                        str(c.id),
                        str(c.parent_id),
                        str(doc_id),
                        c.chunk_index,
                        c.content,
                        c.page_number,
                        c.embedding,  # El adaptador pgvector se encarga del casteo a vector
                        json.dumps(c.metadata),
                    )
                    for c in child_chunks
                ]
                cur.executemany(
                    """
                    INSERT INTO child_chunks (id, parent_id, document_id, chunk_index, content, page_number, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                    """,
                    child_params,
                )

            conn.commit()

        logger.info(
            "Documento '%s' guardado con éxito (%d padres, %d hijos)",
            raw_doc.filename,
            len(parent_chunks),
            len(child_chunks),
        )
        return DocumentRecord(**doc_row)

    def delete_document(self, filename: str) -> bool:
        """Elimina un documento y todos sus chunks en cascada."""
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM documents WHERE filename = %(filename)s RETURNING id;",
                    {"filename": filename},
                )
                deleted = cur.fetchone()
            conn.commit()
        return deleted is not None

    def list_documents(self) -> list[DocumentRecord]:
        """Retorna todos los documentos registrados."""
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, filename, file_type, total_pages, checksum, metadata, created_at FROM documents ORDER BY created_at DESC;"
                )
                rows = cur.fetchall()
        return [DocumentRecord(**row) for row in rows]
