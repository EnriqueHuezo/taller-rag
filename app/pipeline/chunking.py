"""Chunking Jerárquico (Parent-Document Chunking).

Teoría y Buenas Prácticas:
1. El embedding de textos muy largos diluye la información semántica específica (ej. un valor de voltaje o código de error).
2. El embedding de fragmentos diminutos (100 tokens) tiene alta precisión de búsqueda, pero al entregárselo al LLM,
   éste carece del contexto necesario para formular una respuesta coherente y estructurada.
3. Solución (Parent-Child):
   - Dividimos en Chunks Hijos (Child Chunks) pequeños (~800 caracteres) para el embedding y la búsqueda vectorial/léxica.
   - Cada Hijo apunta a su Chunk Padre (Parent Chunk) más amplio (~3,500 caracteres).
   - En la recuperación, encontramos los mejores Hijos, pero le entregamos al LLM sus Chunks Padres consolidados.
"""

import uuid
from typing import Generator
from app.schemas.models import ChildChunk, ParentChunk, RawDocument


class ParentChildChunker:
    """Implementa la segmentación jerárquica en dos niveles: Padre e Hijo."""

    def __init__(
        self,
        parent_chunk_size: int = 3500,
        parent_chunk_overlap: int = 400,
        child_chunk_size: int = 800,
        child_chunk_overlap: int = 150,
    ) -> None:
        self.parent_chunk_size = parent_chunk_size
        self.parent_chunk_overlap = parent_chunk_overlap
        self.child_chunk_size = child_chunk_size
        self.child_chunk_overlap = child_chunk_overlap

    @staticmethod
    def _split_text_with_overlap(
        text: str, chunk_size: int, chunk_overlap: int
    ) -> Generator[str, None, None]:
        """
        Divide texto intentando cortar en límites de párrafo (\n\n) o punto final (.),
        evitando partir palabras a la mitad.
        """
        if not text:
            return

        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + chunk_size
            if end >= text_len:
                chunk = text[start:].strip()
                if chunk:
                    yield chunk
                break

            # Intentar encontrar un límite natural antes del final del chunk
            split_at = -1
            # 1. Párrafos
            p_split = text.rfind("\n\n", start, end)
            if p_split != -1 and p_split > start + (chunk_size // 2):
                split_at = p_split + 2
            else:
                # 2. Puntos seguidos de espacio o salto de línea
                dot_split = text.rfind(". ", start, end)
                if dot_split != -1 and dot_split > start + (chunk_size // 2):
                    split_at = dot_split + 2
                else:
                    # 3. Espacio en blanco
                    space_split = text.rfind(" ", start, end)
                    if space_split != -1 and space_split > start + (chunk_size // 2):
                        split_at = space_split + 1

            if split_at == -1:
                split_at = end

            chunk = text[start:split_at].strip()
            if chunk:
                yield chunk

            start = split_at - chunk_overlap
            if start < 0 or start >= split_at:
                start = split_at

    def process(
        self, raw_doc: RawDocument
    ) -> tuple[list[ParentChunk], list[ChildChunk]]:
        """
        Genera la jerarquía completa de Parent Chunks y Child Chunks vinculados.
        """
        parent_chunks: list[ParentChunk] = []
        child_chunks: list[ChildChunk] = []

        parent_idx = 0
        child_idx = 0

        for page_num, page_text in raw_doc.pages:
            # 1. Generar Chunks Padres para la página
            for p_text in self._split_text_with_overlap(
                page_text, self.parent_chunk_size, self.parent_chunk_overlap
            ):
                parent_id = uuid.uuid4()
                parent = ParentChunk(
                    id=parent_id,
                    chunk_index=parent_idx,
                    content=p_text,
                    page_number=page_num,
                    metadata={
                        "filename": raw_doc.filename,
                        "file_type": raw_doc.file_type,
                    },
                )
                parent_chunks.append(parent)
                parent_idx += 1

                # 2. Subdividir este Chunk Padre en Chunks Hijos
                for c_text in self._split_text_with_overlap(
                    p_text, self.child_chunk_size, self.child_chunk_overlap
                ):
                    child = ChildChunk(
                        id=uuid.uuid4(),
                        parent_id=parent_id,
                        chunk_index=child_idx,
                        content=c_text,
                        page_number=page_num,
                        metadata={
                            "filename": raw_doc.filename,
                            "parent_index": parent.chunk_index,
                        },
                    )
                    child_chunks.append(child)
                    child_idx += 1

        return parent_chunks, child_chunks
