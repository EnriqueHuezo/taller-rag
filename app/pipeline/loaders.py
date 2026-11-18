"""Cargadores de documentos especializados para PDF y Markdown."""

import hashlib
import logging
from abc import ABC, abstractmethod
from pathlib import Path

from app.schemas.models import RawDocument

logger = logging.getLogger(__name__)


class BaseDocumentLoader(ABC):
    """Interfaz base para cargadores de documentos."""

    @abstractmethod
    def load(self, file_path: Path) -> RawDocument:
        """Extrae el contenido y estructura del archivo."""
        pass


class PDFLoader(BaseDocumentLoader):
    """
    Carga y extrae texto de archivos PDF página a página usando PyMuPDF.
    Conserva la numeración de páginas original para trazabilidad y citas exactas.
    """

    def load(self, file_path: Path) -> RawDocument:
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:
            raise RuntimeError(
                "PyMuPDF no está instalado. Ejecute 'pip install pymupdf'."
            ) from exc

        pages: list[tuple[int, str]] = []
        with fitz.open(str(file_path)) as doc:
            total_pages = doc.page_count
            for i in range(total_pages):
                page = doc.load_page(i)
                text = (page.get_text("text") or "").strip()
                if text:
                    pages.append((i + 1, text))
                else:
                    # Registra página vacía o escaneada sin texto para mantener secuencia
                    pages.append((i + 1, "[Página sin texto seleccionable]"))

        checksum = self._calculate_checksum(file_path)
        logger.info("PDF cargado: %s (%d páginas extraídas)", file_path.name, len(pages))
        return RawDocument(
            filename=file_path.name,
            file_type="pdf",
            pages=pages,
            metadata={"total_pages": len(pages), "checksum": checksum},
        )

    @staticmethod
    def _calculate_checksum(file_path: Path) -> str:
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()


class MarkdownLoader(BaseDocumentLoader):
    """
    Carga archivos Markdown (.md).
    Permite separar por encabezados de nivel principal o cargarlo como documento único.
    """

    def load(self, file_path: Path) -> RawDocument:
        content = file_path.read_text(encoding="utf-8")
        
        # Para Markdown, tratamos las secciones o el archivo como página 1 (o bloques por encabezados)
        pages: list[tuple[int, str]] = [(1, content.strip())]
        
        checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()
        logger.info("Markdown cargado: %s (%d caracteres)", file_path.name, len(content))
        return RawDocument(
            filename=file_path.name,
            file_type="markdown",
            pages=pages,
            metadata={"total_pages": 1, "checksum": checksum},
        )


def load_document(file_path: Path) -> RawDocument:
    """Fábrica que selecciona el cargador apropiado según la extensión del archivo."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return PDFLoader().load(file_path)
    elif suffix in {".md", ".markdown"}:
        return MarkdownLoader().load(file_path)
    else:
        raise ValueError(
            f"Formato no soportado '{suffix}'. Solo se admiten archivos .pdf y .md."
        )
