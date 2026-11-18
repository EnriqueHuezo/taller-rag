"""Script CLI para ingesta rápida de documentos PDF o Markdown.

Uso:
    python scripts/ingest.py ruta/a/archivo.pdf
    python scripts/ingest.py data/sample_docs/
"""

import argparse
import sys
from pathlib import Path

# Agregar raíz del proyecto al sys.path
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from app.core.config import get_settings
from app.db.connection import init_db_schema
from app.pipeline.engine import RAGPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingesta de documentos en RAG Enterprise Pipeline.")
    parser.add_argument("path", type=str, help="Ruta a un archivo .pdf, .md o carpeta con documentos.")
    args = parser.parse_args()

    target_path = Path(args.path)
    if not target_path.exists():
        print(f"Error: La ruta '{args.path}' no existe.")
        sys.exit(1)

    print("Verificando base de datos...")
    init_db_schema()

    pipeline = RAGPipeline(get_settings())

    files_to_process = []
    if target_path.is_file():
        files_to_process.append(target_path)
    elif target_path.is_dir():
        for ext in ("*.pdf", "*.md", "*.markdown"):
            files_to_process.extend(target_path.glob(ext))

    if not files_to_process:
        print(f"No se encontraron archivos .pdf o .md en '{args.path}'.")
        sys.exit(0)

    print(f"\nProcesando {len(files_to_process)} archivo(s)...")
    for file_path in files_to_process:
        print(f"\n-> Procesando: {file_path.name}")
        try:
            res = pipeline.ingest_file(file_path)
            print(f"   [OK] {res.message}")
            print(f"   Chunks Padres: {res.parent_chunks_count} | Chunks Hijos (con embeddings): {res.child_chunks_count}")
        except Exception as exc:
            print(f"   [ERROR] Falló la ingesta de {file_path.name}: {exc}")


if __name__ == "__main__":
    main()
