"""Script CLI interactivo para consultar el pipeline RAG y verificar las citas.

Uso:
    python scripts/query.py "¿Qué significa el código de falla E6 y qué debo medir?"
    python scripts/query.py  # Modo interactivo
"""

import argparse
import sys
from pathlib import Path

# Agregar raíz del proyecto al sys.path
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from app.core.config import get_settings
from app.pipeline.engine import RAGPipeline
from app.schemas.models import QueryRequest


def run_single_query(pipeline: RAGPipeline, question: str) -> None:
    print(f"\n" + "=" * 80)
    print(f"PREGUNTA: {question}")
    print("=" * 80)

    req = QueryRequest(question=question)
    res = pipeline.answer_query(req)

    print(f"\n[1. ESTADO DE GUARDRAIL]")
    print(f"  Permitido: {res.guardrail.is_allowed}")
    print(f"  Motivo:    {res.guardrail.reason}")

    if not res.guardrail.is_allowed:
        print(f"\n[RESPUESTA BLOQUEADA POR GUARDRAIL]:")
        print(f"  {res.answer}\n")
        return

    print(f"\n[2. RESULTADOS DE RECUPERACIÓN]")
    print(f"  Fuentes seleccionadas por Reranker: {res.retrieved_count}")
    print(f"  Latencia total: {res.latency_ms} ms")

    print(f"\n[3. RESPUESTA GENERADA]")
    print(res.answer)

    print(f"\n[4. CITAS DOCUMENTALES FORMALES]")
    if not res.citations:
        print("  (No se citaron fuentes específicas)")
    else:
        for cite in res.citations:
            p_info = f" (Pág. {cite.page_number})" if cite.page_number else ""
            print(f"  [{cite.index}] {cite.filename}{p_info}")
            print(f"      Evidencia: \"{cite.quote_snippet}\"")
    print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Consulta interactiva en RAG Enterprise Pipeline.")
    parser.add_argument("query", nargs="?", type=str, help="Pregunta a consultar.")
    args = parser.parse_args()

    pipeline = RAGPipeline(get_settings())

    if args.query:
        run_single_query(pipeline, args.query)
    else:
        print("\n=== MODO INTERACTIVO RAG ENTERPRISE PIPELINE ===")
        print("Escribe tu pregunta (o 'salir' para terminar):")
        while True:
            try:
                q = input("\n> ")
                if q.strip().lower() in {"salir", "exit", "quit"}:
                    break
                if not q.strip():
                    continue
                run_single_query(pipeline, q)
            except (KeyboardInterrupt, EOFError):
                break
        print("\n¡Hasta luego!")


if __name__ == "__main__":
    main()
