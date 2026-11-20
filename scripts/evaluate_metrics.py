"""Script de Evaluación Científica de RAG: Golden Set y Métricas de Recuperación.

Calcula métricas estándar de Information Retrieval (IR):
1. Hit Rate @ K (¿Apareció la evidencia en los top K?)
2. MRR (Mean Reciprocal Rank: qué tan arriba en el ranking apareció la respuesta)
3. Comparativa en vivo: Búsqueda Densa vs. Léxica vs. Híbrida (RRF) vs. ReRanker

Uso:
    python scripts/evaluate_metrics.py
"""

import sys
from pathlib import Path
from dataclasses import dataclass

# Agregar raíz del proyecto al path
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from app.core.config import get_settings
from app.pipeline.engine import RAGPipeline


@dataclass
class GoldenTestCase:
    question: str
    target_evidence_keyword: str  # Frase o término único que DEBE estar en el chunk correcto
    expected_answer: str


# ==============================================================================
# GOLDEN SET (Conjunto Dorado de Evaluación)
# Preguntas curadas con su evidencia esperada en el manual de climatización
# ==============================================================================
GOLDEN_SET: list[GoldenTestCase] = [
    GoldenTestCase(
        question="¿Qué significa el código E6 en el equipo Split?",
        target_evidence_keyword="Bloqueo o Falla del Motor Ventilador Exterior",
        expected_answer="Falla en la señal Hall del motor ventilador exterior o rotor trabado.",
    ),
    GoldenTestCase(
        question="¿Cuáles son los valores de resistencia que deben medir los devanados del motor?",
        target_evidence_keyword="45Ω y 55Ω",
        expected_answer="Entre 45 y 55 ohmios balanceados entre U-V, V-W y W-U.",
    ),
    GoldenTestCase(
        question="¿Qué voltaje DC alimenta al conector del motor del ventilador?",
        target_evidence_keyword="310V DC",
        expected_answer="310V DC entre pin 1 (Rojo) y pin 3 (Negro).",
    ),
    GoldenTestCase(
        question="¿Cuál es el valor nominal de resistencia del termistor de tubería a 25°C?",
        target_evidence_keyword="10.0 kΩ",
        expected_answer="10k ohmios a 25 grados centígrados (NTC).",
    ),
    GoldenTestCase(
        question="¿A cuántos micrones de mercurio se debe realizar el proceso de vacío?",
        target_evidence_keyword="500 micrones",
        expected_answer="Por debajo de 500 micrones de mercurio.",
    ),
]


def calculate_metrics(ranks: list[int | None]) -> tuple[float, float]:
    """
    Calcula Hit Rate y MRR a partir de una lista de posiciones (ranks).
    ranks contiene [1, 2, None, 1, 3] donde None significa que no apareció en el top K.
    """
    total = len(ranks)
    if total == 0:
        return 0.0, 0.0

    hits = sum(1 for r in ranks if r is not None)
    hit_rate = hits / total

    # MRR = Promedio de (1 / rank)
    reciprocal_ranks = [(1.0 / r) if (r is not None and r > 0) else 0.0 for r in ranks]
    mrr = sum(reciprocal_ranks) / total

    return hit_rate, mrr


def evaluate_pipeline() -> None:
    print("\n" + "=" * 85)
    print("      EVALUACIÓN CIENTÍFICA DE RECUPERACIÓN (GOLDEN SET & IR METRICS)")
    print("=" * 85)
    print(f"Total de consultas en el Golden Set: {len(GOLDEN_SET)}")
    print("Métricas evaluadas: Hit Rate@5 y MRR (Mean Reciprocal Rank)")
    print("=" * 85 + "\n")

    cfg = get_settings()
    pipeline = RAGPipeline(cfg)

    ranks_dense: list[int | None] = []
    ranks_sparse: list[int | None] = []
    ranks_hybrid_rrf: list[int | None] = []
    ranks_reranked: list[int | None] = []

    for idx, item in enumerate(GOLDEN_SET, start=1):
        print(f"Q{idx}: \"{item.question}\"")
        target = item.target_evidence_keyword.lower()

        # 1. Ejecutar embedding de la query para densa
        q_vec = pipeline.embedding_service.embed_query(item.question)

        # 2. Obtener listas crudas de cada método
        dense_hits = pipeline.retriever._dense_search(q_vec, limit=5)
        sparse_hits = pipeline.retriever._sparse_search(item.question, limit=5)

        # 3. Híbrida RRF
        hybrid_candidates = pipeline.retriever.retrieve(item.question, dense_limit=10, sparse_limit=10)

        # 4. ReRanker Cross-Encoder
        reranked_candidates = pipeline.reranker.rerank(item.question, hybrid_candidates, top_k=5)

        # Buscar la posición del fragmento correcto en cada rama
        def find_rank(items: list, content_key: str = "child_content") -> int | None:
            for rank, chunk in enumerate(items, start=1):
                text = getattr(chunk, content_key, None) if not isinstance(chunk, dict) else chunk.get(content_key, "")
                if target in (text or "").lower():
                    return rank
            return None

        r_dense = find_rank(dense_hits)
        r_sparse = find_rank(sparse_hits)
        r_hybrid = find_rank(hybrid_candidates[:5])
        r_rerank = find_rank(reranked_candidates[:5])

        ranks_dense.append(r_dense)
        ranks_sparse.append(r_sparse)
        ranks_hybrid_rrf.append(r_hybrid)
        ranks_reranked.append(r_rerank)

        print(f"   Posición de la evidencia -> Densa: {r_dense or 'Miss'} | Léxica: {r_sparse or 'Miss'} | Híbrida: {r_hybrid or 'Miss'} | ReRanker: {r_rerank or 'Miss'}")

    # Cálculos globales
    hr_dense, mrr_dense = calculate_metrics(ranks_dense)
    hr_sparse, mrr_sparse = calculate_metrics(ranks_sparse)
    hr_hybrid, mrr_hybrid = calculate_metrics(ranks_hybrid_rrf)
    hr_rerank, mrr_rerank = calculate_metrics(ranks_reranked)

    print("\n" + "=" * 85)
    print("                    TABLA COMPARATIVA DE RENDIMIENTO")
    print("=" * 85)
    print(f"{'Método de Recuperación':<30} | {'Hit Rate @ 5':<15} | {'MRR (Mean Reciprocal Rank)':<25}")
    print("-" * 85)
    print(f"{'1. Solo Densa (pgvector)':<30} | {hr_dense*100:>12.1f}% | {mrr_dense:>20.4f}")
    print(f"{'2. Solo Léxica (tsvector BM25)':<30} | {hr_sparse*100:>12.1f}% | {mrr_sparse:>20.4f}")
    print(f"{'3. Híbrida (RRF)':<30} | {hr_hybrid*100:>12.1f}% | {mrr_hybrid:>20.4f}")
    print(f"{'4. Híbrida + ReRanker':<30} | {hr_rerank*100:>12.1f}% | {mrr_rerank:>20.4f}")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    evaluate_pipeline()
