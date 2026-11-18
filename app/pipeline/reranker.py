"""Re-Ranking Neuronal (Cross-Encoder / Two-Stage Retrieval).

Teoría y Buenas Prácticas:
1. Bi-Encoders (Embeddings):
   - Generan vectores aislados E(consulta) y E(fragmento). Son ultra-rápidos (HNSW en BD), pero pierden
     la atención cruzada ("cross-attention") entre los términos específicos de la consulta y el texto.
2. Cross-Encoders (ReRankers):
   - Evalúan la consulta y el fragmento juntos en una sola red neuronal: Score = M(consulta, fragmento).
   - Su precisión es incomparablemente superior a la distancia coseno vectorial, pero son computacionalmente
     demasiado costosos para ejecutarse sobre millones de documentos.
3. Arquitectura en Dos Fases (Two-Stage Pattern):
   - Fase 1 (Recuperación Híbrida): Filtra de 10,000+ chunks a los mejores 15-20 candidatos.
   - Fase 2 (Re-Ranking Neuronal): El Cross-Encoder reordena con precisión quirúrgica esos 15-20 candidatos
     y entrega al LLM únicamente los 3 a 5 más relevantes.
"""

import logging
from app.core.config import Settings, get_settings
from app.schemas.models import RetrievedCandidate

logger = logging.getLogger(__name__)


class NeuralReranker:
    """Aplica re-ranking basado en Cross-Encoder sobre candidatos recuperados."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._ranker = None
        self._init_ranker()

    def _init_ranker(self) -> None:
        try:
            from flashrank import Ranker
            # flashrank utiliza modelos ONNX compactos optimizados para CPU (~10-20ms)
            self._ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
            logger.info("Motor de re-ranking neuronal FlashRank inicializado.")
        except Exception as exc:
            logger.warning(
                "FlashRank no disponible (%s). Se utilizará re-ranking basado en RRF.",
                exc,
            )
            self._ranker = None

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedCandidate],
        top_k: int | None = None,
    ) -> list[RetrievedCandidate]:
        """
        Reordena los candidatos utilizando el modelo Cross-Encoder.
        Si el modelo neuronal no está disponible, conserva el ordenamiento por RRF.
        """
        k = top_k or self.settings.rerank_top_k
        if not candidates:
            return []

        if self._ranker is None:
            # Fallback limpio al orden de RRF
            return candidates[:k]

        from flashrank import RerankRequest

        # Preparamos los pasajes para FlashRank. Evaluamos el texto del hijo
        # (o del padre) contra la pregunta.
        passages = [
            {"id": str(c.child_id), "text": c.child_content}
            for c in candidates
        ]

        rerank_req = RerankRequest(query=query, passages=passages)
        ranked_results = self._ranker.rerank(rerank_req)

        # Mapeamos los puntajes neuronales de vuelta a los candidatos
        score_by_id = {str(item["id"]): float(item["score"]) for item in ranked_results}

        for candidate in candidates:
            candidate.rerank_score = score_by_id.get(str(candidate.child_id), 0.0)

        # Ordenar por el puntaje del Cross-Encoder
        reordered = sorted(
            candidates,
            key=lambda c: c.rerank_score or 0.0,
            reverse=True,
        )

        logger.info(
            "Re-ranking completado: %d candidatos ordenados -> top %d seleccionados",
            len(candidates),
            min(k, len(reordered)),
        )
        return reordered[:k]
