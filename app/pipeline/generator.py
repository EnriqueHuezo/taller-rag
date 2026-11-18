"""Generación Aumentada con Citas Estructuradas y Grounding Estricto."""

import logging
import re
from uuid import UUID

from app.core.config import Settings, get_settings
from app.schemas.models import Citation, RetrievedCandidate

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Eres un asistente técnico especializado de alto nivel. Tu función es responder a la consulta del usuario ÚNICAMENTE con base en la evidencia factual proporcionada en el bloque de contexto.

REGLAS DE OBLIGATORIO CUMPLIMIENTO:
1. Responde con claridad, precisión técnica y paso a paso.
2. CITACIÓN EXACTA: Cada afirmación, valor técnico (voltajes, presiones, resistencias) o paso de procedimiento DEBE llevar su referencia entre corchetes vinculada a la fuente, por ejemplo: [1], [2].
3. SI LA INFORMACIÓN NO ESTÁ EN EL CONTEXTO: Declara explícitamente: "La información disponible en los manuales indexados no es suficiente para responder a esta consulta." No intentes adivinar ni extrapolar procedimientos de otros equipos.
4. NUNCA inventes códigos de error, números de parte ni valores numéricos.
5. Responde en el mismo idioma en que se formula la pregunta (español por defecto)."""


class RAGGenerator:
    """Orquesta la síntesis del LLM y la extracción de citas verificadas."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def generate_response(
        self,
        question: str,
        candidates: list[RetrievedCandidate],
    ) -> tuple[str, list[Citation]]:
        """
        Construye el contexto jerárquico consolidado, invoca al LLM y mapea las citas.
        """
        if not candidates:
            return (
                "No se encontraron fragmentos relevantes en los manuales para responder a tu consulta.",
                [],
            )

        # 1. Consolidar Chunks Padres (Deduplicación de contexto)
        # Si dos chunks hijos pertenecen al mismo chunk padre, usamos el padre una sola vez
        unique_parents: dict[UUID, RetrievedCandidate] = {}
        for c in candidates:
            if c.parent_id not in unique_parents:
                unique_parents[c.parent_id] = c

        ordered_sources = list(unique_parents.values())

        # 2. Construir el bloque de contexto para el prompt
        context_parts: list[str] = []
        source_mapping: dict[int, RetrievedCandidate] = {}

        for idx, src in enumerate(ordered_sources, start=1):
            source_mapping[idx] = src
            p_num = f" | Página: {src.page_number}" if src.page_number else ""
            header = f"--- [Fuente {idx}] Documento: {src.filename}{p_num} ---"
            context_parts.append(f"{header}\n{src.parent_content}\n")

        full_context = "\n".join(context_parts)

        # 3. Llamar al proveedor de LLM configurado
        prompt = (
            f"CONTEXTO DOCUMENTAL DISPONIBLE:\n"
            f"================================\n"
            f"{full_context}\n"
            f"================================\n\n"
            f"PREGUNTA DEL USUARIO: {question}\n\n"
            f"RESPUESTA FUNDAMENTADA (Recuerda incluir citas [1], [2], etc.):"
        )

        answer = self._call_llm(prompt)

        # 4. Extraer citas estructuradas y vincularlas a los documentos y páginas
        citations = self._extract_citations(answer, source_mapping)

        return answer, citations

    def _call_llm(self, prompt: str) -> str:
        """Invoca al modelo generativo correspondiente."""
        provider = self.settings.llm_provider

        if provider == "gemini":
            from google import genai
            from google.genai import types

            if not self.settings.google_api_key:
                raise ValueError("Falta GOOGLE_API_KEY para invocar Gemini.")

            client = genai.Client(api_key=self.settings.google_api_key)
            model_id = self.settings.llm_model.removeprefix("models/")

            response = client.models.generate_content(
                model=model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=self.settings.llm_temperature,
                ),
            )
            return (response.text or "").strip()

        elif provider == "openai":
            from openai import OpenAI

            if not self.settings.openai_api_key:
                raise ValueError("Falta OPENAI_API_KEY para invocar OpenAI.")

            client = OpenAI(api_key=self.settings.openai_api_key)
            response = client.chat.completions.create(
                model=self.settings.llm_model,
                temperature=self.settings.llm_temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            return (response.choices[0].message.content or "").strip()

        else:
            raise ValueError(f"Proveedor de LLM no soportado: {provider}")

    @staticmethod
    def _extract_citations(
        answer: str, source_mapping: dict[int, RetrievedCandidate]
    ) -> list[Citation]:
        """
        Encuentra todas las referencias [1], [2] en la respuesta y construye
        objetos Citation con el documento, página y snippet que lo sustenta.
        """
        # Expresión regular flexible para capturar [1], [2], o [Fuente 1], [Fuente 2], etc.
        cited_indices = set(
            int(m)
            for m in re.findall(r"\[(?:Fuente\s+|Source\s+)?(\d+)\]", answer, re.IGNORECASE)
        )
        citations: list[Citation] = []

        for idx in sorted(cited_indices):
            if idx in source_mapping:
                cand = source_mapping[idx]
                # Tomamos un snippet representativo del contenido padre
                snippet = (cand.parent_content[:280] + "...") if len(cand.parent_content) > 280 else cand.parent_content
                citations.append(
                    Citation(
                        index=idx,
                        filename=cand.filename,
                        page_number=cand.page_number,
                        quote_snippet=snippet.replace("\n", " "),
                    )
                )

        return citations
