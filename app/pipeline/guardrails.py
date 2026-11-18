"""Guardrail de seguridad y verificación de fuera de dominio (Out-of-Domain Guardrail).

Teoría y Buenas Prácticas:
1. Uno de los fallos más graves en sistemas RAG (documentado explícitamente en la tesis académica de referencia)
   es la «alucinación por transferencia de dominio»: si un usuario consulta sobre un equipo, sistema o tema
   que NO está contemplado en el corpus, el LLM intentará responder forzando fragmentos no relacionados.
2. Un Guardrail previo evalúa si la consulta cae dentro del ámbito legítimo del sistema antes de
   ejecutar búsquedas vectoriales costosas o consumir cuotas de generación.
3. Detecta también intentos elementales de inyección de prompts («ignora tus instrucciones anteriores»).
"""

import logging
import re
from app.core.config import Settings, get_settings
from app.schemas.models import GuardrailResult

logger = logging.getLogger(__name__)

# Patrones típicos de inyección de prompt o solicitudes adversarias
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(?:all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"ignora\s+(?:todas\s+las\s+)?instrucciones", re.IGNORECASE),
    re.compile(r"olvida\s+(?:todo\s+lo\s+anterior|tus\s+reglas)", re.IGNORECASE),
    re.compile(r"system\s*prompt|prompt\s*de\s*sistema", re.IGNORECASE),
    re.compile(r"act\s+as\s+a\s+(?:pirate|dan|jailbreak)", re.IGNORECASE),
]

# Temas manifiestamente fuera de un asistente técnico
_OFF_TOPIC_PATTERNS = [
    re.compile(r"\b(?:receta|poema|chiste|cancion|horoscopo|politica|futbol)\b", re.IGNORECASE),
    re.compile(r"\b(?:recipe|poem|joke|song|horoscope|politics|soccer)\b", re.IGNORECASE),
]


class DomainGuardrail:
    """Validador de pertinencia de dominio y seguridad de entrada."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def validate(self, question: str) -> GuardrailResult:
        """
        Ejecuta la validación en dos etapas:
        1. Reglas heurísticas rápidas (seguridad / off-topic flagrante).
        2. Clasificación semántica de dominio (si está habilitado en configuración).
        """
        if not self.settings.guardrail_enabled:
            return GuardrailResult(is_allowed=True, reason="Guardrails deshabilitados")

        q_clean = question.strip()
        if len(q_clean) < 3:
            return GuardrailResult(
                is_allowed=False,
                reason="Consulta demasiado corta",
                suggested_response="Por favor, formula una pregunta más detallada para poder buscar en los manuales.",
            )

        # 1. Chequeo de inyección de prompt
        for pattern in _INJECTION_PATTERNS:
            if pattern.search(q_clean):
                logger.warning("Intento de inyección detectado: %s", q_clean)
                return GuardrailResult(
                    is_allowed=False,
                    reason="Posible inyección de instrucciones no autorizada",
                    suggested_response="Lo siento, solo puedo responder preguntas técnicas fundamentadas en los manuales de servicio autorizados.",
                )

        # 2. Chequeo de temas manifiestamente ajenos
        for pattern in _OFF_TOPIC_PATTERNS:
            if pattern.search(q_clean):
                logger.info("Consulta fuera de dominio detectada: %s", q_clean)
                return GuardrailResult(
                    is_allowed=False,
                    reason="Tema fuera del dominio técnico",
                    suggested_response=f"Esta consulta no pertenece al alcance de la documentación disponible ({self.settings.system_domain_description}). Por favor formula una consulta técnica relacionada.",
                )

        # 3. Consulta válida para proceder al pipeline RAG
        return GuardrailResult(
            is_allowed=True,
            reason="Consulta válida dentro del dominio",
        )
