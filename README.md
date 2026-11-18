# Taller RAG: Pipeline Empresarial de Alto Rendimiento 🚀

> **Repositorio de referencia técnica y educativa para el Taller de RAG.**
> Implementación completa de un pipeline de grado de producción: **Ingesta $\rightarrow$ Parent-Child Chunking $\rightarrow$ Embeddings $\rightarrow$ PostgreSQL (pgvector HNSW + tsvector) $\rightarrow$ Recuperación Híbrida (RRF) $\rightarrow$ Re-Ranking Neuronal (FlashRank) $\rightarrow$ Generación con Citas y Guardrails**.

---

## 🎯 1. ¿Qué contiene este repositorio?

Este proyecto implementa una arquitectura modular y desacoplada construida en **Python** y **FastAPI**, lista para ejecutarse localmente o en **Docker**:

1. **Ingesta Especializada (`.pdf` y `.md`):** Extracción de texto página a página con PyMuPDF (conservando números de página) y Markdown con checksum SHA-256.
2. **Chunking Jerárquico (Parent-Child):**
   * **Chunks Hijos (~800 chars):** Alta resolución semántica para la búsqueda vectorial y de texto sin dilución.
   * **Chunks Padres (~3,500 chars):** Bloques de contexto amplio que se le entregan al LLM para que no pierda coherencia técnica.
3. **Persistencia Dual en PostgreSQL 16:**
   * **Búsqueda Densa:** Vectores con índice navegable **HNSW** (`pgvector`) sin necesidad de reentrenamiento.
   * **Búsqueda Léxica:** Índice invertido **GIN** sobre columna calculada en español (`tsvector`).
4. **Recuperación Híbrida con RRF:** Fusión matemática mediante *Reciprocal Rank Fusion* ($k=60$) que combina las dos ramas sin sesgo numérico.
5. **Re-Ranking Neuronal (Two-Stage Retrieval):** Cross-Encoder optimizado en CPU con **FlashRank** para seleccionar los 3 a 5 fragmentos más relevantes en ~15 ms.
6. **Domain Guardrails:** Filtro previo para evitar alucinaciones por transferencia de dominio e intentos de inyección de prompts.
7. **Generación con Citas Verificadas:** El LLM (Gemini / OpenAI) fundamenta cada afirmación con referencias `[1]`, `[2]` vinculadas al archivo, página y extracto textual.
8. **Suite Automatizada de Pruebas:** 6 escenarios de prueba con 100% de aprobación para evaluar latencia, exactitud y guardrails.

---

## 🏗️ 2. Arquitectura del Flujo

```
========================================================================================
                                PIPELINE DE INGESTA
========================================================================================
[Archivos .pdf / .md]
         │
         ▼
[1. Loaders Especializados]  (PyMuPDF para PDF por página / MarkdownLoader con checksum)
         │
         ▼
[2. Parent-Child Chunking]   Padre: ~3,500 chars (Contexto amplio para síntesis)
                             Hijo:  ~800 chars   (Alta resolución para búsqueda)
         │
         ▼
[3. Embeddings por Lote]     (Gemini gemini-embedding-001 o OpenAI text-embedding-3-small)
         │
         ▼
[4. PostgreSQL 16]           • documents (Metadatos maestros)
                             • parent_chunks (Bloques de contexto completo)
                             • child_chunks (Vectores + índice HNSW + tsvector GIN)

========================================================================================
                                PIPELINE DE CONSULTA
========================================================================================
[Pregunta del Usuario]
         │
         ▼
[1. Domain Guardrail]        ¿Está dentro del dominio técnico? ¿Es inyección de prompt?
         │                    └──> Si es NO: Rechazo seguro e inmediato (0 costo de LLM)
         ▼ (Si es SÍ)
[2. Búsqueda Híbrida]
   ├── Rama Densa:  pgvector HNSW (Distancia Coseno sobre Chunks Hijos)
   └── Rama Léxica: tsvector GIN (Full-Text Search BM25 en español)
         │
         ▼
[3. Reciprocal Rank Fusion]  Combina ambas listas por posición: RRF = SUM(1 / (60 + rank))
         │
         ▼
[4. Resolución de Padres]    Mapea los mejores Chunks Hijos hacia sus Chunks Padres
         │
         ▼
[5. Re-Ranking Neuronal]     Cross-Encoder (FlashRank) reordena los candidatos con precisión
         │
         ▼
[6. Generación con Citas]    LLM sintetiza la respuesta obligando referencias formales [1], [2]
         │
         ▼
[Respuesta + Citas Estructuradas] (Texto fundamentado + Documento, Página y Snippet exacto)
```

---

## 📂 3. Estructura del Proyecto (Clean Architecture)

```text
taller-rag/
├── docker-compose.yml          # PostgreSQL 16 con pgvector + App FastAPI
├── Dockerfile                  # Contenedor optimizado Python 3.11
├── init_db.sql                 # Esquema DDL: Tablas, extensiones, HNSW y GIN
├── requirements.txt            # Dependencias versionadas
├── .env.example                # Plantilla de variables de entorno (sin secretos)
├── .gitignore                  # Protección estricta contra subida de credenciales
├── README.md                   # Esta documentación técnica
├── app/
│   ├── main.py                 # Punto de entrada FastAPI y ciclo de vida (lifespan)
│   ├── core/
│   │   ├── config.py           # Pydantic Settings inmutable
│   │   └── __init__.py
│   ├── schemas/
│   │   ├── models.py           # Contratos Pydantic (Parent, Child, Query, Citation)
│   │   └── __init__.py
│   ├── db/
│   │   ├── connection.py       # ConnectionPool con adaptador nativo pgvector
│   │   └── __init__.py
│   ├── pipeline/
│   │   ├── loaders.py          # Extracción especializada de PDF y Markdown
│   │   ├── chunking.py         # Parent-Document Chunking (Jerárquico)
│   │   ├── embeddings.py       # Servicios de embedding (Gemini y OpenAI)
│   │   ├── storage.py          # Persistencia relacional y vectorial atómica
│   │   ├── guardrails.py       # Domain & Prompt-Injection Guardrail
│   │   ├── retrieval.py        # Búsqueda híbrida (HNSW + GIN tsvector + RRF)
│   │   ├── reranker.py         # Re-ranking neuronal con FlashRank (Cross-Encoder)
│   │   ├── generator.py        # Síntesis con citas estrictas [1], [2]
│   │   ├── engine.py           # Orquestador RAGPipeline
│   │   └── __init__.py
│   └── api/
│       ├── routes.py           # Endpoints REST (/api/chat, /api/documents)
│       └── __init__.py
├── scripts/
│   ├── ingest.py               # CLI para indexar archivos o directorios completos
│   ├── query.py                # CLI interactivo con desglose de citas y guardrails
│   └── test_cases.py           # Suite automatizada de pruebas de evaluación
└── data/
    └── sample_docs/
        └── manual_split_ejemplo.md # Documento técnico de muestra listo para probar
```

---

## 🚀 4. Inicio Rápido con Docker

1. **Configurar tu clave de API:**
   Copia `.env.example` a `.env` y coloca tu clave de Google AI Studio:
   ```dotenv
   GOOGLE_API_KEY=tu_clave_de_google_ai_studio
   ```

2. **Levantar contenedores:**
   ```bash
   docker compose up --build -d
   ```

3. **Acceder a la documentación interactiva:**
   Abre en tu navegador: `http://localhost:8000/docs`

---

## 🧪 5. Pruebas y Validación

```bash
# 1. Indexar el documento técnico de prueba:
docker exec rag_api python scripts/ingest.py data/sample_docs/manual_split_ejemplo.md

# 2. Hacer una consulta técnica con citas formales:
docker exec rag_api python scripts/query.py "¿Qué significa el código E6 y qué debo medir en el motor?"

# 3. Ejecutar la suite completa de 6 casos de prueba:
docker exec rag_api python scripts/test_cases.py
```

---

## 🔒 6. Seguridad y Buenas Prácticas
* El archivo `.env` que contiene las claves privadas de API **está explícitamente ignorado por `.gitignore`** y nunca debe subirse al repositorio.
* Solo se versiona `.env.example` con variables de plantilla y valores seguros por defecto.
