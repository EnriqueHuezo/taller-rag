# Taller RAG: Pipeline Empresarial Modular 🚀

Implementación limpia, modular y orientada a producción de un sistema de **Generación Aumentada por Recuperación (RAG)** en **Python** y **FastAPI**, utilizando **PostgreSQL** con **pgvector**.

---

## 🌟 Características Principales

* **📄 Ingesta de Documentos:** Compatible exclusivamente con archivos `.pdf` (con extracción de páginas) y `.md` (Markdown).
* **🧩 Parent-Child Chunking:** Divide en fragmentos pequeños (Hijos, ~800 chars) para búsqueda precisa, y recupera bloques amplios (Padres, ~3,500 chars) para que el LLM no pierda contexto.
* **🗄️ PostgreSQL + pgvector + tsvector:**
  * **Búsqueda Densa:** Embeddings con índice **HNSW** de alta velocidad.
  * **Búsqueda Léxica:** Índice invertido **GIN** sobre texto completo en español.
* **⚡ Fusión Híbrida (RRF):** Combina ambas búsquedas con *Reciprocal Rank Fusion* ($k=60$) sin calibrar pesos arbitrarios.
* **🎯 Re-Ranking Neuronal:** Cross-Encoder con **FlashRank** para seleccionar los 3 a 5 fragmentos más relevantes en CPU (~15 ms).
* **🛡️ Guardrails:** Detección de consultas fuera de dominio e intentos de inyección de prompts antes de llamar al LLM.
* **📌 Generación con Citas:** El modelo genera respuestas con referencias formales `[1]`, `[2]` vinculadas a su archivo, página y extracto textual.

---

## 📂 Estructura del Proyecto

```text
taller-rag/
├── docker-compose.yml          # PostgreSQL con pgvector + API FastAPI
├── Dockerfile                  # Imagen optimizada en Python 3.11
├── init_db.sql                 # Esquema de base de datos e índices HNSW/GIN
├── requirements.txt            # Dependencias del proyecto
├── .env.example                # Plantilla de variables de entorno (sin secretos)
├── .gitignore                  # Exclusión de credenciales y archivos temporales
├── app/
│   ├── main.py                 # Punto de entrada y ciclo de vida de FastAPI
│   ├── core/config.py          # Configuración con Pydantic Settings
│   ├── schemas/models.py       # Modelos de datos (Parent, Child, Query, Citations)
│   ├── db/connection.py        # Pool de conexiones a PostgreSQL
│   ├── pipeline/               # Módulos del flujo RAG
│   │   ├── loaders.py          # Carga de PDF y Markdown
│   │   ├── chunking.py         # Parent-Child Chunking
│   │   ├── embeddings.py       # Embeddings (Gemini / OpenAI)
│   │   ├── storage.py          # Guardado en base de datos
│   │   ├── guardrails.py       # Validador de dominio y seguridad
│   │   ├── retrieval.py        # Búsqueda Híbrida (HNSW + tsvector + RRF)
│   │   ├── reranker.py         # Cross-Encoder ReRanker
│   │   ├── generator.py        # Síntesis con citas formales
│   │   └── engine.py           # Orquestador del pipeline
│   └── api/routes.py           # Endpoints REST (/api/chat, /api/documents)
├── scripts/
│   ├── ingest.py               # CLI para indexar archivos o carpetas
│   ├── query.py                # CLI para realizar consultas con citas
│   └── test_cases.py           # Suite automatizada de 6 casos de prueba
└── data/sample_docs/
    └── manual_split_ejemplo.md # Documento de muestra listo para probar
```

---

## 🚀 Inicio Rápido (Docker)

### 1. Configurar variables de entorno
Copia la plantilla y coloca tu clave de API de Google AI Studio:
```bash
cp .env.example .env
```
Edita `.env`:
```dotenv
GOOGLE_API_KEY=tu_clave_aqui
```

### 2. Levantar los servicios
```bash
docker compose up --build -d
```

### 3. Abrir la documentación interactiva
Visita en tu navegador:
👉 **`http://localhost:8000/docs`**

---

## 🧪 Pruebas por Consola (CLI)

```bash
# 1. Indexar el documento técnico de muestra:
docker exec rag_api python scripts/ingest.py data/sample_docs/manual_split_ejemplo.md

# 2. Hacer una consulta técnica con citas formales:
docker exec rag_api python scripts/query.py "¿Qué significa el código E6 y qué debo medir en el motor?"

# 3. Probar el Guardrail (fuera de dominio):
docker exec rag_api python scripts/query.py "¿Cómo se prepara una pizza?"

# 4. Ejecutar la suite completa de pruebas:
docker exec rag_api python scripts/test_cases.py
```

---

## 🛠️ Tecnologías

* **Lenguaje:** Python 3.11
* **API:** FastAPI + Uvicorn
* **Base de Datos:** PostgreSQL 16 + pgvector (HNSW) + tsvector (GIN)
* **Embeddings:** Google `gemini-embedding-001` (o OpenAI `text-embedding-3-small`)
* **LLM:** Google `gemini-3.5-flash-lite` (rápido y eficiente)
* **ReRanker:** FlashRank (Cross-Encoder ligero en CPU)
