# Taller RAG: Pipeline Empresarial Modular

Implementación limpia, modular y orientada a producción de un sistema de **Generación Aumentada por Recuperación (RAG)** en **Python** y **FastAPI**, utilizando **PostgreSQL** con **pgvector**.

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
│   └── test_cases.py           # Suite automatizada de pruebas
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

### 3. Documentación interactiva
Abre en tu navegador:
👉 **`http://localhost:8000/docs`**

---

## 🛠️ Tecnologías

* **Lenguaje:** Python 3.11
* **API:** FastAPI + Uvicorn
* **Base de Datos:** PostgreSQL 16 + pgvector (HNSW) + tsvector (GIN)
* **Embeddings:** Google `gemini-embedding-001` (o OpenAI `text-embedding-3-small`)
* **LLM:** Google `gemini-3.5-flash-lite` (o OpenAI `gpt-4o-mini`)
* **ReRanker:** FlashRank (Cross-Encoder ligero en CPU)
