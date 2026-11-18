-- ============================================================================
-- RAG ENTERPRISE PIPELINE: Inicialización de Base de Datos
-- PostgreSQL 16 + pgvector + tsvector (Full-Text Search) + HNSW Index
-- ============================================================================

-- 1. Extensiones necesarias
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "unaccent";

-- 2. Tabla de Documentos Maestros
-- Almacena metadatos del archivo origen (.pdf o .md)
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename VARCHAR(255) NOT NULL UNIQUE,
    file_type VARCHAR(50) NOT NULL, -- 'pdf' o 'markdown'
    total_pages INT DEFAULT 1,
    checksum VARCHAR(64), -- SHA-256 para evitar duplicados
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 3. Tabla de Chunks Padres (Parent Chunks)
-- Representa bloques de contexto extensos (~800 a 1200 tokens)
-- que se le entregarán al LLM para no perder coherencia.
CREATE TABLE IF NOT EXISTS parent_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    page_number INT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_document_parent_chunk UNIQUE (document_id, chunk_index)
);

-- 4. Tabla de Chunks Hijos (Child Chunks)
-- Representa subfragmentos pequeños (~150 a 250 tokens) optimizados
-- para alta resolución semántica en la búsqueda vectorial y de palabras clave.
CREATE TABLE IF NOT EXISTS child_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    parent_id UUID NOT NULL REFERENCES parent_chunks(id) ON DELETE CASCADE,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    page_number INT,
    -- Dimensión configurable (por defecto 768 para gemini-embedding-001 o text-embedding-004)
    -- Si usas OpenAI text-embedding-3-small usa vector(1536)
    embedding vector(768),
    -- Campo calculado de búsqueda léxica (Full-Text Search en español)
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('spanish', coalesce(content, ''))) STORED,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 5. Índices de Alto Rendimiento

-- A) Índice HNSW (Hierarchical Navigable Small World) para Búsqueda Vectorial Densa
-- ¿Por qué HNSW y no IVFFlat?
-- 1. HNSW no necesita una fase previa de entrenamiento ("clustering") con datos ya insertados.
-- 2. Permite inserciones continuas sin degradar la precisión del índice.
-- 3. Ofrece una tasa de acierto (Recall) superior a >98% con latencias de pocos milisegundos.
-- Parámetros:
--   m = 16 (conexiones bidireccionales por nodo)
--   ef_construction = 64 (tamaño de la lista dinámica durante la indexación)
CREATE INDEX IF NOT EXISTS idx_child_embedding_hnsw 
ON child_chunks 
USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);

-- B) Índice GIN para Búsqueda Léxica Dispersa (Sparse BM25 / Full-Text Search)
-- Permite búsquedas instantáneas por palabras clave exactas, códigos de error o números de parte
CREATE INDEX IF NOT EXISTS idx_child_tsv_gin 
ON child_chunks 
USING gin (tsv);

-- C) Índices Relacionales B-Tree para consultas y borrados en cascada rápidos
CREATE INDEX IF NOT EXISTS idx_parent_document_id ON parent_chunks (document_id);
CREATE INDEX IF NOT EXISTS idx_child_parent_id ON child_chunks (parent_id);
CREATE INDEX IF NOT EXISTS idx_child_document_id ON child_chunks (document_id);
