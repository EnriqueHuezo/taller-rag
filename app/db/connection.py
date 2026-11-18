"""Gestión de conexiones a PostgreSQL con soporte nativo de pgvector."""

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

_pool: ConnectionPool | None = None


def get_connection_pool(settings: Settings | None = None) -> ConnectionPool:
    """
    Retorna o inicializa el pool de conexiones singleton.
    Configura automáticamente el adaptador nativo de pgvector en cada conexión.
    """
    global _pool
    if _pool is None:
        cfg = settings or get_settings()
        
        def configure_conn(conn: psycopg.Connection) -> None:
            # Registra los tipos de pgvector (vector) para serialización automática a Python list[float]
            register_vector(conn)

        _pool = ConnectionPool(
            conninfo=cfg.postgres_dsn,
            min_size=2,
            max_size=10,
            timeout=30.0,
            configure=configure_conn,
            kwargs={"row_factory": dict_row},
        )
        logger.info("Pool de conexiones a PostgreSQL inicializado exitosamente.")
    return _pool


def close_connection_pool() -> None:
    """Cierra el pool de conexiones de manera ordenada."""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None
        logger.info("Pool de conexiones cerrado.")


@contextmanager
def get_db_connection(settings: Settings | None = None) -> Generator[psycopg.Connection, None, None]:
    """
    Context manager para obtener una conexión del pool y devolverla al finalizar.
    
    Ejemplo:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
    """
    pool = get_connection_pool(settings)
    with pool.connection() as conn:
        yield conn


def init_db_schema(sql_file_path: Path | None = None) -> None:
    """
    Ejecuta el script SQL de inicialización (tablas, extensiones e índices HNSW/GIN)
    en caso de que no hayan sido creados previamente.
    """
    if sql_file_path is None:
        # Busca init_db.sql en la raíz del proyecto
        sql_file_path = Path(__file__).resolve().parents[2] / "init_db.sql"

    if not sql_file_path.exists():
        logger.warning("No se encontró el archivo init_db.sql en %s", sql_file_path)
        return

    sql_script = sql_file_path.read_text(encoding="utf-8")
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql_script)
        conn.commit()
    logger.info("Esquema de base de datos verificado/creado con éxito.")
