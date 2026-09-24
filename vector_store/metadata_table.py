# ==============================================================================
# vector_store/metadata_table.py
# ------------------------------------------------------------------------------
# STEP 5 of the RAG pipeline: "Store Embeddings in Delta Table" -- implemented
# locally with SQLite instead of a Databricks Delta Table.
#
# WHAT THIS FILE DOES
#   Maintains a local SQLite database with two tables:
#     1. chunk_metadata  -- one row per chunk, mirroring the exact schema
#        requested in the spec (chunk_id, document_name, document_type,
#        page_number, chunk_text, embedding_vector, created_timestamp).
#     2. ingested_files   -- one row per source PDF, tracking a content hash
#        so re-running ingestion is idempotent (a file that hasn't changed
#        is skipped; a changed or new file is (re-)processed).
#
# WHY SQLITE STANDS IN FOR A DELTA TABLE HERE
#   A Databricks Delta Table is a versioned, ACID-compliant table format
#   built on Parquet, designed for large-scale, concurrent, distributed
#   reads/writes across a cluster. For a single-machine local sandbox with a
#   few thousand chunks, none of that distributed-systems machinery is
#   needed -- SQLite gives us the same *relational, queryable, structured*
#   properties (a real schema, SQL queries, ACID single-writer transactions)
#   with zero setup: it's a single file on disk, built into Python's
#   standard library.
#
# WHY THIS TABLE EXISTS *IN ADDITION TO* CHROMADB
#   ChromaDB (vector_store/chroma_manager.py) is the system of record for
#   *similarity search* -- finding the nearest vectors fast. This table is
#   the system of record for *structured inspection and auditing*: "show me
#   every chunk that came from Summary_of_Benefits.pdf page 3", "how many
#   chunks total do we have per document_type", "has this file already been
#   ingested". Chroma is not built for that kind of relational querying;
#   splitting the two concerns is exactly the same reasoning that leads a
#   Databricks deployment to keep the Delta table (system of record, SQL
#   queryable) separate from the Vector Search index (fast ANN search)
#   synced FROM it.
#
# WHY embedding_vector IS STORED AS A JSON STRING
#   SQLite has no native array/vector column type. We serialize the vector
#   to a JSON string purely for inspection/audit purposes (e.g. "does this
#   chunk actually have a non-empty embedding recorded"). Real similarity
#   search always goes through ChromaDB, never through this table --
#   comparing vectors with raw SQL would be slow and is not what this table
#   is for.
#
# INPUT / OUTPUT
#   create_tables()                        -> None (creates schema if absent)
#   insert_chunks(chunks, embeddings)      -> None
#   get_ingested_file_hash(file_name)      -> Optional[str]
#   upsert_ingested_file(file_name, hash, chunk_count) -> None
#   query_by_document(document_name)       -> List[dict]
#   count_chunks()                         -> int
# ==============================================================================

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional

from chunking.chunker import Chunk
from config.settings import settings
from utils.logging_utils import get_logger

logger = get_logger(__name__)

_CREATE_CHUNK_METADATA_SQL = """
CREATE TABLE IF NOT EXISTS chunk_metadata (
    chunk_id           TEXT PRIMARY KEY,
    document_name      TEXT NOT NULL,
    document_type      TEXT NOT NULL,
    page_number        INTEGER NOT NULL,
    chunk_text         TEXT NOT NULL,
    embedding_vector   TEXT NOT NULL,   -- JSON-encoded list[float]; audit copy only
    created_timestamp  TEXT NOT NULL,
    chapter_title      TEXT,            -- NULL for non-chaptered documents
    section_title      TEXT             -- NULL for non-chaptered documents
);
"""

_CREATE_INGESTED_FILES_SQL = """
CREATE TABLE IF NOT EXISTS ingested_files (
    file_name    TEXT PRIMARY KEY,
    file_hash    TEXT NOT NULL,
    ingested_at  TEXT NOT NULL,
    chunk_count  INTEGER NOT NULL
);
"""

_INDEX_SQL = "CREATE INDEX IF NOT EXISTS idx_chunk_document_name ON chunk_metadata(document_name);"


@contextmanager
def _connection() -> Iterator[sqlite3.Connection]:
    """
    Open a SQLite connection with row access by column name, and always
    close it -- a context manager guarantees the connection (and any
    uncommitted transaction) is cleaned up even if a query raises.
    """
    os.makedirs(os.path.dirname(settings.metadata_db_path), exist_ok=True)
    conn = sqlite3.connect(settings.metadata_db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def create_tables() -> None:
    """Create the chunk_metadata and ingested_files tables if they don't exist yet."""
    with _connection() as conn:
        conn.execute(_CREATE_CHUNK_METADATA_SQL)
        conn.execute(_CREATE_INGESTED_FILES_SQL)
        conn.execute(_INDEX_SQL)
    logger.info("Metadata tables ready at '%s'", settings.metadata_db_path)


def insert_chunks(chunks: List[Chunk], embeddings: List[List[float]]) -> None:
    """
    Insert (or overwrite, on chunk_id conflict) rows into chunk_metadata.

    `INSERT OR REPLACE` gives us the same "upsert" idempotency here that
    ChromaDB's `.upsert()` gives us for the vector index -- re-inserting a
    chunk_id that already exists replaces the old row instead of erroring.
    """
    if not chunks:
        return
    rows = [
        (
            chunk.chunk_id,
            chunk.file_name,
            chunk.document_type,
            chunk.page_number,
            chunk.chunk_text,
            json.dumps(embedding),
            chunk.created_timestamp,
            chunk.chapter_title,
            chunk.section_title,
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]
    with _connection() as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO chunk_metadata
                (chunk_id, document_name, document_type, page_number,
                 chunk_text, embedding_vector, created_timestamp,
                 chapter_title, section_title)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    logger.info("Inserted %d rows into chunk_metadata", len(rows))


def delete_by_document(document_name: str) -> None:
    """Remove all chunk_metadata rows for one source document (before re-ingesting it)."""
    with _connection() as conn:
        conn.execute("DELETE FROM chunk_metadata WHERE document_name = ?", (document_name,))
    logger.info("Deleted existing chunk_metadata rows for document '%s'", document_name)


def query_by_document(document_name: str) -> List[Dict[str, Any]]:
    """Return every chunk row for one source document, ordered by page number."""
    with _connection() as conn:
        cursor = conn.execute(
            "SELECT * FROM chunk_metadata WHERE document_name = ? ORDER BY page_number",
            (document_name,),
        )
        return [dict(row) for row in cursor.fetchall()]


def count_chunks() -> int:
    """Return the total number of chunks currently recorded."""
    with _connection() as conn:
        return conn.execute("SELECT COUNT(*) FROM chunk_metadata").fetchone()[0]


def get_ingested_file_hash(file_name: str) -> Optional[str]:
    """
    Return the previously recorded content hash for a file, or None if it
    has never been ingested. Used by scripts/ingest.py to decide whether a
    PDF needs (re-)processing.
    """
    with _connection() as conn:
        row = conn.execute(
            "SELECT file_hash FROM ingested_files WHERE file_name = ?", (file_name,)
        ).fetchone()
        return row["file_hash"] if row else None


def upsert_ingested_file(file_name: str, file_hash: str, chunk_count: int) -> None:
    """Record (or update) that a file has been ingested, with its content hash."""
    with _connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO ingested_files (file_name, file_hash, ingested_at, chunk_count)
            VALUES (?, ?, ?, ?)
            """,
            (file_name, file_hash, datetime.now(timezone.utc).isoformat(), chunk_count),
        )
    logger.info("Recorded ingestion of '%s' (%d chunks)", file_name, chunk_count)
