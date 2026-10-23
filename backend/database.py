import os
import sqlite3
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "documents.db")


def get_connection():
    connection = sqlite3.connect(DATABASE_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                pages INTEGER NOT NULL,
                chunks INTEGER NOT NULL,
                uploaded_at TEXT NOT NULL,
                file_path TEXT NOT NULL
            )
        """)


def save_document(document):
    with get_connection() as connection:
        connection.execute("""
            INSERT INTO documents (
                document_id, filename, pages, chunks,
                uploaded_at, file_path
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            document["document_id"],
            document["filename"],
            document["pages"],
            document["chunks"],
            document["uploaded_at"],
            document["file_path"]
        ))


def get_all_documents():
    with get_connection() as connection:
        rows = connection.execute("""
            SELECT document_id, filename, pages, chunks,
                   uploaded_at, file_path
            FROM documents
            ORDER BY uploaded_at
        """).fetchall()

    return [dict(row) for row in rows]


def get_document(document_id):
    with get_connection() as connection:
        row = connection.execute("""
            SELECT document_id, filename, pages, chunks,
                   uploaded_at, file_path
            FROM documents
            WHERE document_id = ?
        """, (document_id,)).fetchone()

    return dict(row) if row else None


def delete_document(document_id):
    with get_connection() as connection:
        cursor = connection.execute(
            "DELETE FROM documents WHERE document_id = ?",
            (document_id,)
        )
        return cursor.rowcount > 0
