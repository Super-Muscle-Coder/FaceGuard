"""Database helper module for FaceGuard SQLite metadata storage."""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class Homeowner:
    name: str
    full_name: Optional[str] = None
    status: str = "active"
    notes: str = ""


@dataclass
class HomeownerRecord:
    id: str
    name: str
    full_name: str
    status: str
    notes: str


class HomeownerDatabase:
    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            root = Path(__file__).resolve().parents[1]
            db_path = root / "face_recognition.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self):
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS homeowners (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                full_name TEXT,
                status TEXT DEFAULT 'active',
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                id TEXT PRIMARY KEY,
                homeowner_id TEXT NOT NULL,
                embedding_type TEXT DEFAULT 'center',
                vector_blob BLOB NOT NULL,
                dim INTEGER NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(homeowner_id) REFERENCES homeowners(id)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS recognition_logs (
                id TEXT PRIMARY KEY,
                homeowner_id TEXT,
                confidence REAL,
                image_path TEXT,
                camera_id TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(homeowner_id) REFERENCES homeowners(id)
            )
            """
        )
        self.conn.commit()

    def add_homeowner(self, homeowner: Homeowner) -> str:
        homeowner_id = str(uuid.uuid4())
        self.conn.execute(
            """
            INSERT INTO homeowners (id, name, full_name, status, notes)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                homeowner_id,
                homeowner.name,
                homeowner.full_name or homeowner.name,
                homeowner.status,
                homeowner.notes,
            ),
        )
        self.conn.commit()
        return homeowner_id

    def get_homeowner_by_name(self, name: str) -> Optional[HomeownerRecord]:
        row = self.conn.execute(
            "SELECT * FROM homeowners WHERE name = ?",
            (name,),
        ).fetchone()
        if row is None:
            return None
        return HomeownerRecord(
            id=row["id"],
            name=row["name"],
            full_name=row["full_name"],
            status=row["status"],
            notes=row["notes"] or "",
        )

    def get_all_homeowners(self, status: Optional[str] = None) -> List[HomeownerRecord]:
        if status:
            rows = self.conn.execute(
                "SELECT * FROM homeowners WHERE status = ? ORDER BY created_at DESC",
                (status,),
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM homeowners ORDER BY created_at DESC").fetchall()
        return [
            HomeownerRecord(
                id=row["id"],
                name=row["name"],
                full_name=row["full_name"],
                status=row["status"],
                notes=row["notes"] or "",
            )
            for row in rows
        ]

    def save_embedding(self, homeowner_id: str, embedding: np.ndarray, embedding_type: str = "center"):
        embedding = np.asarray(embedding, dtype=np.float32)
        blob = embedding.tobytes()
        self.conn.execute(
            "DELETE FROM embeddings WHERE homeowner_id = ? AND embedding_type = ?",
            (homeowner_id, embedding_type),
        )
        self.conn.execute(
            """
            INSERT INTO embeddings (id, homeowner_id, embedding_type, vector_blob, dim)
            VALUES (?, ?, ?, ?, ?)
            """,
            (str(uuid.uuid4()), homeowner_id, embedding_type, blob, int(embedding.size)),
        )
        self.conn.commit()

    def get_all_embeddings(self, embedding_type: str = "center") -> Dict[str, np.ndarray]:
        rows = self.conn.execute(
            """
            SELECT h.name, e.vector_blob, e.dim
            FROM embeddings e
            JOIN homeowners h ON h.id = e.homeowner_id
            WHERE e.embedding_type = ? AND h.status = 'active'
            """,
            (embedding_type,),
        ).fetchall()

        result: Dict[str, np.ndarray] = {}
        for row in rows:
            vec = np.frombuffer(row["vector_blob"], dtype=np.float32)
            dim = int(row["dim"])
            if vec.size != dim:
                continue
            result[row["name"]] = vec
        return result

    def log_recognition(
        self,
        homeowner_id: Optional[str],
        confidence: Optional[float],
        image_path: Optional[str],
        camera_id: Optional[str],
    ):
        self.conn.execute(
            """
            INSERT INTO recognition_logs (id, homeowner_id, confidence, image_path, camera_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (str(uuid.uuid4()), homeowner_id, confidence, image_path, camera_id),
        )
        self.conn.commit()

    def close(self):
        if self.conn:
            self.conn.close()


__all__ = ["Homeowner", "HomeownerRecord", "HomeownerDatabase"]