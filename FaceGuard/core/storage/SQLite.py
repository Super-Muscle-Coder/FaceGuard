"""SQLite storage manager for FaceGuard metadata."""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from typing import Dict, List, Optional

from config.settings import DATABASE_SQLITE_PATH


class SQLiteManager:
    def __init__(self, db_path=DATABASE_SQLITE_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _ensure_schema(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS persons (
                    person_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    full_name TEXT,
                    status TEXT DEFAULT 'active',
                    vector_storage_key TEXT,
                    vector_count INTEGER DEFAULT 0,
                    metadata_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS access_logs (
                    log_id TEXT PRIMARY KEY,
                    person_id TEXT,
                    camera_id TEXT,
                    result TEXT NOT NULL,
                    confidence REAL,
                    image_path TEXT,
                    metadata_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(person_id) REFERENCES persons(person_id)
                )
                """
            )

    @staticmethod
    def _loads_metadata(value: Optional[str]) -> Dict:
        if not value:
            return {}
        try:
            return json.loads(value)
        except Exception:
            return {}

    # ==================== PERSONS ====================

    def get_person_by_name(self, name: str) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM persons WHERE name = ?",
                (name,),
            ).fetchone()
            if row is None:
                return None
            result = dict(row)
            result["metadata"] = self._loads_metadata(result.get("metadata_json"))
            return result

    def get_person_by_id(self, person_id: str) -> Optional[Dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM persons WHERE person_id = ?",
                (person_id,),
            ).fetchone()
            if row is None:
                return None
            result = dict(row)
            result["metadata"] = self._loads_metadata(result.get("metadata_json"))
            return result

    def list_persons(self, status: Optional[str] = None) -> List[Dict]:
        query = "SELECT * FROM persons"
        params: List[object] = []
        if status:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC"

        with self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                item["metadata"] = self._loads_metadata(item.get("metadata_json"))
                result.append(item)
            return result

    def add_person(
        self,
        name: str,
        full_name: Optional[str] = None,
        vector_storage_key: Optional[str] = None,
        vector_count: int = 0,
        metadata: Optional[Dict] = None,
    ) -> str:
        person_id = str(uuid.uuid4())
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO persons (
                    person_id, name, full_name, status,
                    vector_storage_key, vector_count, metadata_json
                ) VALUES (?, ?, ?, 'active', ?, ?, ?)
                """,
                (
                    person_id,
                    name,
                    full_name or name,
                    vector_storage_key,
                    int(vector_count),
                    metadata_json,
                ),
            )
        return person_id

    def update_person(
        self,
        person_id: str,
        status: Optional[str] = None,
        vector_storage_key: Optional[str] = None,
        vector_count: Optional[int] = None,
        metadata: Optional[Dict] = None,
    ) -> bool:
        fields = []
        params = []

        if status is not None:
            fields.append("status = ?")
            params.append(status)
        if vector_storage_key is not None:
            fields.append("vector_storage_key = ?")
            params.append(vector_storage_key)
        if vector_count is not None:
            fields.append("vector_count = ?")
            params.append(int(vector_count))
        if metadata is not None:
            fields.append("metadata_json = ?")
            params.append(json.dumps(metadata, ensure_ascii=False))

        fields.append("updated_at = CURRENT_TIMESTAMP")
        params.append(person_id)

        with self._connect() as conn:
            cursor = conn.execute(
                f"UPDATE persons SET {', '.join(fields)} WHERE person_id = ?",
                tuple(params),
            )
            return cursor.rowcount > 0

    # ==================== ACCESS LOGS ====================

    def add_access_log(
        self,
        result: str,
        person_id: Optional[str] = None,
        camera_id: Optional[str] = None,
        confidence: Optional[float] = None,
        image_path: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        log_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO access_logs (
                    log_id, person_id, camera_id, result,
                    confidence, image_path, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    log_id,
                    person_id,
                    camera_id,
                    result,
                    confidence,
                    image_path,
                    json.dumps(metadata or {}, ensure_ascii=False),
                ),
            )
        return log_id

    def list_access_logs(self, limit: int = 100, person_id: Optional[str] = None) -> List[Dict]:
        query = "SELECT * FROM access_logs"
        params: List[object] = []
        if person_id:
            query += " WHERE person_id = ?"
            params.append(person_id)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(int(limit))

        with self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                item["metadata"] = self._loads_metadata(item.get("metadata_json"))
                result.append(item)
            return result

    def get_stats(self) -> Dict[str, int]:
        with self._connect() as conn:
            persons = conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
            active_persons = conn.execute("SELECT COUNT(*) FROM persons WHERE status = 'active'").fetchone()[0]
            logs = conn.execute("SELECT COUNT(*) FROM access_logs").fetchone()[0]
            return {
                "persons": int(persons),
                "active_persons": int(active_persons),
                "access_logs": int(logs),
            }


_sqlite_manager: Optional[SQLiteManager] = None


def get_sqlite_manager() -> SQLiteManager:
    global _sqlite_manager
    if _sqlite_manager is None:
        _sqlite_manager = SQLiteManager()
    return _sqlite_manager


__all__ = ["SQLiteManager", "get_sqlite_manager"]