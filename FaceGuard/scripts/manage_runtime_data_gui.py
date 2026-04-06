"""GUI tool to manage runtime data consistency between MinIO and SQLite.

Features:
- View persons from SQLite
- View MinIO objects
- Create/update person metadata
- Delete with synchronized cleanup (SQLite + MinIO) and confirmation
"""

from __future__ import annotations

import io
import sys
from datetime import datetime

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QSplitter,
)

from config.settings import IOT_MINIO_KEYS
from core.adapters.StorageAdapter import StorageAdapter
from core.storage import get_sqlite_manager


class RuntimeDataManagerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FaceGuard Runtime Data Manager")
        self.resize(1300, 820)

        self.sqlite = get_sqlite_manager()
        self.storage = StorageAdapter()

        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QVBoxLayout(root)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(14)

        title = QLabel("FaceGuard Runtime Data Manager")
        title.setObjectName("title")
        subtitle = QLabel("SQLite metadata + MinIO vectors (synchronized CRUD)")
        subtitle.setObjectName("subtitle")

        main_layout.addWidget(title)
        main_layout.addWidget(subtitle)

        splitter = QSplitter(Qt.Horizontal)

        # Left: SQLite persons
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setSpacing(10)
        left_layout.addWidget(QLabel("SQLite Persons"))
        self.person_list = QListWidget()
        left_layout.addWidget(self.person_list)

        form_row = QHBoxLayout()
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("name")
        self.fullname_input = QLineEdit()
        self.fullname_input.setPlaceholderText("full name")
        form_row.addWidget(self.name_input)
        form_row.addWidget(self.fullname_input)
        left_layout.addLayout(form_row)

        btn_row = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh")
        self.btn_add = QPushButton("Add/Upsert")
        self.btn_delete = QPushButton("Delete Selected (Sync)")
        self.btn_purge_orphans = QPushButton("Purge Orphan Metadata")
        btn_row.addWidget(self.btn_refresh)
        btn_row.addWidget(self.btn_add)
        btn_row.addWidget(self.btn_delete)
        btn_row.addWidget(self.btn_purge_orphans)
        left_layout.addLayout(btn_row)

        # Right: MinIO objects
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setSpacing(10)
        right_layout.addWidget(QLabel("MinIO Objects"))
        self.object_list = QListWidget()
        right_layout.addWidget(self.object_list)

        self.btn_delete_object = QPushButton("Delete Selected Object (Sync)")
        right_layout.addWidget(self.btn_delete_object)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        right_layout.addWidget(self.log_box)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([560, 700])

        main_layout.addWidget(splitter)

        self.btn_refresh.clicked.connect(self.refresh_all)
        self.btn_add.clicked.connect(self.add_or_update_person)
        self.btn_delete.clicked.connect(self.delete_selected_person_sync)
        self.btn_purge_orphans.clicked.connect(self.purge_orphan_metadata_sync)
        self.btn_delete_object.clicked.connect(self.delete_selected_object_sync)

        self.setStyleSheet(
            """
            QMainWindow, QWidget { background:#0b1220; color:#e5e7eb; font-size:16px; }
            QLabel#title { font-size:30px; font-weight:700; color:#f8fafc; padding:6px 0; }
            QLabel#subtitle { font-size:18px; color:#94a3b8; padding-bottom:8px; }
            QListWidget, QTextEdit, QLineEdit {
                background:#111827; border:1px solid #334155; border-radius:10px;
                padding:10px; font-size:16px;
            }
            QPushButton {
                background:#1f2937; border:1px solid #334155; border-radius:10px;
                padding:12px 14px; font-size:16px; font-weight:600;
            }
            QPushButton:hover { background:#334155; }
            """
        )

        self.refresh_all()

    def _log(self, text: str):
        self.log_box.append(f"[{datetime.now().strftime('%H:%M:%S')}] {text}")

    def refresh_all(self):
        self.person_list.clear()
        for p in self.sqlite.list_persons():
            self.person_list.addItem(
                f"{p['name']} | id={p['person_id'][:8]} | key={p.get('vector_storage_key') or '-'}"
            )

        self.object_list.clear()
        for key in self.storage.list(prefix="", max_keys=5000):
            self.object_list.addItem(key)

        self._log("Refreshed SQLite persons and MinIO objects")

    def add_or_update_person(self):
        name = self.name_input.text().strip()
        full_name = self.fullname_input.text().strip() or name
        if not name:
            QMessageBox.warning(self, "Validation", "Name is required")
            return

        key = IOT_MINIO_KEYS["DATABASE_NPZ"]
        existing = self.sqlite.get_person_by_name(name)
        if existing is None:
            self.sqlite.add_person(
                name=name,
                full_name=full_name,
                vector_storage_key=key,
                vector_count=1,
                metadata={"managed_by": "gui"},
            )
            self._log(f"Added person: {name}")
        else:
            self.sqlite.update_person(
                existing["person_id"],
                status="active",
                vector_storage_key=key,
                vector_count=1,
                metadata={"managed_by": "gui", "updated": True},
            )
            self._log(f"Updated person: {name}")

        self.refresh_all()

    def _delete_person_and_logs(self, person_id: str):
        with self.sqlite._connect() as conn:  # controlled internal use for tool
            conn.execute("DELETE FROM access_logs WHERE person_id = ?", (person_id,))
            conn.execute("DELETE FROM persons WHERE person_id = ?", (person_id,))

    def _remove_person_vector_from_shared_npz(self, key: str, person_name: str) -> bool:
        """Remove only one person's vector from shared MinIO NPZ object."""
        raw = self.storage.get(key, use_cache=False)
        if raw is None:
            self._log(f"MinIO key not found: {key}")
            return False

        try:
            npz = np.load(io.BytesIO(raw), allow_pickle=False)
            merged = {name: np.asarray(npz[name], dtype=np.float32) for name in npz.files}
        except Exception as ex:
            self._log(f"Failed to parse NPZ '{key}': {ex}")
            return False

        if person_name not in merged:
            self._log(f"Person '{person_name}' not found in NPZ '{key}' (metadata will still be removed)")
            return True

        merged.pop(person_name, None)

        buff = io.BytesIO()
        np.savez_compressed(buff, **merged)
        ok = self.storage.put(key, buff.getvalue())
        if not ok:
            self._log(f"Failed to upload updated NPZ after removing '{person_name}'")
            return False

        self._log(f"Removed vector for '{person_name}' from MinIO key '{key}' (remaining={len(merged)})")
        return True

    def _is_person_vector_present(self, key: str, person_name: str) -> bool:
        raw = self.storage.get(key, use_cache=False)
        if raw is None:
            return False
        try:
            npz = np.load(io.BytesIO(raw), allow_pickle=False)
            return person_name in set(npz.files)
        except Exception:
            return False

    def _get_orphan_persons(self) -> list[dict]:
        orphans: list[dict] = []
        persons = self.sqlite.list_persons()
        for person in persons:
            key = person.get("vector_storage_key") or IOT_MINIO_KEYS["DATABASE_NPZ"]
            if not self.storage.exists(key):
                orphans.append(person)
                continue
            if not self._is_person_vector_present(key, person["name"]):
                orphans.append(person)
        return orphans

    def purge_orphan_metadata_sync(self):
        orphans = self._get_orphan_persons()
        if not orphans:
            QMessageBox.information(self, "Orphan metadata", "No orphan metadata found.")
            return

        names = ", ".join(p["name"] for p in orphans)
        confirm = QMessageBox.question(
            self,
            "Confirm orphan metadata purge",
            (
                f"Delete metadata for orphan persons with missing vectors in MinIO?\n\n"
                f"Orphans detected: {names}\n\n"
                "This will remove only SQLite metadata/access logs for these users."
            ),
            QMessageBox.Yes | QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return

        for person in orphans:
            self._delete_person_and_logs(person["person_id"])

        self._log(f"Purged orphan metadata for {len(orphans)} persons: {names}")
        self.refresh_all()

    def _purge_local_runtime_artifacts(self):
        """Remove local runtime DB artifacts that can cause stale recognition fallback."""
        from config.settings import DATABASE_DIR

        removed = 0
        for p in [DATABASE_DIR / "face_recognition_db.npz", DATABASE_DIR / "fine_tune_head.pt"]:
            if p.exists():
                try:
                    p.unlink()
                    removed += 1
                    self._log(f"Removed local runtime artifact: {p.name}")
                except Exception as ex:
                    self._log(f"Failed to remove local artifact {p.name}: {ex}")
        if removed == 0:
            self._log("No local runtime artifacts to purge")

    def delete_selected_person_sync(self):
        item = self.person_list.currentItem()
        if item is None:
            return

        raw = item.text()
        name = raw.split("|")[0].strip()

        confirm = QMessageBox.question(
            self,
            "Confirm synchronized delete",
            (
                f"Delete person '{name}' from SQLite and remove corresponding vector from MinIO?\n\n"
                "This action helps keep MinIO/SQLite consistent."
            ),
            QMessageBox.Yes | QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return

        person = self.sqlite.get_person_by_name(name)
        if person:
            key = person.get("vector_storage_key") or IOT_MINIO_KEYS["DATABASE_NPZ"]

            # Shared DB key: only remove this person's vector, do not delete whole object.
            if not self._remove_person_vector_from_shared_npz(key, name):
                QMessageBox.warning(
                    self,
                    "Sync delete failed",
                    (
                        "Không thể cập nhật MinIO vector DB cho thao tác xóa người dùng này.\n"
                        "SQLite chưa bị thay đổi để tránh mất đồng bộ."
                    ),
                )
                return

            self._delete_person_and_logs(person["person_id"])
            self._log(f"Deleted person '{name}' from SQLite + removed only their vector in MinIO")

        self.refresh_all()

    def delete_selected_object_sync(self):
        item = self.object_list.currentItem()
        if item is None:
            return
        key = item.text().strip()

        confirm = QMessageBox.question(
            self,
            "Confirm synchronized delete",
            (
                f"Delete MinIO object '{key}' and remove linked persons in SQLite?\n\n"
                "This action helps keep MinIO/SQLite consistent."
            ),
            QMessageBox.Yes | QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return

        self.storage.delete(key)

        with self.sqlite._connect() as conn:  # controlled internal use for tool
            rows = conn.execute("SELECT person_id FROM persons WHERE vector_storage_key = ?", (key,)).fetchall()
            for r in rows:
                pid = r[0]
                conn.execute("DELETE FROM access_logs WHERE person_id = ?", (pid,))
            conn.execute("DELETE FROM persons WHERE vector_storage_key = ?", (key,))

        self._log(f"Deleted object '{key}' and linked SQLite persons (sync delete)")
        self._purge_local_runtime_artifacts()
        self.refresh_all()


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    w = RuntimeDataManagerWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
