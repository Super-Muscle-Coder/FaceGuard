"""List SQLite metadata entries for FaceGuard."""

from __future__ import annotations

from core.storage import get_sqlite_manager


def main() -> int:
    db = get_sqlite_manager()

    stats = db.get_stats()
    print("=== SQLite Stats ===")
    for k, v in stats.items():
        print(f"{k}: {v}")

    print("\n=== Persons ===")
    persons = db.list_persons()
    if not persons:
        print("(empty)")
    for p in persons:
        print(
            f"- {p['person_id'][:8]} | name={p['name']} | status={p['status']} | "
            f"vectors={p['vector_count']} | key={p['vector_storage_key']}"
        )

    print("\n=== Access Logs (latest 50) ===")
    logs = db.list_access_logs(limit=50)
    if not logs:
        print("(empty)")
    for log in logs:
        print(
            f"- {log['created_at']} | result={log['result']} | "
            f"confidence={log['confidence']} | camera={log['camera_id']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
