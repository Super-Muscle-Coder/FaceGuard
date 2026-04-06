"""List all files from FaceGuard MinIO bucket."""

from __future__ import annotations

from minio import Minio

from config.settings import MINIO_STORAGE_CONFIG


def main() -> int:
    client = Minio(
        endpoint=MINIO_STORAGE_CONFIG["ENDPOINT"],
        access_key=MINIO_STORAGE_CONFIG["ACCESS_KEY"],
        secret_key=MINIO_STORAGE_CONFIG["SECRET_KEY"],
        secure=MINIO_STORAGE_CONFIG["SECURE"],
    )

    bucket = MINIO_STORAGE_CONFIG["BUCKET_NAME"]
    if not client.bucket_exists(bucket):
        print(f"Bucket not found: {bucket}")
        return 1

    print(f"Bucket: {bucket}")
    count = 0
    total_size = 0
    for obj in client.list_objects(bucket, recursive=True):
        count += 1
        total_size += obj.size or 0
        print(f"- {obj.object_name} | {obj.size} bytes")

    print(f"Total files: {count}")
    print(f"Total size : {total_size / 1024 / 1024:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
