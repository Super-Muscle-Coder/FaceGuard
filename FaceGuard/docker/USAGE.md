# FaceGuard Docker Usage (3 services)

This stack runs **3 containers**:
- `faceguard-training` (fine-tune/training pipeline)
- `faceguard-iot` (runtime IoT recognition API)
- `faceguard-minio` (object storage)

## Important design
- Images contain **code + dependencies only**.
- User/runtime data is stored in Docker volumes (fresh on a new host).
- No local training data, SQLite, or embeddings are baked into images.

## Start all services
```bash
docker compose -f docker/docker-compose.yaml pull
docker compose -f docker/docker-compose.yaml up -d
```

## Access points
- IoT API health: `http://localhost:5000/health`
- MinIO API: `http://localhost:9000`
- MinIO Console: `http://localhost:9001`

## Run training pipeline manually (interactive)
By default `faceguard-training` stays alive and waits for user commands.
Run full training pipeline:
```bash
docker exec -it faceguard-training python FineTuneEntry.py
```

Run packaging GUI test phase:
```bash
docker exec -it faceguard-training python -c "from core.services.PackagingService import launch_packaging_gui; raise SystemExit(launch_packaging_gui())"
```

> Note: PySide6 GUI in container depends on host GUI forwarding setup. For Docker Desktop on Windows/macOS,
> you may run training GUI locally (outside container) while still using MinIO + IoT containers.

## Data persistence
Named volumes used:
- `minio_data`
- `training_database`
- `training_data`
- `training_plots`
- `iot_database`

On a **new machine**, these volumes start empty (fresh state).

## Stop services
```bash
docker compose -f docker/docker-compose.yaml down
```

## Reset to fresh state (delete all runtime data)
```bash
docker compose -f docker/docker-compose.yaml down -v
```
