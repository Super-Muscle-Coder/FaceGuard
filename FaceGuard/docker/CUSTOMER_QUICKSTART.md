# FaceGuard Customer Quickstart

Tài liệu này hướng dẫn khách hàng chạy hệ thống FaceGuard bằng Docker Compose với 3 service:

- `faceguard-training` (pipeline training/fine-tune + packaging test)
- `faceguard-iot` (runtime IoT API cho ESP32-CAM)
- `faceguard-minio` (object storage)

## 1) Yêu cầu môi trường
- Cài Docker Desktop (Windows/macOS) hoặc Docker Engine + Docker Compose plugin (Linux)
- Máy có camera (nếu chạy phase GUI DataCollection/Packaging)
- Mạng LAN ổn định cho ESP32-CAM và máy chạy IoT API

## 2) Pull source/compose và chạy stack
Từ thư mục project:

```bash
docker compose -f docker/docker-compose.yaml pull
docker compose -f docker/docker-compose.yaml up -d
```

Sau khi chạy thành công:
- IoT health: `http://localhost:5000/health`
- MinIO Console: `http://localhost:9001`

## 3) Runtime data policy (fresh by default)
Image chỉ chứa code + dependency.
Dữ liệu runtime được lưu trong named volumes:
- `minio_data`
- `training_database`
- `training_data`
- `training_plots`
- `iot_database`

Khi khách hàng chạy trên máy mới, hệ thống ở trạng thái fresh (không có dữ liệu cá nhân từ máy phát triển).

## 4) Cách sử dụng service training
Container training mặc định ở chế độ chờ (sleep) để người dùng tự chạy phase mong muốn.

### 4.1 Chạy pipeline fine-tune
```bash
docker exec -it faceguard-training python FineTuneEntry.py
```

### 4.2 Chạy packaging GUI (kiểm thử nhận diện)
```bash
docker exec -it faceguard-training python -c "from core.services.PackagingService import launch_packaging_gui; raise SystemExit(launch_packaging_gui())"
```

> Lưu ý GUI PySide6 trong container cần host hỗ trợ GUI forwarding. Nếu không có, chạy GUI trực tiếp ở host local và giữ MinIO + IoT chạy trong Docker.

## 5) Cấu hình ESP32-CAM
Trong firmware ESP32:
- `API_HOST` = IP máy chạy `faceguard-iot`
- `API_PORT` = `5000`
- `API_PATH` = `/api/v1/recognize`
- `API_KEY_HEADER`/`API_KEY` phải trùng với config IoT service

## 6) Vận hành MinIO
- Endpoint nội bộ container: `minio:9000`
- Endpoint host: `localhost:9000`
- Console: `localhost:9001`
- Bucket mặc định: `faceguard`

## 7) Dừng và reset
Dừng stack:
```bash
docker compose -f docker/docker-compose.yaml down
```

Reset toàn bộ dữ liệu runtime (về fresh):
```bash
docker compose -f docker/docker-compose.yaml down -v
```

## 8) Gợi ý publish image DockerHub
Ví dụ:
```bash
docker tag faceguard-training:latest <dockerhub_user>/faceguard-training:latest
docker tag faceguard-iot:latest <dockerhub_user>/faceguard-iot:latest
docker push <dockerhub_user>/faceguard-training:latest
docker push <dockerhub_user>/faceguard-iot:latest
```

MinIO dùng image official `minio/minio:latest`, không cần tự build/push.

## 9) Image đang dùng mặc định trong compose
- `supermusclecoder/faceguard-training:latest`
- `supermusclecoder/faceguard-iot:latest`
