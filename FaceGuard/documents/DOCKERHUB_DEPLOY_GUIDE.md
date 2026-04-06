# Hướng dẫn triển khai FaceGuard từ DockerHub

Tài liệu này hướng dẫn người dùng **không cần build source**, chỉ pull image từ DockerHub và chạy hệ thống.

---

## 1) Thành phần hệ thống
FaceGuard chạy với 3 service:
1. `faceguard-training` (training/fine-tune pipeline)
2. `faceguard-iot` (API runtime nhận diện cho ESP32-CAM)
3. `minio` (object storage)

Image DockerHub:
- `supermusclecoder/faceguard-training:latest`
- `supermusclecoder/faceguard-iot:latest`
- `minio/minio:latest`

---

## 2) Yêu cầu môi trường
- Docker Desktop (Windows/macOS) hoặc Docker Engine + Compose plugin (Linux)
- Internet để pull image
- (Tùy chọn) camera nếu chạy GUI training

Kiểm tra nhanh:
```bash
docker version
docker compose version
```

---

## 3) Chuẩn bị file docker-compose
Bạn chỉ cần file:
- `docker/docker-compose.yaml`

(Đã cấu hình sẵn 3 service và image DockerHub)

---

## 4) Pull và chạy hệ thống
Từ thư mục project:
```bash
docker compose -f docker/docker-compose.yaml pull
docker compose -f docker/docker-compose.yaml up -d
```

Kiểm tra service:
```bash
docker compose -f docker/docker-compose.yaml ps
```

---

## 5) Các địa chỉ truy cập
- IoT API health: `http://localhost:5000/health`
- MinIO API: `http://localhost:9000`
- MinIO Console: `http://localhost:9001`

---

## 6) Cách chạy training sau khi stack đã lên
Container training mặc định ở trạng thái chờ. Chạy pipeline bằng:
```bash
docker exec -it faceguard-training python FineTuneEntry.py
```

Chạy packaging GUI test phase:
```bash
docker exec -it faceguard-training python -c "from core.services.PackagingService import launch_packaging_gui; raise SystemExit(launch_packaging_gui())"
```

> Lưu ý: GUI PySide6 trong container phụ thuộc host GUI forwarding. Nếu host không hỗ trợ, hãy chạy GUI local.

---

## 7) Kết nối ESP32-CAM với IoT API
Trong firmware ESP32, chỉnh:
- `API_HOST` = IP máy chạy Docker
- `API_PORT` = `5000`
- `API_PATH` = `/api/v1/recognize`
- `API_KEY_HEADER`, `API_KEY` khớp với cấu hình runtime

Nếu ESP32 không connect được nhưng server vẫn healthy, kiểm tra:
- Firewall máy host
- Mạng có chặn client-to-client (AP isolation) hay không

---

## 8) Dữ liệu runtime và trạng thái fresh
Hệ thống dùng volume Docker:
- `minio_data`
- `training_database`
- `training_data`
- `training_plots`
- `iot_database`

Máy mới pull về sẽ ở trạng thái fresh (không mang dữ liệu từ máy dev).

---

## 9) Dừng / khởi động lại / reset
Dừng stack:
```bash
docker compose -f docker/docker-compose.yaml down
```

Chạy lại stack:
```bash
docker compose -f docker/docker-compose.yaml up -d
```

Reset toàn bộ dữ liệu runtime:
```bash
docker compose -f docker/docker-compose.yaml down -v
```

---

## 10) Troubleshooting nhanh
1. `pull access denied`:
   - kiểm tra internet, kiểm tra tên image/tag.
2. `port already in use`:
   - đổi port map trong compose hoặc dừng service đang chiếm cổng.
3. IoT API không lên:
   - kiểm tra log `faceguard-iot`.
4. Training GUI không hiện:
   - do host không hỗ trợ GUI trong container; chạy local GUI hoặc cấu hình forwarding.

Xem log:
```bash
docker compose -f docker/docker-compose.yaml logs -f
```
