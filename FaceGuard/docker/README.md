# docker

## Thư mục này dùng để làm gì
Đóng gói và triển khai FaceGuard bằng Docker với kiến trúc 3 service.

## Bên trong chứa gì
- `Dockerfile.training`: image training pipeline.
- `Dockerfile.iot`: image runtime IoT API.
- `docker-compose.yaml`: định nghĩa stack 3 service (training, iot, minio).
- `requirements.training.txt`, `requirements.iot.txt`: thư viện riêng theo service.
- `USAGE.md`, `CUSTOMER_QUICKSTART.md`: hướng dẫn vận hành.

## Công dụng từng tệp mã
- Dockerfile: build image code-only (không mang data local).
- Compose: orchestrate service + volume + network.
- Requirements: đảm bảo dependency đúng theo workload.
- Docs: hướng dẫn khách hàng pull và chạy.

## Tương tác với thư mục nào
- Copy source từ `api/`, `config/`, `core/`, `scripts/models`.
- Gắn volume runtime cho `database/`, `data/`, `training_plots`.
- Kết nối MinIO để lưu object/vector runtime.
