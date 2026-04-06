# FaceGuard

## Thư mục này dùng để làm gì
Thư mục gốc chứa toàn bộ mã nguồn và tài nguyên của hệ thống FaceGuard (AIoT nhận diện khuôn mặt + kiểm soát ra vào).

## Bên trong chứa gì
- `api/`: HTTP API cho runtime IoT.
- `config/`: cấu hình tập trung theo từng phase.
- `core/`: adapters, entities, services, storage.
- `scripts/`: công cụ vận hành và firmware ESP32-CAM.
- `docker/`: Dockerfile, compose và hướng dẫn triển khai.
- `data/`, `database/`, `training_plots/`: dữ liệu runtime/training.
- `documents/`: tài liệu kỹ thuật và báo cáo.

## Công dụng các tệp mã chính ở thư mục gốc
- `FineTuneEntry.py`: entry point chạy pipeline fine-tune/training.

## Tương tác với thư mục nào
- `FineTuneEntry.py` gọi `core/services/*`, đọc cấu hình từ `config/settings/*`, ghi dữ liệu vào `data/`, `database/`, `training_plots/`, đồng bộ MinIO qua `core/adapters/StorageAdapter.py`.
