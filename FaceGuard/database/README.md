# database

## Thư mục này dùng để làm gì
Lưu dữ liệu runtime cốt lõi của hệ thống (SQLite, checkpoint, artifacts liên quan nhận diện).

## Bên trong chứa gì
- `face_recognition.db` (SQLite metadata, có thể được tạo khi chạy).
- `fine_tune_head.pt` (checkpoint head sau huấn luyện).
- Các artifact runtime khác nếu phát sinh.

## Công dụng từng tệp mã
- SQLite: quản lý users, trạng thái active, access logs.
- Fine-tune checkpoint: hỗ trợ hybrid recognition (cosine + head).

## Tương tác với thư mục nào
- Được `core/storage/SQLite.py`, `core/services/IoTService.py`, `core/services/FineTuneService.py`, `core/services/PackagingService.py` sử dụng.
- Kết hợp MinIO để đồng bộ vector DB object.
