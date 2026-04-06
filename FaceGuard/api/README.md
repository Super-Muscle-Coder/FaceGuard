# api

## Thư mục này dùng để làm gì
Cung cấp lớp API HTTP cho hệ thống IoT runtime.

## Bên trong chứa gì
- `IoTAPI.py`: app factory Flask và root endpoints.
- `routes/`: các route theo chức năng.
- `__init__.py`: export package.

## Công dụng từng tệp mã
- `IoTAPI.py`: khởi tạo app, inject `IoTService`, đăng ký blueprint.
- `routes/health.py`: health, metrics, cameras, reload, stream/debug runtime.
- `routes/recognition.py`: endpoint nhận ảnh từ ESP32-CAM và trả quyết định allow/deny.

## Tương tác với thư mục nào
- Gọi `core/services/IoTService.py` để nhận diện.
- Đọc cấu hình từ `config/settings/IoT.py`.
- Truy cập metadata qua `core/storage/SQLite.py` và MinIO qua adapter.
