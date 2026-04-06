# core/storage

## Thư mục này dùng để làm gì
Cung cấp lớp lưu trữ metadata nội bộ bằng SQLite cho FaceGuard.

## Bên trong chứa gì
- `SQLite.py`: SQLite manager (persons, access_logs, thống kê, cập nhật trạng thái).
- `__init__.py`: export helper `get_sqlite_manager()`.

## Công dụng từng tệp mã
- `SQLite.py`: tạo/migrate schema, CRUD person, ghi access log, truy vấn runtime stats.
- `__init__.py`: singleton/factory để services dùng chung connection manager.

## Tương tác với thư mục nào
- Được `core/services/FineTuneService.py`, `IoTService.py`, `PackagingService.py` dùng để đồng bộ trạng thái người dùng active và log truy cập.
- Kết hợp `core/adapters/StorageAdapter.py` (MinIO) để tạo mô hình dữ liệu 2 tầng: vector object + metadata relational.
