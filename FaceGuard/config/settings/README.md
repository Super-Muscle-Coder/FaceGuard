# config/settings

## Thư mục này dùng để làm gì
Quản lý cấu hình chi tiết cho từng phase của pipeline và runtime service.

## Bên trong chứa gì
- `path.py`: định nghĩa đường dẫn chuẩn nội bộ dự án.
- `IoT.py`: cấu hình service IoT/API.
- `packager.py`: cấu hình phase realtime test GUI.
- `fine_tune.py`: cấu hình huấn luyện head.
- `data_collection.py`, `frame_extraction.py`, `frame_sanitizer.py`, `embedding.py`, `sanitizer.py`, `recognition.py`, `video_quality.py`: cấu hình cho từng phase xử lý dữ liệu.
- `storage.py`: cấu hình MinIO/cache.
- `logging.py`: nhãn/log helper.
- `__init__.py`: export cấu hình thống nhất bằng `__all__`.

## Công dụng từng tệp mã
Mỗi tệp quản lý tham số riêng theo domain để services có thể đổi hành vi mà không đổi code nghiệp vụ.

## Tương tác với thư mục nào
- Được `core/services/*` và `api/*` sử dụng trực tiếp.
- Kết hợp `core/storage` để xác định vị trí SQLite/DB/runtime artifacts.
