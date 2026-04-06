# core/entities

## Thư mục này dùng để làm gì
Định nghĩa thực thể dữ liệu (dataclass/model nhẹ), không chứa logic nghiệp vụ nặng.

## Bên trong chứa gì
- Các entity theo phase: `IoT.py`, `packager.py`, `fine_tune.py`, `frame_extraction.py`, `frame_sanitizer.py`, `recognition.py`, `sanitizer.py`, `video_quality.py`, `data_collection.py`, `embedding.py`.
- `__init__.py` export tập trung.

## Công dụng từng tệp mã
- Mỗi file entity biểu diễn dữ liệu input/output của một phase hoặc service.
- Giúp type-safe, dễ truyền dữ liệu giữa adapter/service/API.

## Tương tác với thư mục nào
- Được `core/services` sinh ra/tiêu thụ.
- Được `api/routes` trả về thành JSON response.
