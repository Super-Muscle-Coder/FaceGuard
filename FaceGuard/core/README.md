# core

## Thư mục này dùng để làm gì
Chứa lõi nghiệp vụ của FaceGuard theo kiến trúc module: adapters, entities, services, storage.

## Bên trong chứa gì
- `adapters/`: lớp tích hợp camera/model/storage/fine-tune utils.
- `entities/`: dataclass thực thể dữ liệu.
- `services/`: nghiệp vụ chính của toàn hệ thống.
- `storage/`: lớp truy cập SQLite.

## Công dụng tệp mã trong thư mục
- `__init__.py` (nếu có): đóng vai trò package root.

## Tương tác với thư mục nào
- Nhận cấu hình từ `config/settings`.
- Được gọi bởi `api/` và `FineTuneEntry.py`.
- Dùng `scripts/models` làm model runtime/training.
