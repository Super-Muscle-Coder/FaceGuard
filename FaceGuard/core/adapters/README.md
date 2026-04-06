# core/adapters

## Thư mục này dùng để làm gì
Cung cấp các adapter để kết nối tài nguyên bên ngoài (camera, model ONNX, MinIO, tensor/data loader).

## Bên trong chứa gì
- `VideoAdapter.py`
- `ModelAdapter.py`
- `StorageAdapter.py`
- `FineTuneAdapter.py`
- `__init__.py`

## Công dụng từng tệp mã
- `VideoAdapter.py`: thao tác camera/video I/O.
- `ModelAdapter.py`: load SCRFD/ArcFace, detect/alignment/extract embedding.
- `StorageAdapter.py`: đọc/ghi object storage MinIO + cache.
- `FineTuneAdapter.py`: helper data pipeline cho huấn luyện PyTorch.
- `__init__.py`: export adapter thống nhất bằng `__all__`.

## Tương tác với thư mục nào
- Được `core/services/*` sử dụng để tách biệt nghiệp vụ và hạ tầng.
- Đọc cấu hình từ `config/settings/*`.
