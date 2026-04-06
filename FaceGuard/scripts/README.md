# scripts

## Thư mục này dùng để làm gì
Chứa script vận hành nhanh, công cụ hỗ trợ kiểm tra dữ liệu và firmware ESP32-CAM.

## Bên trong chứa gì
- `RawArduinoCode_ESP32CAM.txt`: firmware client IoT cho ESP32-CAM.
- `download_models.py`: tải model ONNX cần thiết.
- `list_all_minio_files.py`: liệt kê object trong MinIO.
- `list_all_sqlite_entries.py`: kiểm tra dữ liệu SQLite.
- `manage_runtime_data_gui.py`: công cụ GUI quản trị runtime data.
- `models/`: thư mục model `.onnx` dùng cho training/runtime.

## Công dụng từng tệp mã
- Firmware ESP32: gửi ảnh, parse kết quả nhận diện, kích relay mở khóa.
- Các script list/download/manage: hỗ trợ vận hành và debug nhanh mà không cần chạy full app.

## Tương tác với thư mục nào
- Dùng chung config với `config/settings/*`.
- Tương tác storage qua `core/adapters` và `core/storage`.
- Model trong `scripts/models` được `ModelAdapter` sử dụng ở cả training và runtime.
