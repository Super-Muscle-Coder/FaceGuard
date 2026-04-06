# scripts/models

## Thư mục này dùng để làm gì
Lưu model ONNX chuẩn của FaceGuard dùng cho detect + embedding.

## Bên trong chứa gì
- `scrfd_10g_bnkps.onnx` (face detection + landmarks)
- `glintr100.onnx` (ArcFace embedding)

## Công dụng từng tệp mã
- `scrfd_10g_bnkps.onnx`: phát hiện khuôn mặt và landmarks phục vụ align.
- `glintr100.onnx`: trích vector embedding 512D cho nhận diện/fine-tune.

## Tương tác với thư mục nào
- Được `core/adapters/ModelAdapter.py` load theo đường dẫn trong `config/settings/embedding.py`.
- Dùng bởi `core/services/*` (IoTService, PackagingService, FineTuneService, ...).
