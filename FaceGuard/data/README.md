# data

## Thư mục này dùng để làm gì
Lưu dữ liệu ảnh/video phục vụ pipeline training và buffer vận hành.

## Bên trong chứa gì
- Thư mục theo từng người dùng (`Nghi`, `Lien`, ...).
- `temp/` cho dữ liệu trung gian theo phase.
- Cấu trúc con thường gồm `video/`, `frames/`, `sanitized_frames/` tùy phase.

## Công dụng từng tệp mã
- Không chứa mã nguồn; đây là dữ liệu input/output của pipeline.

## Tương tác với thư mục nào
- Được tạo/đọc bởi `core/services/DataCollectionService.py`, `FrameExtractionService.py`, `FrameSanitizerService.py`, `FineTuneService.py`.
- Đường dẫn chuẩn được định nghĩa tại `config/settings/path.py` và các config phase liên quan.
