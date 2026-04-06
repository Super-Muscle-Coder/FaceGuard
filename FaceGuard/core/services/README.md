# core/services

## Thư mục này dùng để làm gì
Chứa toàn bộ logic nghiệp vụ chính của FaceGuard, từ pipeline training đến runtime IoT.

## Bên trong chứa gì
- Data pipeline: `DataCollectionService.py`, `DataCollectionGUI.py`, `VideoQualityService.py`, `FrameExtractionService.py`, `FrameSanitizerService.py`, `EmbeddingService.py`, `SanitizerService.py`, `RecognitionService.py`.
- Training: `FineTuneService.py`, `MasterTraningService.py`, `MasterWorkflowService.py`.
- Runtime: `IoTService.py`, `PackagingService.py`.
- `__init__.py`: lazy export service bằng `__all__`.

## Công dụng từng tệp mã
- `DataCollectionGUI.py`: GUI quay 3 video (frontal/horizontal/vertical), kiểm định quality và keep/re-record.
- `DataCollectionService.py`: wrapper orchestration cho phase thu thập dữ liệu.
- `VideoQualityService.py`: đánh giá chất lượng video (Gate 1).
- `FrameExtractionService.py`: trích frame theo chính sách sampling/chất lượng.
- `FrameSanitizerService.py`: lọc và chuẩn hóa frame cho ArcFace.
- `EmbeddingService.py`: trích vector embedding từ frame.
- `SanitizerService.py`: làm sạch/tách split dữ liệu theo chiến lược phase cũ.
- `RecognitionService.py`: đánh giá nhận diện offline/research.
- `FineTuneService.py`: fine-tune head, lưu checkpoint/plots/summary, sync MinIO + SQLite.
- `MasterTraningService.py`: chạy chuỗi phase training theo pipeline.
- `MasterWorkflowService.py`: workflow orchestration cho luồng đầy đủ nhiều phase.
- `PackagingService.py`: phase 6 realtime GUI test (PySide6), nhận diện trực tiếp camera.
- `IoTService.py`: service runtime cho API IoT, hybrid recognize, guard sync SQLite active.

## Tương tác với thư mục nào
- Dùng `core/adapters` để thao tác camera/model/storage.
- Dùng `core/entities` làm schema dữ liệu.
- Dùng `core/storage/SQLite.py` cho metadata.
- Đọc config từ `config/settings/*`.
- Runtime IoT được gọi từ `api/`.
