# Copilot Instructions

## Project Guidelines
- Dự án FaceGuard tổ chức theo kiến trúc module rõ ràng gồm 4 thành phần chính: config/settings (tham số cấu hình), entities (dataclass/thực thể không chứa logic), adapters (công cụ kết nối/tương tác), services (logic nghiệp vụ chính sử dụng 3 thành phần còn lại).
- Khi chuẩn hóa module của dự án FaceGuard, các tệp __init__ của config/settings, entities, adapters, services phải export rõ ràng bằng __all__.


## GUI Development
- Trong FaceGuard, PackagingService (phase realtime test) cần xây GUI bằng PySide6 và ưu tiên giao diện dễ nhìn/dễ dùng.
- DataCollectionGUI (Phase 1) phải là GUI lớn dùng camera laptop để quay 3 lượt (frontal/horizontal/vertical), kiểm định chất lượng video ngay sau mỗi lượt bằng VideoQualityService, cho phép Keep hoặc Re-record, rồi mới chuyển pipeline sang các phase tiếp theo.