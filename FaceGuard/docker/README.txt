Trong thư mục docker này có cấu hình triển khai 3 dịch vụ chính của FaceGuard:

1) faceguard-training:
- Chạy pipeline training/fine-tune (DataCollection -> FrameExtraction -> FrameSanitizer -> FineTune).
- Có thể dùng để kiểm thử nhanh bằng phase PackagingService khi cần.

2) faceguard-iot:
- Chạy runtime recognition API cho ESP32-CAM.
- Thành phần chính: config/settings/IoT.py, entities/IoT.py, services/IoTService.py.

3) minio:
- Kho object storage cho vector/runtime artifacts.

Nguyên tắc đóng gói:
- Image chỉ chứa code + thư viện.
- Dữ liệu runtime (SQLite, data buffer, training plots, checkpoint) nằm ở volume.
- Khi người dùng pull về máy mới, hệ thống ở trạng thái fresh (không mang dữ liệu từ máy build).

Xem hướng dẫn chi tiết chạy dịch vụ tại: docker/USAGE.md
