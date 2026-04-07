# FaceGuard Onboarding Guide

## 1. Hệ thống FaceGuard là gì?
FaceGuard là hệ thống **AIoT nhận diện khuôn mặt và kiểm soát ra vào**.

Hệ thống gồm 2 luồng chính:
- **Training/Fine-tune**: thu dữ liệu -> xử lý frame -> huấn luyện head classifier -> đồng bộ runtime DB.
- **Runtime IoT**: nhận ảnh từ ESP32-CAM -> nhận diện hybrid -> trả quyết định mở/không mở khóa.

Mục tiêu là cân bằng giữa:
- độ chính xác nhận diện,
- khả năng triển khai trên phần cứng phổ thông,
- và vận hành ổn định trong thực tế.

---

## 2. Kiến trúc thư mục và hạ tầng

## 2.1 Kiến trúc mã nguồn
- `config/settings/`: cấu hình từng phase và runtime.
- `core/entities/`: dataclass/thực thể dữ liệu.
- `core/adapters/`: lớp tích hợp model/storage/video.
- `core/services/`: nghiệp vụ chính (pipeline + runtime).
- `api/`: Flask API cho service IoT.
- `scripts/`: tiện ích vận hành và firmware ESP32-CAM.
- `docker/`: Dockerfile + compose + hướng dẫn triển khai.

## 2.2 Hạ tầng dữ liệu
- **MinIO**: lưu vector/object runtime (ví dụ NPZ DB).
- **SQLite**: lưu metadata users, trạng thái active, access logs.
- **Training artifacts**: lưu tại `training_plots/`, `database/`, `data/`.

## 2.3 Hạ tầng triển khai container
Stack Docker gồm 3 service:
1. `faceguard-training`
2. `faceguard-iot`
3. `minio`

---

## 3. Mục tiêu dự án
1. Xây dựng pipeline huấn luyện tinh gọn: freeze backbone ArcFace, fine-tune head nhẹ.
2. Triển khai runtime nhận diện thời gian thực cho ESP32-CAM.
3. Tích hợp mở khóa vật lý qua relay + solenoid.
4. Chuẩn hóa triển khai Docker để chạy trên nhiều máy theo trạng thái fresh.

---

## 4. Những gì đã đạt được
- Hoàn thiện pipeline nhiều phase: DataCollection -> FrameExtraction -> FrameSanitizer -> FineTune -> Runtime sync.
- Hoàn thiện IoT API runtime với kiểm soát payload, metrics, debug endpoints.
- Hoàn thiện firmware ESP32-CAM:
  - chunked upload,
  - warm-up frames,
  - smart retry profile,
  - logic kích relay mở khóa.
- Đồng bộ runtime data theo mô hình MinIO + SQLite.
- Đóng gói Docker 3 service, push image lên DockerHub (training + iot).
- Chuẩn hóa tài liệu vận hành cho khách hàng.

---

## 5. Những gì chưa đạt được / hạn chế hiện tại
- Chưa có anti-spoofing chuyên sâu ở runtime.
- Chất lượng nhận diện còn phụ thuộc điều kiện ánh sáng và góc camera.
- GUI training trong container phụ thuộc môi trường host hỗ trợ display forwarding.
- Mạng WiFi có chính sách client isolation có thể chặn ESP32 truy cập IoT server LAN.

---

## 6. Khi mới clone code về cần làm gì?
1. Đọc nhanh:
   - `README.md` (root)
   - `docker/CUSTOMER_QUICKSTART.md`
   - `documents/esp32cam_firmware_guide.md`
2. Chuẩn bị model ở `scripts/models/` (`scrfd_10g_bnkps.onnx`, `glintr100.onnx`).
3. Khởi chạy stack Docker theo hướng dẫn quickstart.
4. Kiểm tra service:
   - IoT health endpoint
   - MinIO console
5. Cấu hình firmware ESP32 trỏ đúng `API_HOST` và API key.

---

## 7. Luồng sử dụng chuẩn
1. Thu dữ liệu người dùng bằng training phase GUI.
2. Chạy fine-tune và deploy runtime artifacts.
3. Kiểm tra nhanh bằng PackagingService (phase realtime test).
4. Bật IoT runtime, kết nối ESP32-CAM.
5. Theo dõi access logs, threshold, runtime state để tuning tiếp.

---

## 8. Định hướng mở rộng
- Voting đa frame để tăng độ ổn định quyết định mở khóa.
- Anti-spoofing cho ảnh/video giả.
- Versioning runtime DB + rollback an toàn.
- Tăng tự động hóa CI/CD và kiểm thử end-to-end.
