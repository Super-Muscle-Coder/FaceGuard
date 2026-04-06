# BÁO CÁO CHUYÊN ĐỀ IOT: HỆ THỐNG KIỂM SOÁT RA VÀO THÔNG MINH FACEGUARD

## CHƯƠNG 1: GIỚI THIỆU ĐỀ TÀI VÀ ĐỘNG LỰC PHÁT TRIỂN

### 1.1 Motivation (Động lực)
Trong bối cảnh các hệ thống ra vào truyền thống (thẻ từ, mật khẩu) bộc lộ nhiều giới hạn về trải nghiệm và bảo mật, FaceGuard định hướng xây dựng một mô hình AIoT kết hợp nhận diện khuôn mặt và điều khiển khóa điện tử theo thời gian thực. Hướng tiếp cận này tận dụng ưu điểm của sinh trắc học (không cần mang thẻ, khó chia sẻ trái phép) và khả năng kết nối linh hoạt của thiết bị IoT.

### 1.2 Mục tiêu đồ án
Xây dựng kiến trúc Edge-to-Cloud hoàn chỉnh gồm:
- Edge device: ESP32-CAM thu ảnh và truyền qua WiFi.
- Cloud/Server: API + service suy luận nhận diện.
- Actuator: relay điều khiển khóa điện tử LY03.

Mục tiêu không chỉ dừng ở nhận diện chính xác, mà còn bao gồm độ ổn định truyền thông, tính nhất quán dữ liệu runtime và độ tin cậy khi điều khiển phần cứng thật.

### 1.3 Ý nghĩa thực tiễn
Hệ thống hướng tới triển khai chi phí thấp cho nhà ở/văn phòng nhỏ, trong khi vẫn bảo đảm:
- Quyết định ra vào dựa trên dữ liệu sinh trắc học.
- Khả năng vận hành container hóa.
- Khả năng mở rộng danh sách người dùng và bảo trì runtime DB.

---

## CHƯƠNG 2: THIẾT KẾ PHẦN CỨNG VÀ CÀI ĐẶT HỆ THỐNG

### 2.1 Thành phần phần cứng
- **ESP32-CAM**: thu ảnh JPEG, truyền dữ liệu HTTP multipart.
- **Relay 1 kênh (opto + diode)**: trung gian cách ly điều khiển.
- **Khóa điện từ LY03 (12V)**: cơ cấu chấp hành.
- **Nguồn 5V-2A**: cấp ổn định cho ESP32-CAM + relay logic.
- **Nguồn 12V**: cấp riêng cho tải khóa LY03.

### 2.2 Sơ đồ đấu nối và logic điều khiển
Thiết kế đã triển khai thành công theo 2 mạch tách biệt:

1) **Mạch 5V điều khiển (song song):**
- Adapter 5V -> chân 5V ESP32-CAM.
- Adapter 5V -> DC+ relay.
- GND chung: ESP32-CAM GND, relay DC-, GND adapter.
- GPIO ESP32-CAM -> IN relay (đang dùng IO12 trong firmware hiện tại).

2) **Mạch tải 12V khóa điện:**
- +12V adapter -> COM relay.
- NO relay -> cực dương LY03.
- Cực âm LY03 -> âm adapter 12V.

Việc dùng tiếp điểm **NO** phù hợp với đặc tính LY03: khóa chỉ mở khi được cấp điện.

### 2.3 Cài đặt firmware và server

#### a) Tối ưu firmware ESP32-CAM
Firmware đã được nâng cấp theo hướng thực chiến:
- Gửi ảnh multipart qua HTTP với cơ chế **chunked write loop** (khắc phục ghi thiếu bytes).
- Timeout rõ ràng và xử lý lỗi gửi.
- Warm-up frames trước khi chụp chính để ổn định AE/AWB.
- Smart retry capture với profile cảm biến thay thế khi confidence thấp hoặc no-face.
- Profile chống cháy sáng: tuning brightness/contrast/saturation, AE/AEC/AGC.

#### b) Triển khai server bằng Docker
Stack IoT chạy production với:
- Flask API + Gunicorn.
- MinIO (runtime vector DB dạng NPZ).
- SQLite (metadata + trạng thái active).
- Mount đồng bộ model path và database path để tránh lệch local/container.

---

## CHƯƠNG 3: THỰC NGHIỆM VÀ KẾT QUẢ CUỐI CÙNG

### 3.1 Kịch bản thực nghiệm
Các kịch bản kiểm thử chính:
- Nhận diện người hợp lệ ở nhiều trạng thái ánh sáng và khoảng cách.
- Dịch chuyển camera, thay đổi đối tượng trong khung hình.
- Kiểm thử vòng phản hồi mở khóa: ảnh -> server suy luận -> trả JSON -> kích relay.

### 3.2 Kết quả nhận diện (Hybrid Recognition)
IoTService sử dụng hybrid inference (cosine + fine-tune head), đồng bộ với runtime DB hợp lệ (MinIO ∩ SQLite active). Kết quả thực tế sau các bản tối ưu cho thấy:
- Tỷ lệ nhận diện đúng tăng rõ so với giai đoạn đầu.
- Đã xuất hiện các chuỗi phản hồi `allowed=true` liên tiếp với identity đúng (ví dụ Nghi).
- Confidence thực tế từ ESP32-CAM đạt vùng vận hành khả dụng (~0.5 trở lên trong điều kiện tốt).

### 3.3 Đánh giá hiệu năng IoT
- **Latency:** trong phạm vi đáp ứng realtime cho kịch bản ra vào nội bộ.
- **Độ ổn định truyền tin:** cải thiện mạnh sau khi áp dụng chunked write + retry logic.
- **Độ ổn định runtime:** giảm lỗi lệch dữ liệu nhờ cơ chế debug/runtime check và đồng bộ mount DB.

Các endpoint hỗ trợ vận hành/debug đã phát huy hiệu quả:
- `/health`, `/metrics`, `/cameras`, `/reload`
- `/camera/<id>/latest.jpg`, `/camera/<id>/stream.mjpg`, `/camera/<id>/viewer`
- `/debug/runtime`

### 3.4 Kết quả thực tế
Hệ thống đã đạt trạng thái hoạt động end-to-end trên phần cứng thật:
- ESP32-CAM gửi ảnh về server ổn định.
- Server trả quyết định nhận diện theo thời gian thực.
- Khi `allowed=true`, GPIO kích relay và relay cấp dòng cho LY03 mở khóa.
- Quá trình đóng/mở khóa vận hành đúng với sơ đồ phần cứng đã đấu nối.

---

## CHƯƠNG 4: TỔNG KẾT

### 4.1 Những điều đã làm được
- Hoàn thiện pipeline AIoT khép kín từ edge device đến actuator.
- Tối ưu đáng kể độ ổn định truyền ảnh từ ESP32-CAM (khắc phục partial write).
- Tối ưu chất lượng ảnh đầu vào bằng cấu hình cảm biến và retry profile.
- Đồng bộ runtime data chuyên nghiệp giữa MinIO/SQLite trong môi trường Docker.
- Triển khai thành công hệ thống điều khiển khóa LY03 qua relay bằng nhận diện khuôn mặt.

### 4.2 Những điều chưa làm được và hạn chế
- Chưa có anti-spoofing chuyên sâu trong luồng IoT realtime.
- Chất lượng ảnh vẫn phụ thuộc phần cứng camera và điều kiện ánh sáng thực địa.
- Hệ thống còn phụ thuộc độ ổn định của mạng WiFi nội bộ.

### 4.3 Hướng phát triển
- Bổ sung voting nhiều khung hình để tăng độ vững quyết định mở khóa.
- Tự động chọn profile camera theo bối cảnh ánh sáng.
- Tăng cường bảo mật phần cứng và cơ chế fail-safe cho relay/lock.
- Xây dựng ứng dụng giám sát/điều khiển từ xa trên mobile.

---

## Kết luận
Chuyên đề IoT của FaceGuard đã đạt mục tiêu triển khai thực tế: nhận diện khuôn mặt qua ESP32-CAM, xử lý server ổn định và điều khiển khóa LY03 thành công. Kết quả cho thấy hướng AIoT chi phí thấp nhưng có độ tin cậy vận hành là khả thi trong bối cảnh ứng dụng nội bộ.
