# Tài liệu hệ thống mã nguồn ESP32-CAM (FaceGuard)

> Tệp nguồn được mô tả: `scripts/RawArduinoCode_ESP32CAM.txt`

## 1) Tổng quan chức năng chính
Firmware hiện tại được thiết kế theo mô hình **Edge client** cho hệ thống AIoT FaceGuard, gồm các khối chức năng lớn:

1. **Cấu hình hệ thống**
   - WiFi, API endpoint, API key, camera ID.
   - Cấu hình chất lượng ảnh và profile cảm biến.
   - Cấu hình relay mở khóa.

2. **Kết nối mạng ổn định**
   - Kết nối WiFi theo chế độ STA.
   - Tự retry khi quá timeout kết nối.

3. **Khởi tạo camera + tự phục hồi camera**
   - Init camera với cấu hình JPEG tối ưu cho ESP32-CAM.
   - Retry init nhiều lần nếu lỗi.
   - Power-cycle cảm biến khi cần.

4. **Chụp ảnh và gửi server nhận diện**
   - Chụp JPEG theo chu kỳ.
   - Gửi lên `/api/v1/recognize` bằng multipart/form-data.
   - Truyền dữ liệu theo chunk để tránh ghi thiếu.

5. **Nhận phản hồi và điều khiển khóa**
   - Parse JSON trả về (`allowed`, `identity`, `confidence`, `reason`).
   - Nếu `allowed=true` thì kích relay mở khóa trong khoảng thời gian cấu hình.

6. **Vòng lặp runtime thông minh**
   - Reconnect WiFi khi mất mạng.
   - Warm-up frame trước khi gửi.
   - Smart retry khi kết quả yếu/no-face.

---

## 2) Đặc tả chi tiết theo module

## 2.1 Khối cấu hình
Các hằng số ở đầu file đóng vai trò “control panel”:

- **Mạng/API:** `WIFI_SSID`, `WIFI_PASS`, `API_HOST`, `API_PORT`, `API_PATH`, `API_KEY`, `API_KEY_HEADER`, `CAMERA_ID`.
- **Chu kỳ/timeout:** `CAPTURE_INTERVAL_MS`, `HTTP_TIMEOUT_MS`.
- **Ảnh/cảm biến:** `CAMERA_FRAME_SIZE`, `CAMERA_JPEG_QUALITY`, `SENSOR_*`, `DISCARD_WARMUP_FRAMES`.
- **Retry thông minh:** `ENABLE_SMART_RETRY_CAPTURE`, `RETRY_IF_CONF_BELOW`, `RETRY_*`.
- **Relay:** `ENABLE_RELAY`, `RELAY_PIN`, `RELAY_OPEN_MS`.

Ý nghĩa: dễ tinh chỉnh thực địa mà không phải sửa logic lõi.

### Bảng tham số quan trọng nên nắm

| Nhóm | Biến | Ý nghĩa | Khuyến nghị vận hành |
|---|---|---|---|
| Network | `HTTP_TIMEOUT_MS` | Timeout TCP/HTTP | 15s–25s cho WiFi dân dụng |
| Capture | `CAPTURE_INTERVAL_MS` | Chu kỳ gửi ảnh | 2s–4s để cân bằng tải và độ phản hồi |
| Image | `CAMERA_FRAME_SIZE` | Độ phân giải JPEG | CIF là điểm cân bằng tốt cho ESP32-CAM |
| Image | `CAMERA_JPEG_QUALITY` | Chất lượng JPEG | 10–14 thường ổn (file vừa đủ chi tiết) |
| Retry | `RETRY_IF_CONF_BELOW` | Ngưỡng kích hoạt retry | 0.40–0.50 tùy model runtime |
| Relay | `RELAY_OPEN_MS` | Thời gian mở khóa | 800–1500ms tùy loại khóa |

---

## 2.2 Khối WiFi – `connectWiFi()`
Chức năng:
- Đặt `WIFI_STA`, tắt sleep để ổn định truyền.
- Nếu chưa kết nối: `WiFi.begin(...)` và chờ.
- Quá 20 giây sẽ tự `disconnect` và thử lại.

Lợi ích: thiết bị có khả năng tự phục hồi mạng, giảm thao tác thủ công.

## 2.3 Khối camera

### a) `powerCycleCameraSensor()`
- Tắt/bật cảm biến qua chân PWDN.
- Dùng khi camera init lỗi hoặc trạng thái không ổn định.

### b) `initCameraOnce()`
- Cấu hình pin camera, XCLK, format JPEG.
- Ưu tiên CIF + JPEG quality 12.
- Nếu có PSRAM: tăng `fb_count`, `grab_mode` để ổn định capture.
- Set thông số cảm biến (AE/AEC/AGC/AWB/contrast/saturation...).

### c) `initCameraWithRetry(maxTry=6)`
- Lặp nhiều lần: deinit -> power cycle -> init lại.
- Giúp firmware bền hơn trong điều kiện boot thực tế.

### d) Trình tự init camera thực tế
1. Gọi `esp_camera_deinit()` để reset trạng thái driver cũ.
2. Power-cycle cảm biến qua `PWDN`.
3. Init lại bằng `esp_camera_init(&config)`.
4. Nếu thành công thì set profile cảm biến.
5. Nếu thất bại thì delay và thử lại.

Ý nghĩa: xử lý các lỗi boot không ổn định thường gặp trên ESP32-CAM (đặc biệt khi nguồn yếu hoặc camera bus nhạy).

## 2.4 Khối relay – `openDoorRelay()`
- Nếu `ENABLE_RELAY=false`: bỏ qua tác động phần cứng.
- Nếu bật relay:
  - `digitalWrite(HIGH)` để kích.
  - giữ trong `RELAY_OPEN_MS`.
  - `digitalWrite(LOW)` để nhả.

Mục tiêu: mở khóa có thời lượng cố định, tránh giữ relay liên tục.

## 2.5 Khối HTTP + parse response

### a) `readHttpResponse(...)` + `extractBody(...)`
- Đọc response thô từ socket.
- Tách status code và body JSON.

### b) `sendFrameMultipart(camera_fb_t *fb)`
Luồng xử lý:
1. Kiểm tra frame hợp lệ và WiFi connected.
2. Tạo TCP tới server.
3. Build multipart gồm `camera_id`, `sequence_number`, `metadata`, `image`.
4. Gửi file JPEG bằng chunk loop.
5. Đọc phản hồi, parse JSON.
6. Cập nhật trạng thái `g_last...`.
7. Nếu `allowed=true` thì mở relay.

### c) Đặc tả gói multipart gửi lên server
Firmware gửi các field chính:
- `camera_id`: định danh camera.
- `sequence_number`: số thứ tự frame tăng dần.
- `metadata`: JSON tối thiểu gồm `wifi_rssi`.
- `image`: file JPEG (`capture.jpg`).

Lợi ích:
- Server theo dõi frame loss/out-of-order.
- Có dữ liệu RSSI để đối chiếu lỗi mạng khi confidence giảm.

### d) Đặc tả parse phản hồi JSON
Firmware kỳ vọng body có các khóa:
- `allowed` (bool)
- `identity` (string)
- `confidence` (float)
- `reason` (string)

Sau parse:
- lưu vào `g_lastAllowed`, `g_lastConfidence`, `g_lastHadFace`
- dùng các biến này để quyết định smart retry ở vòng loop tiếp theo.

## 2.6 Vòng đời runtime – `setup()` và `loop()`

### `setup()`
- Init serial log.
- Init relay output (nếu bật).
- Init camera có retry.
- Kết nối WiFi.

### `loop()`
- Giữ WiFi luôn online.
- Nếu camera mất sẵn sàng thì init lại.
- Chụp theo chu kỳ.
- Warm-up trước frame chính.
- Gửi frame chính.
- Nếu kết quả yếu thì kích hoạt smart retry với profile cảm biến thay thế.

### Máy trạng thái logic của `loop()`
Có thể hiểu `loop()` theo state machine đơn giản:

1. **CHECK_WIFI**: mất mạng -> reconnect.
2. **CHECK_CAMERA**: camera không sẵn sàng -> init lại.
3. **WAIT_INTERVAL**: chưa tới chu kỳ chụp -> chờ ngắn.
4. **WARMUP**: bỏ qua vài frame đầu để ổn định AE/AWB.
5. **CAPTURE_SEND**: chụp frame chính và gửi server.
6. **SMART_RETRY (optional)**: nếu kết quả yếu -> đổi profile -> chụp/gửi lần 2.
7. Quay về bước 1.

Thiết kế này giúp firmware không bị “kẹt” vào một nhánh lỗi cố định.

---

## 2.7 Đặc tả biến trạng thái runtime (`g_*`)

| Biến | Vai trò |
|---|---|
| `g_sequence` | Đếm thứ tự request gửi lên server |
| `g_lastCaptureMs` | Mốc thời gian lần chụp gần nhất |
| `g_cameraReady` | Cờ camera đã init thành công |
| `g_captureFailStreak` | Đếm số lần capture fail liên tiếp |
| `g_lastAllowed` | Kết quả allow gần nhất từ server |
| `g_lastConfidence` | Confidence gần nhất |
| `g_lastHadFace` | Frame gần nhất có/không có mặt |

Các biến này là nền tảng cho cơ chế tự phục hồi và retry thông minh.

---

## 2.8 Đặc tả xử lý lỗi và fallback

### Nhóm lỗi mạng
- Dấu hiệu: `[HTTP] Connect failed`, `[LOOP] Send failed`.
- Xử lý: bỏ qua frame hiện tại, vòng sau gửi lại.

### Nhóm lỗi ghi dữ liệu
- Dấu hiệu: `Write timeout` hoặc `Write image incomplete`.
- Xử lý: hủy kết nối hiện tại (`client.stop()`), vòng sau tạo kết nối mới.

### Nhóm lỗi camera runtime
- Dấu hiệu: `Capture failed (n)`.
- Xử lý: nếu fail streak >= 3 thì init lại camera toàn phần.

### Nhóm lỗi parse JSON
- Dấu hiệu: `[JSON] Parse failed`.
- Xử lý: coi lần gửi thất bại logic, không kích relay.

---

## 2.9 Đặc tả bảo mật tối thiểu ở firmware

- Sử dụng API key header (`X-FaceGuard-Api-Key`) cho mọi request nhận diện.
- Không hardcode thông tin nhạy cảm theo môi trường production trong bản public.
- Nên tách biến cấu hình WiFi/API ra file cấu hình riêng khi phát hành nhiều site.

---

## 2.10 Đặc tả relay/khóa trong thực địa

- Firmware đang giả định relay active HIGH.
- Nếu dùng module active LOW, chỉ cần đảo mức trong `openDoorRelay()`.
- `RELAY_PIN = 13` được chọn để ổn định boot hơn một số pin strap.
- `RELAY_OPEN_MS` cần hiệu chỉnh theo quán tính cơ khí khóa LY03 tại site thực tế.

---

## 2.11 Checklist tuning nhanh tại hiện trường

1. Kiểm tra nguồn 5V đủ dòng (>=1A, khuyến nghị 2A).
2. Kiểm tra WiFi RSSI trong metadata (tránh vùng sóng quá yếu).
3. Quan sát stream `/camera/<id>/viewer` để chỉnh profile sáng/tối.
4. Tune `RETRY_IF_CONF_BELOW` theo threshold server thực tế.
5. Tune `RELAY_OPEN_MS` để khóa mở ổn định nhưng không giữ relay quá lâu.

---

## 3) Các kỹ thuật thông minh đã dùng để giảm hạn chế ESP32-CAM

## 3.1 Chunked write loop khi gửi JPEG
- Thay vì gửi toàn bộ ảnh một lần, firmware gửi theo khối 1024 bytes.
- Có timeout + kiểm tra ghi thiếu.
- Giảm lỗi `partial write`, tăng độ ổn định HTTP upload.

## 3.2 Warm-up frame trước capture chính
- Bỏ qua vài frame đầu để AE/AWB ổn định.
- Giảm frame “sốc sáng”, giúp nhận diện ổn định hơn.

## 3.3 Smart retry profile
- Khi no-face hoặc confidence thấp, firmware tự:
  - đổi profile cảm biến,
  - warm-up,
  - gửi thêm 1 frame retry,
  - trả về profile mặc định.

Tác dụng: tăng khả năng nhận diện trong điều kiện ánh sáng biến động.

## 3.4 Auto-reconnect WiFi
- Mất mạng sẽ tự reconnect, không cần reset tay.
- Hữu ích khi triển khai dài hạn trong môi trường WiFi dân dụng.

## 3.5 Camera init retry + power-cycle sensor
- Khi camera lỗi init, firmware có cơ chế tự hồi phục.
- Giảm downtime do lỗi khởi tạo cảm biến.

## 3.6 PSRAM-aware tuning
- Có PSRAM thì tăng buffer và chế độ lấy frame hợp lý.
- Tăng độ mượt và giảm lỗi capture.

---

## 4) Luồng hoạt động end-to-end
1. ESP32-CAM boot, init camera, kết nối WiFi.
2. Theo chu kỳ 3 giây, chụp ảnh và gửi server.
3. Server trả JSON quyết định.
4. Nếu `allowed=true`, ESP32 kích relay mở khóa.
5. Nếu kết quả yếu, firmware tự retry 1 lần với profile khác.
6. Hệ thống tiếp tục chạy vòng lặp liên tục.

---

## 5) Ghi chú vận hành
- Hiện firmware đang để `ENABLE_RELAY = true` và `RELAY_PIN = 13`.
- Nếu module relay là active LOW thì cần đảo mức HIGH/LOW trong `openDoorRelay()`.
- Nên cấp nguồn 5V ổn định (adapter riêng) cho ESP32-CAM + relay logic để tránh sụt áp khi WiFi/camera hoạt động.