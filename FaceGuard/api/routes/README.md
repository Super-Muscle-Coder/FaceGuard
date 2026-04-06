# api/routes

## Thư mục này dùng để làm gì
Định nghĩa các endpoint HTTP theo nhóm nghiệp vụ của runtime IoT.

## Bên trong chứa gì
- `health.py`
- `recognition.py`
- `__init__.py`

## Công dụng từng tệp mã
- `health.py`: endpoint theo dõi hệ thống (`/health`, `/metrics`, `/cameras`), reload runtime DB, stream ảnh debug, trạng thái runtime.
- `recognition.py`: xử lý request nhận diện ảnh và chuẩn hoá phản hồi JSON cho ESP32-CAM.
- `__init__.py`: export blueprint.

## Tương tác với thư mục nào
- Dùng `core/services/IoTService.py` cho nghiệp vụ nhận diện.
- Dùng `config/settings/*` cho threshold, api key, timeout, stream config.
