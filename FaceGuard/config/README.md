# config

## Thư mục này dùng để làm gì
Chứa toàn bộ tham số cấu hình cho các phase và runtime của FaceGuard.

## Bên trong chứa gì
- `settings/`: module cấu hình theo domain (IoT, training, paths, storage, ...).

## Công dụng tệp mã trong thư mục
- Truy cập cấu hình thống nhất thông qua `config/settings/__init__.py`.

## Tương tác với thư mục nào
- Được import rộng rãi trong `core/services`, `core/adapters`, `api` và script vận hành.
