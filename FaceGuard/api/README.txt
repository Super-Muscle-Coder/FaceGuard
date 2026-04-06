Thư mục api của FaceGuard chứa các module liên quan đến việc xây dựng và triển khai API cho hệ thống IoT, cụ thể ta sẽ sử dụng ESP32CAM và Docker:
- Tệp __init__.py: Đây là tệp khởi tạo của package api, giúp Python nhận diện đây là một package và cho phép chúng ta import các module con bên trong.
- Tệp IoTAPI.py: Chứa các hàm và lớp liên quan đến việc xây dựng API để tương tác với các thiết bị IoT

Tóm lại: thư mục API này sẽ định nghĩa các tệp dùng để xây dựng nên kênh giao tiếp giữa sever chủ ( chính là máy laptop của ta ) với thiết bị IoT ( chính là ESP32CAM ), khi ESP32CAM gửi dữ liệu, nó thông qua kênh này để gửi đi, và khi IoTService xử lí xong dữ liệu đó, nó cũng thông qua kênh này để phản hồi lại cho ESP32CAM