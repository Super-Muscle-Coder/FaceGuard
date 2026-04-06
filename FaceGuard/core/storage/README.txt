Thư mục storage chứa các module liên quan đến lưu trữ dữ liệu của hệ thống FaceGuard. Dưới đây là mô tả chi tiết về các module trong thư mục này: 
+ Gồm có 1 tệp SQLite.py, chứa các hàm để tương tác với cơ sở dữ liệu SQLite:
	from core.storage import SQLiteManager, get_sqlite_manager
+ 1 tệp có tên là face_recognition.db, chứa toàn bộ dữ liệu về metadata của người dùng
+ Một tệp __init__.py để đánh dấu đây là một package Python.
