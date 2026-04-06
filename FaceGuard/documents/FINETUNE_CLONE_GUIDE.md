# Hướng dẫn chạy Fine-tune sau khi clone FaceGuard

Tài liệu này dành cho người mới clone code, cần chạy được pipeline fine-tune từ đầu đến cuối.

---

## 1) Fine-tune trong FaceGuard là gì?
Fine-tune của FaceGuard là pipeline huấn luyện **head classifier** dựa trên embedding ArcFace đã có sẵn.  
Mục tiêu: thêm/cập nhật người dùng mới với chi phí thấp hơn so với retrain toàn bộ backbone.

Pipeline chính:
1. Phase 1: Data Collection (GUI quay 3 video)
2. Phase 2: Frame Extraction
3. Phase 3: Frame Sanitizer
4. Phase 4: Fine-tune head + sync runtime

Entry point: `FineTuneEntry.py` - Nếu mục đích là kiểm thử khả năng Fine-tune, bạn chỉ cần chạy tệp này là đủ (đảm bảo đã có model ONNX và camera hoạt động).

---

## 2) Yêu cầu trước khi chạy

## 2.1 Yêu cầu phần mềm
- Python 3.11+
- Git
- (Khuyến nghị) venv
- Docker Desktop (nếu chạy theo mode container)

## 2.2 Yêu cầu model
FaceGuard cần 2 model ONNX:
- `scrfd_10g_bnkps.onnx`
- `glintr100.onnx`

Theo chuẩn dự án, model được đặt tại:
- `E:\Project\FaceGuard\FaceGuard\FaceGuard\scripts\models`

Nếu chưa có model, hãy chạy tệp `download_models.py` trong thư mục `scripts/` để tải model.

## 2.3 Yêu cầu phần cứng tối thiểu
- CPU phổ thông (vẫn fine-tune được, nhưng sẽ chậm hơn GPU)
- RAM >= 8GB
- Camera laptop/USB camera cho Phase 1 GUI

---

## 3) Cách chạy nhanh (Local Python)

### Bước 1: clone và vào thư mục
```bash
git clone https://github.com/Super-Muscle-Coder/FaceGuard.git
cd FaceGuard/FaceGuard
```

### Bước 2: tạo môi trường và cài thư viện
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r docker/requirements.training.txt
```

### Bước 3: chạy fine-tune pipeline
```bash
python FineTuneEntry.py
```

Chương trình sẽ hỏi tên người dùng nếu chưa truyền `--person`.

Ví dụ chạy có tham số:
```bash
python FineTuneEntry.py --person "Nghi" --target-frames 100 --auto-reject-threshold 0.6
```

---

## 4) Cách chạy bằng Docker (khuyến nghị cho người dùng mới)

### Bước 1: kéo stack
```bash
docker compose -f docker/docker-compose.yaml pull
docker compose -f docker/docker-compose.yaml up -d
```

### Bước 2: chạy fine-tune trong container training
```bash
docker exec -it faceguard-training python FineTuneEntry.py
```

> Container `faceguard-training` mặc định ở trạng thái chờ (`sleep infinity`), bạn chủ động gọi lệnh training bằng `docker exec`.

---

## 5) Dữ liệu input/output sẽ nằm ở đâu?

Trong quá trình chạy, hệ thống tạo dữ liệu tại:
- `data/temp/<person>/...` (video, frame, sanitized frame)
- `database/fine_tune_head.pt` (checkpoint head)
- `training_plots/fine_tune_*.png` (biểu đồ)
- `training_plots/fine_tune_summary_<person>.json` (tóm tắt run)

Nếu chọn deploy sau training:
- Đồng bộ vector runtime lên MinIO
- Cập nhật metadata SQLite

---

## 6) Cách kiểm tra chạy thành công

Dấu hiệu thành công:
- Có log `Fine-tune complete` / `best_val_acc=...`
- Có file `database/fine_tune_head.pt`
- Có `training_plots/fine_tune_loss.png` và `fine_tune_accuracy.png`
- Có `fine_tune_summary_<person>.json`

---

## 7) Lỗi thường gặp và cách xử lý

## 7.1 Không mở được camera ở Phase 1
- Đóng các app đang giữ camera (Teams/Zoom/Meet)
- Kiểm tra quyền camera trong Windows Privacy
- Thử camera index khác (nếu có nhiều camera)

## 7.2 Báo thiếu model ONNX
- Kiểm tra đúng đường dẫn model chuẩn của dự án
- Đảm bảo đủ 2 file SCRFD + ArcFace

## 7.3 GUI không hiện khi chạy Docker
- GUI trong container phụ thuộc môi trường host forwarding
- Nếu host không hỗ trợ, chạy fine-tune local Python (mục 3)

## 7.4 Không sync được MinIO
- Kiểm tra `minio` container đã chạy
- Kiểm tra credential trong compose/env

---

## 8) Luồng đề xuất cho người mới
1. Chạy local lần đầu để xác nhận camera + model + pipeline.
2. Khi ổn định, chuyển sang Docker để vận hành chuẩn hóa.
3. Sau mỗi lần fine-tune, kiểm tra summary + biểu đồ trước khi deploy.
4. Chỉ deploy khi quality dữ liệu và metrics đạt mức mong muốn.

---

## 9) Tài liệu liên quan
- `documents/PROJECT_ONBOARDING_GUIDE.md`
- `docker/CUSTOMER_QUICKSTART.md`
- `documents/esp32cam_firmware_guide.md`
