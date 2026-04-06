# training_plots

## Thư mục này dùng để làm gì
Lưu biểu đồ và báo cáo tóm tắt kết quả huấn luyện fine-tune.

## Bên trong chứa gì
- `fine_tune_loss.png`, `fine_tune_accuracy.png`
- `fine_tune_summary_<person>.json`

## Công dụng từng tệp mã
- Các file PNG: trực quan quá trình hội tụ loss/accuracy.
- JSON summary: lưu config train, class distribution, metric theo epoch, checkpoint info.

## Tương tác với thư mục nào
- Được ghi bởi `core/services/FineTuneService.py`.
- Được dùng trong `documents/` để làm báo cáo kỹ thuật/học thuật.
