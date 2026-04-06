Thư mục services chứa các module liên quan đến các dịch vụ của hệ thống FaceGuard. Dưới đây là mô tả chi tiết về các module trong thư mục này:
- Tệp __init__.py: Đây là tệp khởi tạo của package services, giúp Python nhận diện đây là một package và cho phép chúng ta import các module con bên trong.
- Tệp DataCollectionService.py: Chứa các hàm và lớp liên quan đến việc thu thập dữ liệu khuôn mặt từ người dùng, logic của nó đã được trình bày rõ trong tệp code ( Đây là phase 1 )
- Tệp FrameExtractionService.py: Chứa các hàm và lớp liên quan đến việc trích xuất khung hình từ video, logic của nó đã được trình bày rõ trong tệp code ( Đây là phase 2 )
- Tệp EmbeddingService.py: Chứa các hàm và lớp liên quan đến việc tạo embedding từ khuôn mặt, logic của nó đã được trình bày rõ trong tệp code ( Đây là phase 3 )
- Tệp SanitizerService.py: Chứa các hàm và lớp liên quan đến việc làm sạch dữ liệu khuôn mặt, logic của nó đã được trình bày rõ trong tệp code ( Đây là phase 4 )
- Tệp RecognitionService.py: Chứa các hàm và lớp liên quan đến việc nhận diện khuôn mặt, logic của nó đã được trình bày rõ trong tệp code ( Đây là phase 5 )
- Tệp FineTuneService.py: Chứa các hàm và lớp liên quan đến việc tinh chỉnh mô hình nhận diện khuôn mặt, logic của nó đã được trình bày rõ trong tệp code
- Tệp PackagingService.py: Chứa các hàm và lớp liên quan đến việc kiểm thử khả năng hoạt động của nó, bao gồm có 1 GUI dễ dùng, nó sử dụng dữ liệu vector lấy từ MinIO cùng với đó là mô hình glintr100.onnx để tiến hành so khớp theo thời gian thực logic của nó đã được trình bày rõ trong tệp code ( Đây là phase 6 )
- Tệp MasterWorkflowService.py: Chứa các hàm và lớp liên quan đến việc điều phối toàn bộ quy trình từ thu thập dữ liệu đến nhận diện khuôn mặt, logic của nó đã được trình bày rõ trong tệp code
- Tệp VideoQualityService.py: Chứa các hàm và lớp liên quan đến việc đánh giá chất lượng video, logic của nó đã được trình bày rõ trong tệp code ( Đây là phase 7 )
- Tệp DataCollectionGUI.py: Chứa các hàm và lớp liên quan đến việc tạo giao diện người dùng để thu thập dữ liệu khuôn mặt, logic của nó đã được trình bày rõ trong tệp code ( Đây là phần mở rộng của phase 1 )
- Tệp IoTService.py: Chứa các hàm và lớp liên quan đến việc tương tác với các thiết bị IoT

+ Trình bày sơ bộ về cơ chế hoạt động:
- Các tệp Service này sẽ là các tệp sử dụng các thành phần config, entities và adapters để thực hiện các chức năng cụ thể của từng phase trong quy trình nhận diện khuôn mặt. Ví dụ như với DataCollectionService sẽ sử dụng config data_collection.py, entities data_collection.py
- Mỗi tệp Service sẽ có các hàm và lớp riêng biệt để thực hiện các nhiệm vụ cụ thể, ví dụ như thu thập dữ liệu, trích xuất khung hình, tạo embedding, làm sạch dữ liệu, nhận diện khuôn mặt, tinh chỉnh mô hình, đóng gói mô hình và đánh giá chất lượng video.

Danh sách các bộ services, config, entities và adapters sẽ hoạt động cùng nhau để tạo thành một hệ thống hoàn chỉnh:
1. Phase 1: Data Collection
- config: data_collection.py, video_quality.py 
- entities: data_collection.py, video_quality.py
- adapters: Video_Adatper.py
- services: DataCollectionService.py, DataCollectionGUI.py, VideoQualityService.py

2. Phase 2: Frame Extraction
- config: frame_extraction.py
- entities: frame_extraction.py
- services: FrameExtractionService.py

3. Phase 3: Vector Embedding
- config: embedding.py
- entities: embedding.py
- services: EmbeddingService.py

4. Phase 4: Data Sanitization
- config: sanitizer.py
- entities: sanitizer.py
- services: SanitizerService.py

5. Phase 5: Face Recognition
- config: recognition.py
- entities: recognition.py
- services: RecognitionService.py

6. Phase 6: Realtime Recognition
- config: packager.py
- entities: packager.py
- services: PackagingService.py

Ngoài luồng cho IoT	thì sẽ có:
- config: IoT.py
- entities: IoT.py
- services: IoTService.py
 Đây là bộ 3 sẽ cùng nhau hoạt động để tạo thành 1 cổng xử lí dữ liệu được gửi về từ thiết bị IoT, sau đó phản hồi trở lại thiết bị IoT qua API 

 MasterWorkflowService.py sẽ là một service đặc biệt, nó sẽ điều phối toàn bộ quy trình từ thu thập dữ liệu đến nhận diện khuôn mặt, nó sẽ sử dụng tất cả các service khác để thực hiện các nhiệm vụ cụ thể của từng phase, từ đó tạo thành một hệ thống hoàn chỉnh và đồng bộ.

*** Bonus ***
Ngoài ra, còn có một số module khác như FineTuneService.py để tinh chỉnh mô hình nhận diện khuôn mặt và MasterWorkflowService.py để điều phối toàn bộ quy trình từ thu thập dữ liệu đến nhận diện khuôn mặt. Toàn bộ các Service này thuộc về một offline pipeline chuyên dùng để huấn luyện nhận diện người dùng và lưu dữ liệu của họ và hệ thống, với vector thì lưu vào MinIO còn metadata sẽ được lưu vào SQLite database