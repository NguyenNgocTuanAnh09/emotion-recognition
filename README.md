# 🎭 Hệ thống Nhận diện Cảm xúc Khuôn mặt (Facial Emotion Recognition - FER)


Hệ thống Deep Learning nhận diện 5 trạng thái cảm xúc chính từ ảnh khuôn mặt sử dụng công nghệ.

---

## 1. 📌 Giới thiệu về dự án

Bài toán nhận diện cảm xúc khuôn mặt (Facial Emotion Recognition) phục vụ trong giao tiếp người-máy, y tế tâm lý và phân tích trải nghiệm người dùng. Dự án này nhằm tạo ra một hệ thống AI có khả năng phân loại 5 cảm xúc từ khuôn mặt: Angry, Happy, Neutral, Sad và Surprise.

---

## 2. 📊 Dữ liệu sử dụng (Dataset)

Dự án sử dụng bộ dữ liệu **FER-2013**.

* **Định dạng ảnh:** Grayscale, 48x48 pixels
* **Số lượng nhãn:** 5 lớp cảm xúc
* **Nhãn:** Angry, Happy, Neutral, Sad, Surprise 
* **Số lượng ảnh:** 30000 ảnh
---

## 3. ⚙️ Quy trình huấn luyện và xử lý dữ liệu

Dự án được tổ chức theo hai giai đoạn chính:

### Giai đoạn 1: Tiền xử lý & EDA
* **Tiền xử lý ảnh:** Chuyển ảnh về grayscale, chuẩn hóa kích thước và chuẩn hóa giá trị pixel, chuẩn hoá nhãn
* **Data Augmentation:** Sử dụng ImageDataGenerator với xoay, dịch chuyển, zoom, lật ngang và thay đổi độ sáng
* **Cân bằng dữ liệu:** Áp dụng Class Weights.
* **Khám phá dữ liệu:** Vẽ histograms, số lượng mỗi lớp và phân bố nhãn, đặc tính dữ liệu

### Giai đoạn 2: Huấn luyện & đánh giá mô hình
* **Chia dữ liệu:** Chia train/validation/test hợp lý
* **So sánh mô hình:** So sánh giữa **Custom CNN**, **MobileNetV2** và **VGG16**
* **Xây dựng mô hình tốt nhất*** 
* **Callback:** Sử dụng EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
* **Huấn luyện mô hình tốt nhất** 
* **Đánh giá:** Confusion Matrix, Classification Report, Accuracy, Loss
* **Xuất mô hình:** Lưu file .h5 để dùng tiếp trong Web App

---

## 4. 📸 Ảnh Demo

| Ảnh Demo | Mô tả |
| :--- | :--- |
| ![Screenshot 1](images/DemoPicture1(Neutral).png) | Nhận diện ảnh cảm xúc bình thường |
| ![Screenshot 2](images/DemoPicture2(Surprise).png) | Nhận diện ảnh cảm xúc bất ngờ |
| ![Screenshot 3](images/DemoPicture3(Happy).png) | Nhận diện ảnh cảm xúc vui vẻ |
| ![Screenshot 4](images/DemoPicture4(Angry).png) | Nhận diện ảnh cảm xúc tức giận |
| ![Screenshot 5](images/DemoPicture5(Sad).png) | Nhận diện ảnh cảm xúc buồn bã |
| ![Screenshot 6](images/DemoFace1(Neutral).png) | Nhận diện mặt cảm xúc bình thường |
| ![Screenshot 7](images/DemoFace2(Sad).png) | Nhận diện mặt cảm xúc buồn bã |
| ![Screenshot 8](images/DemoFace3(Angry).png) | Nhận diện mặt cảm xúc tức giận |
| ![Screenshot 9](images/DemoFace4(Happy).png) | Nhận diện mặt cảm xúc vui vẻ |
| ![Screenshot 10](images/DemoFace5(Surprise).png) | Nhận diện mặt cảm xúc bất ngờ |

---

## 5. 🏆 Đánh giá và kết quả

Sau quá trình thử nghiệm và so sánh chéo, kiến trúc mạng tự xây dựng đã được lựa chọn để tiến hành huấn luyện chuyên sâu. Kết quả cuối cùng đạt được trên tập dữ liệu chưa từng thấy (Test Set) cực kỳ ấn tượng:

* **Độ chính xác (Accuracy) trên tập Test:** 71.93%
* **Mức độ sai số (Loss) trên tập Test:** 0.7338

### 📊 So sánh hiệu năng các mô hình (Giai đoạn tiền thử nghiệm)

| Mô hình | Validation Accuracy | Nhận xét chi tiết |
| :--- | :---: | :--- |
| **CNN** | **57.5%** | Tốc độ hội tụ nhanh, học đặc trưng tốt trên ảnh xám (Grayscale). Kiến trúc tối ưu, phù hợp nhất với bài toán. |
| MobileNetV2 | 29.8% | Hiệu suất kém (tương đương đoán mò). Khó trích xuất đặc trưng do kiến trúc gốc được thiết kế cho ảnh màu (RGB) 3 kênh. |
| VGG16 | 29.8% | Tương tự MobileNetV2, mô hình quá nặng, tiêu tốn nhiều tài nguyên nhưng lại xảy ra hiện tượng khó hội tụ trên tập dữ liệu này. |

*(Lưu ý: Sau khi chọn CNN làm mô hình nòng cốt và huấn luyện chuyên sâu với 50 Epochs cùng các kỹ thuật tối ưu như Callbacks và Class Weights, mô hình đã hội tụ và đạt đỉnh **71.93%** trên tập Test).*

---

### 📈 Trực quan hóa và Phân tích chuyên sâu

Để chứng minh độ tin cậy của mô hình, hệ thống đã kết xuất các biểu đồ phân tích kỹ thuật sau:

**1. Phân tích Biểu đồ học tập (Learning Curve)**
* **Sự hội tụ xuất sắc:** Trên cả hai biểu đồ Accuracy và Loss, đường Training và Validation bám sát nhau xuyên suốt 50 Epochs. Đường Val Accuracy luôn giữ vững đà tăng và tiệm cận với Train Accuracy.
* **Kiểm soát Overfitting:** Đồ thị cho thấy hệ thống **không hề bị Overfitting**. Việc kết hợp hoàn hảo giữa các lớp `Dropout`, `BatchNormalization` và kỹ thuật tăng cường ảnh (`Data Augmentation`) đã phát huy tác dụng tối đa, giúp mô hình tổng quát hóa cực tốt trên dữ liệu mới thay vì chỉ "học vẹt".

**2. Phân tích Ma trận nhầm lẫn (Confusion Matrix)**
Dựa vào ma trận dự đoán chéo trên tập Test, có thể rút ra đặc tính nhận diện cảm xúc của AI như sau:
* **Nhóm xuất sắc:** Cảm xúc **Happy (Vui vẻ)** có độ phân loại chính xác cao nhất (1442 mẫu đoán đúng), tiếp theo là **Surprise (Bất ngờ)**. Các biểu cảm này có sự thay đổi cơ mặt rõ rệt (cười lớn, mở to mắt) nên AI bóc tách đặc trưng rất dễ dàng.
* **Nhóm ổn định:** Cảm xúc **Angry (Tức giận)** và **Neutral (Bình thường)** được phân loại tương đối tốt, tỷ lệ nhận diện đúng cao dù vẫn còn một vài mẫu bị nhầm lẫn chéo với nhau.
* **Nhóm thách thức:** Cảm xúc **Sad (Buồn bã)** là điểm yếu nhất của hệ thống. Có đến 379 bức ảnh "Buồn bã" bị AI nhận diện nhầm thành "Bình thường". Lý do xuất phát từ thực tế: biểu cảm buồn thường rất vi tế (chỉ hơi chùng khóe miệng hoặc ánh mắt). Ranh giới giữa một khuôn mặt buồn nhẹ và khuôn mặt vô cảm (Neutral) là rất mong manh, ngay cả với mắt người cũng khó có thể phân biệt chính xác 100%.

---

## 6. 🛠️ Công nghệ sử dụng

* **Ngôn ngữ:** Python 
* **Deep Learning:** TensorFlow, Keras, Skikit-learn
* **Xử lý dữ liệu:** NumPy, Pandas
* **Xử lý ảnh:** OpenCV, Pillow
* **Trực quan hóa:** Matplotlib, Seaborn
* **Web App:** Flask

---

## 7. 💻 Cài đặt

### Clone repository:
```bash
git clone https://github.com/NguyenNgocTuanAnh09/emotion-recognition.git
cd emotion-recognition
```


## 8. 🚀 Cách chạy dự án

### Chạy Web App:
```bash
cd app
python web.py
```

---

---

## 9. 👨‍💻 Tác giả

**Nguyễn Ngọc Tuấn Anh**

* **Chuyên ngành:** Hệ thống Thông tin 
* **Đơn vị:** Trường Đại học Thủy lợi (TLU) 
* **Liên hệ:** ngoctuananh09@gmail.com

---




