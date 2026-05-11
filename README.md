# 🎭 Hệ thống Nhận diện Cảm xúc Khuôn mặt (Facial Emotion Recognition - FER)


Hệ thống Deep Learning nhận diện 5 trạng thái cảm xúc chính từ ảnh khuôn mặt sử dụng công nghệ CNN.

---

## 1. 📌 Giới thiệu về dự án

Bài toán nhận diện cảm xúc khuôn mặt (Facial Emotion Recognition) phục vụ trong giao tiếp người-máy, y tế tâm lý và phân tích trải nghiệm người dùng. Dự án này nhằm tạo ra một hệ thống AI có khả năng phân loại 5 cảm xúc từ khuôn mặt: Angry, Happy, Neutral, Sad và Surprise.

---

## 2. 📊 Dữ liệu sử dụng (Dataset)

Dự án sử dụng bộ dữ liệu **FER-2013**.

* **Định dạng ảnh:** Grayscale, 48x48 pixels
* **Số lượng nhãn:** 7 lớp cảm xúc
* **Nhãn:** Angry, Happy, Neutral, Sad, Surprise 
* **Số lượng ảnh:** 30000 ảnh
---

## 3. ⚙️ Quy trình huấn luyện và xử lý dữ liệu

Dự án được tổ chức theo hai giai đoạn chính:

### Giai đoạn 1: Tiền xử lý & EDA
* **Tiền xử lý ảnh:** Chuyển ảnh về grayscale, chuẩn hóa kích thước và chuẩn hóa giá trị pixel
* **Data Augmentation:** Sử dụng ImageDataGenerator với xoay, dịch chuyển, zoom, lật ngang và thay đổi độ sáng
* **Cân bằng dữ liệu:** Áp dụng Class Weights hoặc oversampling khi cần
* **Khám phá dữ liệu:** Vẽ histograms, số lượng mỗi lớp và phân bố nhãn

### Giai đoạn 2: Huấn luyện & đánh giá mô hình
* **Chia dữ liệu:** Chia train/validation/test hợp lý
* **Xây dựng mô hình:** So sánh giữa **Custom CNN**, **MobileNetV2** và **VGG16**
* **Callback:** Sử dụng EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
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

Sau quá trình thử nghiệm nhiều mô hình, kết quả tốt nhất đạt được:

* **Accuracy trên tập Test:** 71.93%
* **Loss:** 0.7338

### So sánh mô hình

| Mô hình | Validation Accuracy | 
| :--- | :---: | :--- |
| Custom CNN | 57.5% | 
| MobileNetV2 | 29.8% | 
| VGG16 | 29.8% | 

**Nhận xét:**
* Mô hình hoạt động tốt với cảm xúc rõ rệt như Happy, Angry và Surprise
* Cần tối ưu thêm cho cảm xúc trung lập và buồn bã

### Biểu đồ kết quả

| Accuracy và Loss | Confusion Matrix |
| :---: | :---: |
| ![Accuracy và Loss](images/AccuracyAndLoss.png) | ![Confusion Matrix](images/MatrixConfusion.png) |

---

## 6. 🛠️ Công nghệ sử dụng

* **Ngôn ngữ:** Python 
* **Deep Learning:** TensorFlow, Keras
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




