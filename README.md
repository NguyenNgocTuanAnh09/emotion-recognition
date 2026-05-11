#  H? th?ng Nh?n di?n C?m xúc Khuôn m?t (Facial Emotion Recognition - FER)

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-3.x-D00000.svg?logo=keras)

H? th?ng Deep Learning nh?n di?n 5 tr?ng thái c?m xúc chính t? ?nh khuôn m?t. D? án xây d?ng t? ti?n x? lý ?nh, hu?n luy?n mô hình, dánh giá hi?u nang d?n giao di?n Web tuong tác.

---

## 1.  Gi?i thi?u v? d? án
Bài toán nh?n di?n c?m xúc khuôn m?t (Facial Emotion Recognition) ph?c v? trong giao ti?p ngu?i-máy, y t? tâm lý và phân tích tr?i nghi?m ngu?i dùng. D? án này nh?m t?o ra m?t h? th?ng AI có kh? nang phân lo?i 5 c?m xúc t? khuôn m?t: Angry, Happy, Neutral, Sad và Surprise.

M?c tiêu chính:
- Xây d?ng pipeline ti?n x? lý ?nh và tang cu?ng d? li?u.
- So sánh các ki?n trúc CNN và Transfer Learning.
- Xu?t mô hình t?t nh?t d? ch?y realtime trên Web App.

---

## 2.  D? li?u s? d?ng (Dataset)
D? án s? d?ng b? d? li?u **FER-2013** ho?c b? d? li?u tuong t? g?m ?nh m?t xám kích thu?c 48x48.

* **Ð?nh d?ng ?nh:** Grayscale 48x48 pixels.
* **S? lu?ng nhãn:** 5 l?p c?m xúc.
* **Nhãn:** Angry, Happy, Neutral, Sad, Surprise.

 D? li?u ?nh có tính nhi?u cao do bi?u c?m m?, góc ch?p khác nhau và ánh sáng thay d?i, nên c?n x? lý k? tru?c khi hu?n luy?n.

---

## 3.  Quy trình hu?n luy?n và x? lý d? li?u
D? án du?c t? ch?c theo hai giai do?n chính:

### Giai do?n 1: Ti?n x? lý & EDA
* **Ti?n x? lý ?nh:** Chuy?n ?nh v? grayscale, chu?n hóa kích thu?c và chu?n hóa giá tr? pixel.
* **Data Augmentation:** S? d?ng ImageDataGenerator v?i xoay, d?ch chuy?n, zoom, l?t ngang và thay d?i d? sáng.
* **Cân b?ng d? li?u:** Áp d?ng Class Weights ho?c oversampling khi c?n.
* **Khám phá d? li?u:** V? histograms, s? lu?ng m?i l?p và phân b? nhãn.

### Giai do?n 2: Hu?n luy?n & dánh giá mô hình
* **Chia d? li?u:** Chia train/validation/test h?p lý.
* **Xây d?ng mô hình:** So sánh gi?a **Custom CNN**, **MobileNetV2** và **VGG16**.
* **Callback:** S? d?ng EarlyStopping, ReduceLROnPlateau, ModelCheckpoint.
* **Ðánh giá:** Confusion Matrix, Classification Report, Accuracy, Loss.
* **Xu?t mô hình:** Luu file .h5 d? dùng ti?p trong Web App.

---

## 4.  Ðóng gói & tri?n khai
Mô hình chi?n th?ng du?c luu du?i d?ng file .h5 trong thu m?c models/.

* models/final_fer_model.h5  mô hình dã hu?n luy?n s?n.

Uu di?m:
* D? tích h?p vào Web App Flask.
* D? doán realtime t? ?nh upload ho?c webcam.

---

## 5.  Ðánh giá & k?t qu?
Sau quá trình th? nghi?m nhi?u mô hình, k?t qu? t?t nh?t d?t du?c:

* **Accuracy trên t?p Test:** 71.93%
* **Loss:** 0.7338

### So sánh mô hình
| Mô hình | Validation Accuracy | Ghi chú |
| :--- | :---: | :--- |
| Custom CNN | [Ði?n %] | Nh?, phù h?p cho Web |
| MobileNetV2 | [Ði?n %] | Transfer Learning |
| VGG16 | [Ði?n %] | Mô hình sâu hon |

### Nh?n xét
* Mô hình ho?t d?ng t?t v?i c?m xúc rõ r?t nhu Happy và Surprise.
* C?n t?i uu thêm cho c?m xúc trung tính và bu?n bã.

---

## 6.  Công ngh? s? d?ng
* **Ngôn ng?:** Python 3.10+
* **Deep Learning:** TensorFlow, Keras
* **X? lý d? li?u:** NumPy, Pandas
* **X? lý ?nh:** OpenCV, Pillow
* **Tr?c quan hóa:** Matplotlib, Seaborn
* **Web App:** Flask

---

## 7.  Cài d?t

1. Clone repository:
   `ash
   git clone https://github.com/your-username/emotion-recognition.git
   cd emotion-recognition
   `
2. T?o môi tru?ng ?o:
   `ash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS / Linux
   python -m venv venv
   source venv/bin/activate
   `
3. Cài dependencies:
   `ash
   pip install --upgrade pip
   pip install -r requirements.txt
   `

---

## 8.  Cách ch?y d? án

### A. Ch?y Web App
`ash
cd app
python web.py
`
M? trình duy?t t?i: http://localhost:5000

### B. Ch?y notebook hu?n luy?n
`ash
jupyter notebook notebooks/Emotion_Recognition.ipynb
`

---

## 9.  C?u trúc d? án

`
emotion-recognition/
 README.md
 requirements.txt
 app/
    web.py
    templates/
    static/
 models/
    final_fer_model.h5
 notebooks/
    Emotion_Recognition.ipynb
 images/
     learning_curve.png
     confusion_matrix.png
     test_demo.png
`

---

## 10.  Tác gi?
**Nguy?n Ng?c Tu?n Anh**

*Chuyên ngành:* H? th?ng Thông tin (Mã ngành: 64HTTT4)

*Ðon v?:* Tru?ng Ð?i h?c Th?y l?i (TLU) - Hà N?i

*Liên h?:* [your-email@example.com]

---

## 11.  Ghi chú
* C?p nh?t l?i du?ng d?n Web App n?u s? d?ng framework khác.
* Thêm s? li?u chính xác vào b?ng so sánh khi có k?t qu? th?c t?.
