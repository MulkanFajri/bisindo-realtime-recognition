<div align="center">

# 🤟 BISINDO Realtime Recognition

### Pengenalan Bahasa Isyarat Indonesia (BISINDO) Secara Realtime Menggunakan Machine Learning

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Flask](https://img.shields.io/badge/Flask-WebApp-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-orange)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-MLP%20%7C%20RandomForest-red)

</div>

---

## 📖 Tentang Proyek

Proyek ini merupakan sistem pengenalan Bahasa Isyarat Indonesia (BISINDO) secara realtime menggunakan teknologi Computer Vision dan Machine Learning.

Sistem memanfaatkan **MediaPipe Hands** untuk mendeteksi posisi tangan, kemudian melakukan ekstraksi landmark dan mengklasifikasikan gesture menggunakan model Machine Learning.

### Fitur Utama

✅ Pengenalan Huruf A-Z

✅ Pengenalan Angka 0-9

✅ Pengenalan Kata BISINDO

✅ Deteksi Landmark Tangan Realtime

✅ Website Interaktif Berbasis Flask

✅ Menampilkan Confidence Prediksi

---

## 🖼️ Tampilan Sistem

### Halaman Utama

![Home](screenshots/home.png)

### Deteksi Huruf dan Angka

![Huruf](screenshots/huruf.png)

### Deteksi Kata BISINDO

![Kata](screenshots/kata.png)

---

## ⚙️ Teknologi yang Digunakan

| Teknologi | Fungsi |
|------------|----------|
| Python | Bahasa Pemrograman |
| Flask | Backend Web |
| OpenCV | Pengolahan Citra |
| MediaPipe Hands | Deteksi Landmark Tangan |
| NumPy | Operasi Numerik |
| Pandas | Pengolahan Dataset |
| Scikit-Learn | Machine Learning |
| MLP | Klasifikasi Huruf & Angka |
| Random Forest | Klasifikasi Kata |

---

## 🏗️ Arsitektur Sistem

```text
Webcam
   │
   ▼
MediaPipe Hands
   │
   ▼
Ekstraksi Landmark
   │
   ▼
126 Fitur Landmark
   │
   ▼
Machine Learning
   │
   ├── MLP
   │     └── Huruf & Angka
   │
   └── Random Forest
         └── Kata BISINDO
   │
   ▼
Flask Web Application
   │
   ▼
Hasil Prediksi Realtime
```

---

## 📂 Dataset

### Huruf dan Angka

Dataset huruf dan angka diperoleh dari Kaggle kemudian diproses menggunakan MediaPipe Hands untuk menghasilkan landmark tangan.

Jumlah kelas:

- 26 Huruf (A-Z)
- 10 Angka (0-9)

Total:

```text
36 Kelas
```

---

### Kata BISINDO

Dataset kata dikumpulkan secara mandiri menggunakan webcam.

Kelas yang digunakan:

```text
Mendengar
Tersenyum
Ragu
Damai
Semoga Beruntung
```

Masing-masing kelas terdiri dari sekitar:

```text
100 Gambar
```

---

## 📊 Ekstraksi Fitur

MediaPipe Hands menghasilkan:

```text
21 Landmark
```

Setiap landmark memiliki:

```text
x
y
z
```

Sehingga:

```text
21 × 3 = 63 fitur
```

Karena sistem mendukung dua tangan:

```text
63 + 63 = 126 fitur
```

Landmark tersebut disimpan ke file CSV sebelum proses training dilakukan.

---

## 🧠 Model Machine Learning

### Huruf dan Angka

Metode:

```text
MLP (Multi Layer Perceptron)
```

Akurasi:

```text
97.9%
```

Alasan penggunaan:

- Cepat
- Ringan
- Cocok untuk data landmark
- Akurasi tinggi

---

### Kata BISINDO

Metode:

```text
Random Forest Classifier
```

Akurasi:

```text
98%
```

Alasan penggunaan:

- Stabil
- Tidak mudah overfitting
- Cocok untuk dataset ukuran menengah

---

## 📁 Struktur Folder

```text
PROJECT_BISINDO

├── models
│   ├── bisindo_mlp.pkl
│   ├── kata_model.pkl
│   └── label_encoder.pkl
│
├── templates
│   └── index.html
│
├── app.py
├── collect_kata.py
├── extract_landmarks.py
├── extract_landmarks_kata.py
├── train_mlp.py
├── train_kata.py
├── realtime_mlp.py
│
├── requirements.txt
└── README.md
```

---

## 🚀 Cara Menjalankan

Clone repository:

```bash
git clone https://github.com/MulkanFajri/bisindo-realtime-recognition.git
```

Masuk ke folder project:

```bash
cd bisindo-realtime-recognition
```

Install library:

```bash
pip install -r requirements.txt
```

Jalankan aplikasi:

```bash
python app.py
```

Buka browser:

```text
http://127.0.0.1:5000
```

---

## 📈 Hasil Pengujian

| Model | Akurasi |
|---------|---------|
| MLP (Huruf & Angka) | 97.9% |
| Random Forest (Kata) | 98% |

---

## 👨‍💻 Pengembang

**Mulkan Fajri**

Teknik Informatika

Politeknik Negeri Lhokseumawe

---

<div align="center">

### ⭐ Jika proyek ini bermanfaat, berikan Star pada repository ini.

</div>