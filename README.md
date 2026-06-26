<div align="center">

# 🤟 BISINDO Realtime Recognition

### Pengenalan Bahasa Isyarat Indonesia (BISINDO) Secara Realtime Menggunakan Machine Learning (MLP & Random Forest)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Flask](https://img.shields.io/badge/Flask-WebApp-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-orange)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-MLP%20%7C%20RandomForest-red)

<img src="https://img.shields.io/badge/Akurasi%20Huruf-97.9%25-success">
<img src="https://img.shields.io/badge/Akurasi%20Kata-98%25-success">
<img src="https://img.shields.io/badge/Kelas-41-blue">

</div>

---

## 📖 Tentang Proyek

Proyek ini merupakan sistem pengenalan Bahasa Isyarat Indonesia (BISINDO) secara realtime menggunakan teknologi Computer Vision dan Machine Learning.

Sistem memanfaatkan MediaPipe Hands untuk mendeteksi posisi tangan, kemudian melakukan ekstraksi landmark dan mengklasifikasikan gesture menggunakan model Machine Learning.

### ✨ Fitur Utama

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

| Teknologi       | Fungsi                    |
| --------------- | ------------------------- |
| Python          | Bahasa Pemrograman        |
| Flask           | Backend Web               |
| OpenCV          | Pengolahan Citra          |
| MediaPipe Hands | Deteksi Landmark Tangan   |
| NumPy           | Operasi Numerik           |
| Pandas          | Pengolahan Dataset        |
| Scikit-Learn    | Machine Learning          |
| MLP             | Klasifikasi Huruf & Angka |
| Random Forest   | Klasifikasi Kata          |

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

Dataset yang digunakan pada penelitian ini berasal dari beberapa sumber.

### 🔤 Dataset Huruf BISINDO

Sumber:

https://www.kaggle.com/datasets/achmadnoer/alfabet-bisindo

Digunakan sebagai dataset huruf A-Z sebelum dilakukan ekstraksi landmark menggunakan MediaPipe Hands.

---

### 🔢 Dataset Angka BISINDO

Sumber:

https://data.mendeley.com/datasets/j4y5w2c8w9/1

Digunakan sebagai dataset angka 0-9 untuk proses pelatihan model MLP.

---

### 🤟 Dataset Kata BISINDO

Sumber:

https://www.kaggle.com/datasets/mulkanfajri/indonesian-sign-language-bisindo-word-dataset

Dataset ini dikumpulkan secara mandiri menggunakan webcam dan terdiri dari 5 kelas:

* Mendengar
* Tersenyum
* Ragu
* Damai
* Semoga Beruntung

Masing-masing kelas berisi sekitar 100 gambar.

---

## 📊 Ekstraksi Fitur

MediaPipe Hands menghasilkan:

* 21 Landmark Tangan
* Koordinat x
* Koordinat y
* Koordinat z

Perhitungan fitur:

```text
21 × 3 = 63 fitur
63 + 63 = 126 fitur
```

Karena sistem mendukung dua tangan maka total fitur yang digunakan adalah 126 fitur.

Landmark kemudian disimpan ke dalam file CSV sebelum dilakukan proses training model.

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

* Cepat
* Ringan
* Cocok untuk data landmark
* Akurasi tinggi

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

* Stabil
* Tidak mudah overfitting
* Cocok untuk dataset ukuran menengah

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

| Model                | Akurasi |
| -------------------- | ------- |
| MLP (Huruf & Angka)  | 97.9%   |
| Random Forest (Kata) | 98%     |

---

## 👨‍💻 Pengembang

**Mulkan Fajri**

Program Studi Teknik Informatika

Politeknik Negeri Lhokseumawe

---

<div align="center">

⭐ Jika proyek ini bermanfaat, berikan Star pada repository ini.

</div>
