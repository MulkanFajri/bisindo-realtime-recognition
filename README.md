Pengenalan Bahasa Isyarat Indonesia (BISINDO) Secara Realtime Menggunakan Machine Learning
Deskripsi

Proyek ini merupakan sistem pengenalan Bahasa Isyarat Indonesia (BISINDO) secara realtime menggunakan teknologi Computer Vision dan Machine Learning.

Sistem mampu mengenali:

Huruf A sampai Z
Angka 0 sampai 9
Kata BISINDO:
Mendengar
Tersenyum
Ragu
Damai
Semoga Beruntung

Proyek dibangun menggunakan MediaPipe Hands untuk ekstraksi landmark tangan, MLP (Multi Layer Perceptron) untuk klasifikasi huruf dan angka, serta Random Forest untuk klasifikasi kata BISINDO.

Teknologi yang Digunakan
Python
Flask
OpenCV
MediaPipe Hands
NumPy
Pandas
Scikit-Learn
MLP Classifier
Random Forest
Arsitektur Sistem

Dataset Gambar

↓

MediaPipe Hands

↓

Ekstraksi Landmark Tangan

↓

126 Fitur Landmark

↓

Training Model

↓

Model (.pkl)

↓

Flask Backend

↓

Website Realtime

↓

Prediksi BISINDO

Dataset
Huruf dan Angka

Dataset huruf dan angka diperoleh dari Kaggle dan kemudian diproses menggunakan MediaPipe Hands untuk menghasilkan fitur landmark.

Jumlah kelas:

26 Huruf (A-Z)
10 Angka (0-9)

Total:

36 Kelas

Kata BISINDO

Dataset kata dikumpulkan secara mandiri menggunakan webcam.

Kelas yang digunakan:

Mendengar
Tersenyum
Ragu
Damai
Semoga Beruntung

Setiap kelas terdiri dari sekitar 100 gambar.

Ekstraksi Fitur

MediaPipe Hands menghasilkan:

21 landmark tangan
Setiap landmark memiliki koordinat:
x
y
z

Perhitungan fitur:

21 landmark × 3 koordinat = 63 fitur

Karena mendukung dua tangan:

63 + 63 = 126 fitur

Seluruh fitur kemudian disimpan dalam file CSV sebelum dilakukan proses training.

Model Machine Learning
Huruf dan Angka

Metode:

MLP (Multi Layer Perceptron)

Alasan penggunaan:

Cocok untuk data numerik hasil landmark
Proses training cepat
Akurasi tinggi

Hasil pengujian:

Accuracy = 97.9%

Kata BISINDO

Metode:

Random Forest Classifier

Alasan penggunaan:

Stabil untuk dataset berukuran kecil hingga menengah
Mudah diimplementasikan
Akurasi tinggi

Hasil pengujian:

Accuracy = 98%

Struktur Folder
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

Cara Menjalankan
1. Clone Repository
git clone https://github.com/MulkanFajri/bisindo-realtime-recognition.git
2. Install Library
pip install -r requirements.txt
3. Jalankan Program
python app.py
4. Buka Browser
http://127.0.0.1:5000
Fitur Sistem
Deteksi tangan secara realtime
Visualisasi landmark MediaPipe
Prediksi huruf A-Z
Prediksi angka 0-9
Prediksi kata BISINDO
Menampilkan confidence prediksi
Pergantian mode huruf dan kata
Pengembang

Mulkan Fajri

Program Studi Teknik Informatika

Politeknik Negeri Lhokseumawe

Lisensi

Proyek ini dibuat untuk keperluan penelitian dan tugas akhir akademik.