# Pengenalan Bahasa Isyarat Indonesia (BISINDO) Realtime

Proyek pengenalan Bahasa Isyarat Indonesia (BISINDO) secara realtime menggunakan Machine Learning, MediaPipe Hands, Flask, dan OpenCV.

## Metode

### Huruf & Angka
- MediaPipe Hands
- MLP Classifier

### Kata BISINDO
- MediaPipe Hands
- Random Forest

## Dataset

### Huruf & Angka
Dataset diperoleh dari Kaggle.

### Kata BISINDO
Dataset dikumpulkan sendiri menggunakan webcam.

## Akurasi

- Huruf & Angka : 97.9%
- Kata BISINDO : 98%

## Fitur

- Deteksi Huruf A-Z
- Deteksi Angka 0-9
- Deteksi Kata BISINDO
- Realtime Webcam
- Tampilan Website Flask

## Cara Menjalankan

```bash
pip install -r requirements.txt
python app.py