# Minimal Math AI

Minimal Math AI adalah model PyTorch ultra-ringan untuk memprediksi hasil ekspresi aritmetika sederhana.

Model memakai embedding token, LSTM kecil, attention pooling, dan dua output:

- **answer** untuk hasil aritmetika
- **confidence** untuk tingkat keyakinan prediksi

## Fitur

- Tokenisasi ekspresi matematika
- Tiny LSTM dengan attention pooling
- Training dan validasi terpisah
- Signed-log normalization untuk angka negatif dan besar
- Confidence berdasarkan jarak prediksi terhadap jawaban
- Best checkpoint otomatis
- Training history dalam CSV dan JSON
- Mode training, testing, dan interactive
- Dukungan CPU dan CUDA

## Persyaratan

```bash
pip install torch numpy
