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
```

## Menjalankan
```bash
python "the_best/AI/mathAI.py"
```

### Menu yang tersedia:

Menu yang tersedia:

```Text
1. Train AI
2. Test AI
3. Interactive Mode
4. Exit
```


## Training
Masukkan difficulty, jumlah epoch, batch size, dan learning rate.

Contoh:

Training
Masukkan difficulty, jumlah epoch, batch size, dan learning rate.

Contoh:

```Text
Difficulty (1-5, default 1): 1
Epochs (default 150): 150
Batch size (default 64): 64
Learning rate (default 0.005): 0.005
```

## Default training:

* 3.000 data training
* 500 data validasi
* Epoch maksimum 150
* Batch size 64
* Learning rate 0.005
* Patience early stopping 30 epoch
* Metrik: loss dan MAE




