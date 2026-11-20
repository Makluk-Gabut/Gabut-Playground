# MaklukGabut's Game of Life

Simulasi multi-agen di grid 12×12. Enam agen utama mengelola energi, mencari makanan, dan saling membantu agar bertahan hidup. Agen utama dilatih dengan **Deep Q-Network (DQN)**; companion yang muncul dalam kondisi tertentu menggunakan policy DQN bersama.

> Ini simulasi survival multi-agen buatan sendiri, bukan implementasi Conway's Game of Life.

## Fitur

- Enam agen utama dengan policy dan replay buffer masing-masing.
- Gerakan biasa dan gerakan boost.
- Sistem energi, makanan, inventory, dorman, dan kematian.
- Interaksi antaragen: memberi makanan, berbagi informasi, menghidupkan kembali, membawa, dan menurunkan agen dorman.
- Companion dengan stamina, inventory, dan aksi pendamping.
- Checkpoint training dan statistik episode.
- Visualisasi langsung menggunakan Pygame.
- Grafik statistik training menggunakan Matplotlib.

## Persyaratan

- Python 3.9 atau lebih baru
- PyTorch
- NumPy
- Matplotlib untuk membuat grafik
- Pygame opsional, diperlukan untuk visualisasi dan demo

Pasang dependensi:

```bash
pip install torch numpy matplotlib pygame
```

Jika tidak membutuhkan visualisasi, Pygame boleh tidak dipasang.

## Menjalankan

Dari root repository:

```bash
python "the_best/AI/MaklukGabut's_gameoflife.py" train
```

Training default berjalan selama 500 episode dengan maksimum 200 langkah per episode. Pengaturan training tersedia di bagian `CFG` dalam script.

### Melanjutkan dari checkpoint

```bash
python "the_best/AI/MaklukGabut's_gameoflife.py" train --resume
```

Opsi `--resume` akan mencoba memuat checkpoint terbaru dari direktori checkpoint yang dikonfigurasi.

### Training dengan visualisasi

```bash
python "the_best/AI/MaklukGabut's_gameoflife.py" train --visual
```

### Menonton demo

```bash
python "the_best/AI/MaklukGabut's_gameoflife.py" demo
```

Demo memerlukan Pygame dan checkpoint training. Di jendela visualisasi, tombol panah atas/bawah mengatur kecepatan dan `Q` menutup visualisasi.

### Melihat grafik

```bash
python "the_best/AI/MaklukGabut's_gameoflife.py" plot
```

Perintah ini memuat statistik dari checkpoint terbaru dan menampilkan grafik jika Matplotlib tersedia.

### Melihat checkpoint

```bash
python "the_best/AI/MaklukGabut's_gameoflife.py" list
```

## Aturan simulasi secara ringkas

### Agen utama

Agen utama dapat bergerak, menggunakan boost, mengambil dan memakan makanan, memberi makanan kepada agen lain, berbagi informasi makanan, menghidupkan kembali agen dorman, serta membawa atau menurunkan agen dorman.

Agen dapat menjadi dorman saat mencapai batas langkah tertentu atau terlalu lama tanpa ditemani. Agen dorman memiliki timer; jika tidak tertolong sebelum timer habis, agen akan mati. Energi yang habis juga menyebabkan kematian.

### Companion

Companion dapat muncul ketika sedikitnya tiga agen utama berada di satu sel selama beberapa langkah berturut-turut, selama jumlah companion aktif belum mencapai batas. Companion memiliki stamina dan policy bersama, serta dapat mengambil atau memberikan makanan, menghidupkan kembali, membawa, dan menurunkan agen dorman.

## Training dan statistik

Nilai utama konfigurasi dapat diubah pada `CFG` di dalam script. Default-nya mencakup:

- Episode: 500
- Maksimum langkah per episode: 200
- Batch size: 64
- Gamma: 0.99
- Learning rate: 0.001
- Epsilon awal: 1.0
- Epsilon minimum: 0.05

Progress dicetak setiap 10 episode. Statistik yang dicatat meliputi total reward, jumlah agen hidup, agen dorman, kematian, jumlah langkah, epsilon, dan companion aktif.

## Hasil dan checkpoint

Secara default, hasil dibuat relatif terhadap direktori kerja saat script dijalankan:

```text
checkpoints/
training_stats.png
```

Checkpoint disimpan berkala dan mencakup bobot policy/target, optimizer, metadata, serta statistik. Saat melanjutkan training, pastikan direktori checkpoint yang dipakai sama dengan direktori sebelumnya. Grafik statistik diperbarui sesuai interval plot dalam konfigurasi.

## Catatan

- Training dan evaluasi menggunakan agen-agen yang dilatih di lingkungan simulasi yang sama; performa yang baik di simulasi belum tentu berlaku di lingkungan lain.
- Setiap agen utama memiliki jaringan sendiri, sedangkan semua companion berbagi satu policy dan satu replay buffer.
- Hasil training dapat berubah karena inisialisasi jaringan dan lingkungan yang menggunakan nilai acak.
- Visualisasi Pygame memerlukan lingkungan desktop; pada notebook atau server tanpa tampilan grafis, jalankan training tanpa `--visual`.
