# TagMind

TagMind adalah eksperimen **reinforcement learning** berupa permainan kejar-kejaran antara dua agen di peta 10×10.

- **Chaser** berusaha mengejar dan menangkap Runner.
- **Runner** berusaha menghindari Chaser dan bertahan sampai batas langkah episode.
- Kedua agen menggunakan **Deep Q-Network (DQN)** dengan target network dan replay buffer masing-masing.
- Tata letak lingkungan dibuat acak pada setiap episode.

> TagMind masih merupakan eksperimen. Agen belajar dari state fitur yang diberikan program, bukan dari gambar atau piksel secara langsung.

## Lingkungan dan Aksi

Peta berisi tembok, item energi, dan posisi awal agen. Setiap agen menerima state yang mencakup posisi relatif lawan, jarak lawan, energi, serta kondisi delapan petak di sekitarnya.

Aksi yang tersedia:

- Bergerak ke atas, bawah, kiri, atau kanan
- Melompat ke atas, bawah, kiri, atau kanan

## Kondisi Menang

- **Chaser menang** jika berhasil berada di petak yang sama dengan Runner.
- **Runner menang** jika bertahan sampai batas langkah episode tercapai.
- Episode berhenti ketika Runner tertangkap atau waktu habis.

## Struktur DQN

Setiap agen memiliki:

- Policy network dan target network
- Adam optimizer
- Replay buffer sendiri
- Nilai epsilon untuk mengatur eksplorasi
- Gradient clipping

Target network disinkronkan secara berkala dari policy network.

## Menjalankan Training di Google Colab

```bash
python "/content/drive/MyDrive/TagMind/Tagmind.py" \
  --rounds 100 \
  --device auto \
  --seed 42 \
  --output-dir "/content/drive/MyDrive/TagMind"
