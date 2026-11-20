# TagMind

TagMind adalah eksperimen **reinforcement learning** berupa permainan kejar-kejaran dua agen di peta 10×10.

- **Chaser** mencoba mengejar dan menangkap Runner.
- **Runner** mencoba menjauh dan bertahan selama mungkin.
- Kedua agen memakai **DQN** dengan replay buffer dan target network terpisah.
- Lingkungan dibuat acak setiap episode.

## Lingkungan

Peta berisi tembok, item energi, dan posisi awal dua agen.

Aksi yang tersedia:

- Gerak atas
- Gerak bawah
- Gerak kiri
- Gerak kanan
- Lompat ke atas
- Lompat ke bawah
- Lompat ke kiri
- Lompat ke kanan

Setiap agen menerima state berupa posisi relatif lawan, jarak lawan, energi, dan keadaan delapan petak di sekitarnya.

## Arah Pemenang

- Chaser menang jika berhasil berada pada petak yang sama dengan Runner.
- Runner menang jika bertahan sampai batas langkah episode berakhir.
- Episode otomatis berhenti saat tertangkap atau waktu habis.

## Struktur DQN

Setiap agen memiliki:

- Policy network
- Target network
- Adam optimizer
- Replay buffer sendiri
- Nilai epsilon untuk eksplorasi
- Gradient clipping

Target jaringan disinkronkan secara berkala dari policy jaringan.

## Training di Google Colab

Gunakan script:

```bash
python "/content/drive/MyDrive/TagMind/Tagmind.py" \
  --rounds 100 \
  --device auto \
  --seed 42 \
  --output-dir "/content/drive/MyDrive/TagMind"
```

Melanjutkan checkpoint:

```bash
python "/content/drive/MyDrive/TagMind/Tagmind.py" \
  --rounds 1000 \
  --device auto \
  --seed 42 \
  --resume "/content/drive/MyDrive/TagMind/checkpoint_final.pth" \
  --output-dir "/content/drive/MyDrive/TagMind"
```

## Parameter Penting

- Ukuran peta: `10 × 10`
- Maksimum langkah: `50`
- Replay buffer: `20.000`
- Batch size: `64`
- Gamma: `0.95`
- Learning rate: `1e-3`
- Epsilon awal: `1.0`
- Epsilon minimum: `0.08`
- Epsilon decay: `0.995`

## File Hasil

Semua hasil training disarankan disimpan di:

```text
/content/drive/MyDrive/TagMind/
```

Contoh isi folder:

```text
TagMind/
├── Tagmind.py
├── training.log
├── tagmind_metrics.csv
├── reward_plot.png
├── checkpoint_ep100.pth
├── checkpoint_final.pth
├── checkpoint_interrupted.pth
└── frames/
```

| File | Fungsi |
|---|---|
| `training.log` | Menyimpan dan menampilkan log episode |
| `tagmind_metrics.csv` | Reward, winner, langkah, epsilon, loss, dan ukuran replay |
| `reward_plot.png` | Grafik reward dan stabilitas training |
| `checkpoint_*.pth` | Bobot model, optimizer, epsilon, dan replay buffer |
| `frames/` | Gambar hasil episode yang disimpan otomatis |

## Membaca Hasil di Colab

```python
import pandas as pd
from IPython.display import display, Image

folder = "/content/drive/MyDrive/TagMind"
df = pd.read_csv(f"{folder}/tagmind_metrics.csv")

display(df.tail(20))
display(Image(filename=f"{folder}/reward_plot.png"))
```

### Evaluasi 500 Episode Terakhir

```python
import pandas as pd

df = pd.read_csv("/content/drive/MyDrive/TagMind/tagmind_metrics.csv")
tail = df.tail(500)

print(tail["winner"].value_counts())
print("Chaser win rate:", (tail["winner"] == "chaser").mean())
print("Runner win rate:", (tail["winner"] == "runner").mean())
print("Average Chaser reward:", tail["chaser_reward"].mean())
print("Average Runner reward:", tail["runner_reward"].mean())
print("Average episode steps:", tail["steps"].mean())
```

## Menggambar Loss

```python
import pandas as pd
import matplotlib.pyplot as plt

folder = "/content/drive/MyDrive/TagMind"
df = pd.read_csv(f"{folder}/tagmind_metrics.csv")

ax = df.plot(
    x="episode",
    y=["chaser_loss", "runner_loss"],
    figsize=(10, 5),
    alpha=0.75,
)
ax.set_title("TagMind Training Loss")
ax.set_xlabel("Episode")
ax.set_ylabel("Loss")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{folder}/loss_plot.png", dpi=150)
plt.show()
```

## Cara Menilai Training

Satu lonjakan kemenangan tidak berarti model sudah stabil. Periksa beberapa indikator berikut:

- Rolling win rate 100–500 episode terakhir
- Rata-rata reward Chaser dan Runner
- Rata-rata langkah setiap episode
- Nilai epsilon
- Nilai loss
- Ukuran replay buffer
- Perubahan performa sebelum dan sesudah checkpoint

Jika satu agen terus menang sementara agen lain hampir tidak berkembang, pertimbangkan untuk meninjau reward shaping, epsilon decay, kualitas state, dan keseimbangan self-play.

## Catatan

TagMind masih berupa eksperimen. Agen mempelajari pola dari state fitur yang disediakan program, bukan dari gambar mentah atau piksel. Hasil 100 atau 1.000 episode belum menjamin strategi yang kuat.
