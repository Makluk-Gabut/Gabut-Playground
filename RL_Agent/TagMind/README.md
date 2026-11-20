# TagMind

TagMind adalah eksperimen reinforcement learning menggunakan dua agen DQN yang saling berlatih:

- **Chaser** mencoba mendekati dan menangkap Runner.
- **Runner** mencoba menjauh dan bertahan sampai batas langkah.
- Keduanya belajar dari reward, replay buffer, dan target network.
- Setiap replay buffer dan policy dipisahkan per agen.

## Eksekusi di Google Colab

Jalankan file utama:

```bash
python "/content/drive/MyDrive/TagMind/Tagmind.py" \
  --rounds 100 \
  --device auto \
  --seed 42 \
  --output-dir "/content/drive/MyDrive/TagMind"
