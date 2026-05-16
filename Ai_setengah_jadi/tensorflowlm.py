"""
==============================================================
  Membangun LLM Sederhana dengan TensorFlow
  Arsitektur: Transformer Decoder (GPT-style)
==============================================================
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import re

# ─────────────────────────────────────────────
# 1. KONFIGURASI MODEL
# ─────────────────────────────────────────────
class Config:
    vocab_size    = 5000      # Ukuran vocabulary
    embed_dim     = 128       # Dimensi embedding
    num_heads     = 4         # Jumlah attention heads
    ff_dim        = 512       # Dimensi feed-forward layer
    num_layers    = 4         # Jumlah transformer block
    max_seq_len   = 128       # Panjang sekuens maksimum
    dropout_rate  = 0.1
    batch_size    = 32
    epochs        = 10
    learning_rate = 3e-4


# ─────────────────────────────────────────────
# 2. TOKENIZER SEDERHANA
# ─────────────────────────────────────────────
class SimpleTokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.word2idx   = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
        self.idx2word   = {v: k for k, v in self.word2idx.items()}

    def fit(self, texts):
        from collections import Counter
        words = []
        for text in texts:
            words.extend(self._tokenize(text))
        counts = Counter(words).most_common(self.vocab_size - len(self.word2idx))
        for word, _ in counts:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx]  = word

    def _tokenize(self, text):
        return re.findall(r'\w+|[^\w\s]', text.lower())

    def encode(self, text, max_len=None):
        tokens = [self.word2idx.get(w, 1) for w in self._tokenize(text)]
        tokens = [2] + tokens + [3]          # <bos> ... <eos>
        if max_len:
            tokens = tokens[:max_len]
            tokens += [0] * (max_len - len(tokens))
        return tokens

    def decode(self, ids):
        words = [self.idx2word.get(i, "<unk>") for i in ids
                 if i not in (0, 2, 3)]
        return " ".join(words)


# ─────────────────────────────────────────────
# 3. POSITIONAL ENCODING
# ─────────────────────────────────────────────
class PositionalEncoding(layers.Layer):
    def __init__(self, max_seq_len, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.pos_encoding = self._make_pos_encoding(max_seq_len, embed_dim)

    def _make_pos_encoding(self, length, depth):
        positions = np.arange(length)[:, np.newaxis]          # (L, 1)
        dims      = np.arange(depth)[np.newaxis, :]           # (1, D)
        angles    = positions / (10000 ** (2 * (dims // 2) / depth))
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        return tf.cast(angles[np.newaxis, :, :], dtype=tf.float32)  # (1,L,D)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]


# ─────────────────────────────────────────────
# 4. CAUSAL MULTI-HEAD SELF-ATTENTION
# ─────────────────────────────────────────────
class CausalSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.mha     = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads,
            dropout=dropout_rate
        )
        self.norm    = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        seq_len = tf.shape(x)[1]
        # Causal mask: token hanya bisa "melihat" token sebelumnya
        mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        mask = mask[tf.newaxis, tf.newaxis, :, :]

        attn_out = self.mha(x, x, x, attention_mask=None,
                            use_causal_mask=True, training=training)
        attn_out = self.dropout(attn_out, training=training)
        return self.norm(x + attn_out)


# ─────────────────────────────────────────────
# 5. FEED-FORWARD NETWORK
# ─────────────────────────────────────────────
class FeedForward(layers.Layer):
    def __init__(self, embed_dim, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.dense1  = layers.Dense(ff_dim, activation="gelu")
        self.dense2  = layers.Dense(embed_dim)
        self.norm    = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        out = self.dense1(x)
        out = self.dense2(out)
        out = self.dropout(out, training=training)
        return self.norm(x + out)


# ─────────────────────────────────────────────
# 6. TRANSFORMER BLOCK
# ─────────────────────────────────────────────
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout_rate)
        self.ff   = FeedForward(embed_dim, ff_dim, dropout_rate)

    def call(self, x, training=False):
        x = self.attn(x, training=training)
        x = self.ff(x, training=training)
        return x


# ─────────────────────────────────────────────
# 7. MODEL LLM UTAMA (GPT-style)
# ─────────────────────────────────────────────
class MiniLLM(keras.Model):
    def __init__(self, config: Config, **kwargs):
        super().__init__(**kwargs)
        self.cfg = config

        self.embedding = layers.Embedding(config.vocab_size, config.embed_dim)
        self.pos_enc   = PositionalEncoding(config.max_seq_len, config.embed_dim)
        self.dropout   = layers.Dropout(config.dropout_rate)

        self.blocks = [
            TransformerBlock(config.embed_dim, config.num_heads,
                             config.ff_dim, config.dropout_rate)
            for _ in range(config.num_layers)
        ]

        self.norm    = layers.LayerNormalization(epsilon=1e-6)
        self.lm_head = layers.Dense(config.vocab_size)

    def call(self, x, training=False):
        x = self.embedding(x)                    # (B, T, D)
        x = self.pos_enc(x)
        x = self.dropout(x, training=training)

        for block in self.blocks:
            x = block(x, training=training)

        x = self.norm(x)
        logits = self.lm_head(x)                 # (B, T, vocab_size)
        return logits

    def generate(self, tokenizer, prompt, max_new_tokens=50, temperature=0.8):
        ids = tokenizer.encode(prompt, max_len=self.cfg.max_seq_len - max_new_tokens)
        ids = [i for i in ids if i != 0]         # hapus padding

        for _ in range(max_new_tokens):
            ctx = ids[-self.cfg.max_seq_len:]
            inp = tf.constant([ctx], dtype=tf.int32)
            logits = self(inp, training=False)[0, -1, :]  # logit token terakhir

            # Temperature sampling
            logits = logits / temperature
            probs  = tf.nn.softmax(logits).numpy()
            next_id = np.random.choice(len(probs), p=probs)

            ids.append(next_id)
            if next_id == 3:                     # <eos>
                break

        return tokenizer.decode(ids)


# ─────────────────────────────────────────────
# 8. TRAINING UTILS
# ─────────────────────────────────────────────
def make_dataset(texts, tokenizer, config):
    """Buat tf.data.Dataset dari list teks."""
    all_ids = []
    for text in texts:
        ids = tokenizer.encode(text, max_len=config.max_seq_len + 1)
        all_ids.append(ids)

    all_ids = np.array(all_ids, dtype=np.int32)
    inputs  = all_ids[:, :-1]    # (N, T)
    targets = all_ids[:, 1:]     # (N, T)  — shift kanan 1

    ds = tf.data.Dataset.from_tensor_slices((inputs, targets))
    ds = ds.shuffle(1000).batch(config.batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


@tf.function
def train_step(model, optimizer, loss_fn, x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        # Flatten untuk cross-entropy
        loss = loss_fn(tf.reshape(y, [-1]),
                       tf.reshape(logits, [-1, model.cfg.vocab_size]))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def train(model, dataset, config):
    optimizer = keras.optimizers.Adam(config.learning_rate)
    loss_fn   = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    print(f"\n{'─'*50}")
    print(f"  Mulai training: {config.epochs} epoch")
    print(f"{'─'*50}")

    for epoch in range(1, config.epochs + 1):
        total_loss, steps = 0.0, 0
        for x_batch, y_batch in dataset:
            loss = train_step(model, optimizer, loss_fn, x_batch, y_batch)
            total_loss += loss.numpy()
            steps += 1
        avg_loss = total_loss / steps
        print(f"  Epoch {epoch:02d}/{config.epochs}  |  Loss: {avg_loss:.4f}  |  "
              f"Perplexity: {np.exp(avg_loss):.2f}")

    print(f"{'─'*50}\n")


# ─────────────────────────────────────────────
# 9. DEMO — JALANKAN LANGSUNG
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # ── Contoh data latih (ganti dengan corpus kamu) ──────────────────────
    sample_texts = [
        "kucing itu makan ikan di dapur setiap pagi",
        "burung terbang tinggi di langit biru yang cerah",
        "anak-anak bermain bola di lapangan sekolah",
        "ibu memasak nasi goreng yang lezat untuk sarapan",
        "buku itu berisi cerita yang sangat menarik dan menghibur",
        "anjing berlari kencang mengejar bola di taman",
        "matahari terbit dari timur menyinari seluruh bumi",
        "nelayan pergi melaut saat fajar untuk mencari ikan",
        "pohon kelapa tumbuh subur di tepi pantai yang indah",
        "hujan turun deras membasahi seluruh kota semalam",
    ] * 20   # duplikasi agar cukup data untuk demo

    cfg = Config()

    # Tokenizer
    print("Membangun vocabulary...")
    tokenizer = SimpleTokenizer(cfg.vocab_size)
    tokenizer.fit(sample_texts)
    cfg.vocab_size = len(tokenizer.word2idx)
    print(f"Vocab size: {cfg.vocab_size} token")

    # Dataset
    dataset = make_dataset(sample_texts, tokenizer, cfg)

    # Model
    model = MiniLLM(cfg, name="MiniLLM")
    # Build model dengan dummy input
    dummy = tf.zeros((1, cfg.max_seq_len), dtype=tf.int32)
    _ = model(dummy)
    model.summary()

    # Training
    train(model, dataset, cfg)

    # Simpan model
    model.save_weights("mini_llm_weights.h5")
    print("Model tersimpan di mini_llm_weights.h5")

    # ── Generate teks ──────────────────────────────────────────────────────
    print("\n── Contoh Generate Teks ──")
    prompts = ["kucing itu", "burung terbang", "ibu memasak"]
    for p in prompts:
        result = model.generate(tokenizer, p, max_new_tokens=15, temperature=0.9)
        print(f"  Prompt : '{p}'")
        print(f"  Output : '{result}'\n")
