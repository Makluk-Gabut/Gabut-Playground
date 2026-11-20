"""Train a small decoder-only TensorFlow language model on a text corpus.

Example (Colab):
  python tensorflowlm.py \
    --data /content/drive/MyDrive/shakespeare/shakespeare_corpus_en_final.txt \
    --output-dir /content/drive/MyDrive/NanoLM_Shakespeare_2k/tensorflow \
    --tokenizer /content/drive/MyDrive/NanoLM_Shakespeare_2k/nanolm_bpe_2k.json \
    --steps 12000 --require-gpu

The tokenizer JSON format is Hugging Face `tokenizers`. If --tokenizer is omitted,
this script trains a byte-level BPE tokenizer from the corpus. Training examples
are contiguous next-token windows; train/validation token streams are split before
windows are created to avoid windows crossing the split boundary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


PAD_TOKEN = "<|pad|>"
BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|eos|>"


@dataclass
class Config:
    vocab_size: int
    embed_dim: int = 128
    num_heads: int = 4
    ff_dim: int = 512
    num_layers: int = 4
    max_seq_len: int = 128
    dropout_rate: float = 0.1
    batch_size: int = 32
    learning_rate: float = 3e-4
    seed: int = 1337


class PositionalEncoding(layers.Layer):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, max_seq_len: int, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        position = np.arange(max_seq_len, dtype=np.float32)[:, None]
        dimension = np.arange(embed_dim, dtype=np.float32)[None, :]
        angle_rates = 1.0 / np.power(10000.0, (2 * np.floor(dimension / 2)) / embed_dim)
        angles = position * angle_rates
        encoding = np.zeros((max_seq_len, embed_dim), dtype=np.float32)
        encoding[:, 0::2] = np.sin(angles[:, 0::2])
        encoding[:, 1::2] = np.cos(angles[:, 1::2])
        self.pos_encoding = tf.constant(encoding[None, :, :], dtype=tf.float32)

    def call(self, x):
        length = tf.shape(x)[1]
        return x + tf.cast(self.pos_encoding[:, :length, :], x.dtype)


class CausalSelfAttention(layers.Layer):
    def __init__(self, embed_dim: int, num_heads: int, dropout_rate: float, **kwargs):
        super().__init__(**kwargs)
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout_rate,
        )
        self.dropout = layers.Dropout(dropout_rate)
        self.norm = layers.LayerNormalization(epsilon=1e-5)

    def call(self, x, attention_mask=None, training=False):
        # Keras combines this key-padding mask with its built-in causal mask.
        attended = self.mha(
            query=x,
            key=x,
            value=x,
            attention_mask=attention_mask,
            use_causal_mask=True,
            training=training,
        )
        return self.norm(x + self.dropout(attended, training=training))


class FeedForward(layers.Layer):
    def __init__(self, embed_dim: int, ff_dim: int, dropout_rate: float, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = layers.Dense(ff_dim, activation="gelu")
        self.dense2 = layers.Dense(embed_dim)
        self.dropout = layers.Dropout(dropout_rate)
        self.norm = layers.LayerNormalization(epsilon=1e-5)

    def call(self, x, training=False):
        out = self.dense2(self.dense1(x))
        return self.norm(x + self.dropout(out, training=training))


class TransformerBlock(layers.Layer):
    def __init__(self, config: Config, **kwargs):
        super().__init__(**kwargs)
        self.attention = CausalSelfAttention(
            config.embed_dim, config.num_heads, config.dropout_rate
        )
        self.feed_forward = FeedForward(
            config.embed_dim, config.ff_dim, config.dropout_rate
        )

    def call(self, x, attention_mask=None, training=False):
        x = self.attention(x, attention_mask=attention_mask, training=training)
        return self.feed_forward(x, training=training)


class MiniLLM(keras.Model):
    def __init__(self, config: Config, pad_id: int, **kwargs):
        super().__init__(**kwargs)
        self.cfg = config
        self.pad_id = int(pad_id)
        self.embedding = layers.Embedding(config.vocab_size, config.embed_dim)
        self.position = PositionalEncoding(config.max_seq_len, config.embed_dim)
        self.dropout = layers.Dropout(config.dropout_rate)
        self.blocks = [TransformerBlock(config, name=f"block_{i}") for i in range(config.num_layers)]
        self.final_norm = layers.LayerNormalization(epsilon=1e-5)
        self.lm_head = layers.Dense(config.vocab_size)

    def call(self, token_ids, training=False):
        # Mask padding keys in attention; causal masking is applied in each block.
        padding_mask = tf.not_equal(token_ids, self.pad_id)[:, tf.newaxis, :]
        x = self.embedding(token_ids)
        x = self.position(x)
        x = self.dropout(x, training=training)
        for block in self.blocks:
            x = block(x, attention_mask=padding_mask, training=training)
        return self.lm_head(self.final_norm(x))

    def get_config_dict(self):
        return {"config": asdict(self.cfg), "pad_id": self.pad_id}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="UTF-8 text corpus")
    parser.add_argument("--output-dir", required=True, help="Directory for tokenizer, checkpoints, logs and plots")
    parser.add_argument("--tokenizer", default=None, help="Existing Hugging Face tokenizers JSON; trains BPE if omitted")
    parser.add_argument("--vocab-size", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=12000, help="Target total optimizer step, including resumed steps")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--resume", action="store_true", help="Restore latest full TensorFlow checkpoint in output-dir")
    parser.add_argument("--require-gpu", action="store_true", help="Stop if no GPU is visible")
    parser.add_argument("--prompt", default="To be, or not to be:")
    parser.add_argument("--generate", type=int, default=200, help="Number of new tokens for final sample")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    return parser.parse_args()


def build_tokenizer(data_path: Path, tokenizer_path: Optional[Path], output_path: Path, vocab_size: int):
    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    except ImportError as exc:
        raise RuntimeError("Install `tokenizers` (`pip install tokenizers`) to use the BPE tokenizer.") from exc

    if tokenizer_path:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    else:
        if vocab_size < 300:
            raise ValueError("Byte-level BPE vocab-size must be at least 300")
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]", byte_fallback=True))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.Sequence([decoders.ByteFallback(), decoders.ByteLevel()])
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["[UNK]", PAD_TOKEN, BOS_TOKEN, EOS_TOKEN],
        )
        text = data_path.read_text(encoding="utf-8")
        tokenizer.train_from_iterator([text], trainer=trainer)

    # Ensure saved tokenizer has explicit control tokens, even if a PyTorch tokenizer is reused.
    tokenizer.add_special_tokens([PAD_TOKEN, BOS_TOKEN, EOS_TOKEN])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    ids = {name: tokenizer.token_to_id(name) for name in (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN)}
    if any(value is None for value in ids.values()):
        raise RuntimeError(f"Tokenizer is missing required special tokens: {ids}")
    return tokenizer, ids


def encode_corpus(tokenizer, text: str, bos_id: int, eos_id: int) -> np.ndarray:
    # The corpus is treated as one stream with explicit document boundaries.
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    if not ids:
        raise ValueError("The corpus produced no tokens")
    return np.asarray([bos_id, *ids, eos_id], dtype=np.int32)


def make_window_dataset(token_ids: np.ndarray, block_size: int, batch_size: int, shuffle: bool, seed: int):
    ids = tf.constant(token_ids, dtype=tf.int32)
    num_windows = int(token_ids.size) - block_size
    if num_windows < 1:
        raise ValueError(f"Need more than {block_size} tokens in each data split; found {token_ids.size}")
    ds = tf.data.Dataset.range(num_windows)

    def get_window(start):
        start = tf.cast(start, tf.int32)
        x = tf.slice(ids, [start], [block_size])
        y = tf.slice(ids, [start + 1], [block_size])
        return x, y

    ds = ds.map(get_window, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(min(num_windows, 10000), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=shuffle)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def masked_loss_and_count(labels, logits, pad_id: int):
    per_token = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    mask = tf.cast(tf.not_equal(labels, pad_id), per_token.dtype)
    loss_sum = tf.reduce_sum(per_token * mask)
    token_count = tf.reduce_sum(mask)
    return tf.math.divide_no_nan(loss_sum, token_count), loss_sum, token_count


def make_train_step(model, optimizer, pad_id: int):
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss, _, _ = masked_loss_and_count(y, logits, pad_id)
        gradients = tape.gradient(loss, model.trainable_variables)
        pairs = [(g, v) for g, v in zip(gradients, model.trainable_variables) if g is not None]
        optimizer.apply_gradients(pairs)
        return loss
    return train_step


def evaluate(model, dataset, pad_id: int, max_batches: int) -> float:
    loss_sum = 0.0
    token_count = 0.0
    for batch_index, (x, y) in enumerate(dataset):
        if batch_index >= max_batches:
            break
        logits = model(x, training=False)
        _, batch_loss_sum, batch_tokens = masked_loss_and_count(y, logits, pad_id)
        loss_sum += float(batch_loss_sum.numpy())
        token_count += float(batch_tokens.numpy())
    if token_count == 0:
        return float("nan")
    return loss_sum / token_count


def sample_text(model, tokenizer, special_ids, prompt: str, max_new_tokens: int, temperature: float, top_k: int, block_size: int):
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")
    ids = [special_ids[BOS_TOKEN]] + tokenizer.encode(prompt, add_special_tokens=False).ids
    if not ids:
        ids = [special_ids[BOS_TOKEN]]
    ids = ids[-block_size:]

    for _ in range(max_new_tokens):
        context = tf.constant([ids[-block_size:]], dtype=tf.int32)
        logits = model(context, training=False)[0, -1, :] / temperature
        if top_k > 0 and top_k < int(logits.shape[-1]):
            values, _ = tf.math.top_k(logits, k=top_k)
            threshold = values[-1]
            logits = tf.where(logits < threshold, tf.constant(-1e10, logits.dtype), logits)
        next_id = int(tf.random.categorical(logits[None, :], num_samples=1)[0, 0].numpy())
        if next_id in (special_ids[PAD_TOKEN], special_ids[BOS_TOKEN]):
            continue
        ids.append(next_id)
        if next_id == special_ids[EOS_TOKEN]:
            break
    return tokenizer.decode(ids, skip_special_tokens=True)


def write_loss_plot(csv_path: Path, png_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
    if not rows:
        return
    steps = [int(row["step"]) for row in rows]
    train_losses = [float(row["train_loss"]) for row in rows]
    val_losses = [float(row["validation_loss"]) for row in rows]
    plt.figure(figsize=(10, 5))
    plt.plot(steps, train_losses, label="Train loss")
    plt.plot(steps, val_losses, label="Validation loss")
    plt.xlabel("Optimizer step")
    plt.ylabel("Token cross-entropy")
    plt.title("TensorFlow LM loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not data_path.is_file():
        raise FileNotFoundError(f"Corpus not found: {data_path}")
    if args.steps < 1 or args.block_size < 2 or args.batch_size < 1:
        raise ValueError("steps, batch-size and block-size must be positive (block-size >= 2)")
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("val-fraction must be between 0 and 1")
    if args.eval_interval < 1 or args.eval_batches < 1:
        raise ValueError("eval-interval and eval-batches must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)

    gpus = tf.config.list_physical_devices("GPU")
    if args.require_gpu and not gpus:
        raise RuntimeError("No TensorFlow GPU found. Enable a GPU runtime, then rerun.")
    print("TensorFlow:", tf.__version__)
    print("Device:", ", ".join(device.name for device in gpus) if gpus else "CPU (no GPU detected)")

    text = data_path.read_text(encoding="utf-8")
    tokenizer_source = Path(args.tokenizer) if args.tokenizer else None
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer, special_ids = build_tokenizer(
        data_path, tokenizer_source, tokenizer_path, args.vocab_size
    )
    all_ids = encode_corpus(tokenizer, text, special_ids[BOS_TOKEN], special_ids[EOS_TOKEN])
    split_at = int(len(all_ids) * (1.0 - args.val_fraction))
    train_ids, val_ids = all_ids[:split_at], all_ids[split_at:]
    if len(train_ids) <= args.block_size or len(val_ids) <= args.block_size:
        raise ValueError(
            f"Corpus too short for block-size {args.block_size}: "
            f"train tokens={len(train_ids)}, validation tokens={len(val_ids)}"
        )

    config = Config(
        vocab_size=tokenizer.get_vocab_size(),
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        num_layers=args.num_layers,
        max_seq_len=args.block_size,
        dropout_rate=args.dropout,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    (output_dir / "config.json").write_text(
        json.dumps({**asdict(config), "special_token_ids": special_ids}, indent=2),
        encoding="utf-8",
    )

    train_ds = make_window_dataset(train_ids, args.block_size, args.batch_size, True, args.seed)
    val_ds = make_window_dataset(val_ids, args.block_size, args.batch_size, False, args.seed)

    model = MiniLLM(config, special_ids[PAD_TOKEN], name="TensorFlowMiniLLM")
    model(tf.zeros((1, args.block_size), dtype=tf.int32), training=False)
    optimizer = keras.optimizers.Adam(
        learning_rate=config.learning_rate,
        global_clipnorm=1.0,
    )
    step = tf.Variable(0, dtype=tf.int64, trainable=False, name="step")
    best_val_loss = tf.Variable(float("inf"), dtype=tf.float32, trainable=False, name="best_val_loss")
    checkpoint = tf.train.Checkpoint(
        model=model, optimizer=optimizer, step=step, best_val_loss=best_val_loss
    )
    checkpoint_dir = output_dir / "checkpoints"
    latest_manager = tf.train.CheckpointManager(checkpoint, str(checkpoint_dir / "latest"), max_to_keep=3)
    best_manager = tf.train.CheckpointManager(checkpoint, str(checkpoint_dir / "best"), max_to_keep=1)
    if args.resume:
        if latest_manager.latest_checkpoint:
            checkpoint.restore(latest_manager.latest_checkpoint).expect_partial()
            print(f"Restored checkpoint: {latest_manager.latest_checkpoint} (step {int(step.numpy())})")
        else:
            print("No checkpoint found to resume; starting from scratch.")

    if int(step.numpy()) >= args.steps:
        print(f"Checkpoint is already at step {int(step.numpy())}, target is {args.steps}; nothing to train.")
        return

    train_step = make_train_step(model, optimizer, special_ids[PAD_TOKEN])
    iterator = iter(train_ds.repeat())
    log_path = output_dir / "training.csv"
    log_exists = log_path.exists() and log_path.stat().st_size > 0
    log_file = log_path.open("a", newline="", encoding="utf-8")
    writer = csv.writer(log_file)
    if not log_exists:
        writer.writerow(["step", "train_loss", "validation_loss", "perplexity", "seconds"])
        log_file.flush()

    interval_losses = []
    interval_start = time.time()
    log_txt_path = output_dir / "training.log"
    log_txt = log_txt_path.open("a", encoding="utf-8")

    def log(message: str):
        print(message, flush=True)
        log_txt.write(message + "\n")
        log_txt.flush()

    log(f"Training from step {int(step.numpy())} to {args.steps}; vocab={config.vocab_size}; train/val tokens={len(train_ids)}/{len(val_ids)}")
    try:
        while int(step.numpy()) < args.steps:
            x_batch, y_batch = next(iterator)
            loss = train_step(x_batch, y_batch)
            step.assign_add(1)
            interval_losses.append(float(loss.numpy()))
            current_step = int(step.numpy())

            if current_step % args.eval_interval == 0 or current_step == args.steps:
                val_loss = evaluate(model, val_ds, special_ids[PAD_TOKEN], args.eval_batches)
                train_loss = float(np.mean(interval_losses)) if interval_losses else float("nan")
                # Clamp before exponentiation to avoid overflow while preserving the logged loss.
                perplexity = math.exp(min(val_loss, 80.0)) if math.isfinite(val_loss) else float("inf")
                elapsed = time.time() - interval_start
                writer.writerow([current_step, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{perplexity:.6g}", f"{elapsed:.2f}"])
                log_file.flush()
                log(f"Step {current_step}/{args.steps} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | perplexity {perplexity:.3f} | {elapsed:.1f}s")
                if math.isfinite(val_loss) and val_loss < float(best_val_loss.numpy()):
                    best_val_loss.assign(val_loss)
                    best_path = best_manager.save(checkpoint_number=current_step)
                    log(f"New best validation loss; checkpoint saved: {best_path}")
                latest_manager.save(checkpoint_number=current_step)
                interval_losses.clear()
                interval_start = time.time()
    except KeyboardInterrupt:
        log("Training interrupted by user; saving the latest checkpoint.")
    finally:
        latest_manager.save(checkpoint_number=int(step.numpy()))
        log_file.close()
        log_txt.close()
        write_loss_plot(log_path, output_dir / "loss.png")

    sample = sample_text(
        model, tokenizer, special_ids, args.prompt, args.generate,
        args.temperature, args.top_k, args.block_size,
    )
    sample_path = output_dir / "sample.txt"
    sample_path.write_text(sample + "\n", encoding="utf-8")
    print("\nFinished. Artifacts:")
    for path in (tokenizer_path, output_dir / "config.json", log_path,
                 log_txt_path, output_dir / "loss.png", sample_path,
                 latest_manager.latest_checkpoint and Path(latest_manager.latest_checkpoint + ".index")):
        if path and Path(path).exists():
            print(" -", path)
    print(" - latest checkpoints:", checkpoint_dir / "latest")
    print(" - best checkpoints:", checkpoint_dir / "best")
    print("\nGenerated sample:\n", sample)


if __name__ == "__main__":
    main()
