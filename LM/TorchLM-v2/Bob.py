"""TorchLM-v2 model Bob: successor to the original PyTorch TorchLM.

Bob keeps TorchLM's byte-level BPE, RoPE, tied embeddings, and PyTorch training
while scaling it to a 16K vocabulary and roughly 15M parameters.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
except ImportError as exc:
    raise SystemExit("Install dependencies with: pip install -r requirements.txt") from exc

PAD_TOKEN = "<|pad|>"
BOS_TOKEN = "<|bos|>"
EOS_TOKEN = "<|eos|>"


@dataclass
class Config:
    vocab_size: int
    embed_dim: int = 384
    nhead: int = 6
    num_layers: int = 6
    ff_dim: int = 1024
    max_len: int = 512
    dropout: float = 0.1


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim % 2:
            raise ValueError("RoPE head dimension must be even")
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor):
        dim = q.size(-1)
        angles = positions.to(device=q.device, dtype=torch.float32)[:, None] * self.inv_freq[None, :]
        angles = torch.cat((angles, angles), dim=-1)[None, None, :, :]
        cos, sin = angles.cos().to(q.dtype), angles.sin().to(q.dtype)

        def rotate(x):
            paired = x.reshape(*x.shape[:-1], dim // 2, 2)
            even, odd = paired.unbind(-1)
            return torch.stack((-odd, even), dim=-1).flatten(-2)

        return q * cos + rotate(q) * sin, k * cos + rotate(k) * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, nhead: int, dropout: float):
        super().__init__()
        if embed_dim % nhead:
            raise ValueError("embed_dim must be divisible by nhead")
        self.nhead, self.head_dim = nhead, embed_dim // nhead
        if self.head_dim % 2:
            raise ValueError("attention head dimension must be even for RoPE")
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.rope = RotaryEmbedding(self.head_dim)
        self.dropout = dropout

    def forward(self, x, positions):
        batch, seq_len, channels = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.nhead, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = self.rope(q, k, positions)
        attended = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, seq_len, channels)
        return self.out_proj(attended)


class TransformerBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.attn = CausalSelfAttention(config.embed_dim, config.nhead, config.dropout)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, config.ff_dim), nn.GELU(),
            nn.Linear(config.ff_dim, config.embed_dim), nn.Dropout(config.dropout),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, positions):
        x = x + self.dropout(self.attn(self.ln1(x), positions))
        return x + self.mlp(self.ln2(x))


class Bob(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.final_norm = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, token_ids):
        _, seq_len = token_ids.shape
        if seq_len > self.config.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len={self.config.max_len}")
        positions = torch.arange(seq_len, device=token_ids.device)
        x = self.token_embedding(token_ids)
        for block in self.blocks:
            x = block(x, positions)
        return self.lm_head(self.final_norm(x))

    @torch.inference_mode()
    def generate(self, tokenizer, prompt, bos_id, eos_id, max_new_tokens=200,
                 temperature=0.8, top_k=40):
        if temperature <= 0 or max_new_tokens < 0:
            raise ValueError("temperature must be positive and max_new_tokens non-negative")
        device = next(self.parameters()).device
        ids = [bos_id] + tokenizer.encode(prompt, add_special_tokens=False).ids
        ids = ids[-self.config.max_len:]
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                context = torch.tensor([ids[-self.config.max_len:]], dtype=torch.long, device=device)
                logits = self(context)[0, -1].float() / temperature
                if top_k > 0:
                    k = min(top_k, logits.numel())
                    threshold = torch.topk(logits, k).values[-1]
                    logits = logits.masked_fill(logits < threshold, float("-inf"))
                next_id = int(torch.multinomial(F.softmax(logits, dim=-1), 1).item())
                if next_id == eos_id:
                    break
                ids.append(next_id)
        finally:
            self.train(was_training)
        return tokenizer.decode(ids, skip_special_tokens=True)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def iter_documents(path: Path) -> Iterator[str]:
    if path.suffix.lower() != ".jsonl":
        text = path.read_text(encoding="utf-8")
        if text.strip():
            yield text
        return
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            body = str(item.get("text", "")).strip()
            title = str(item.get("title", "")).strip()
            if body:
                yield f"{title}\n{body}" if title else body


def train_tokenizer(data_path: Path, tokenizer_path: Path, vocab_size: int):
    if vocab_size < 300:
        raise ValueError("Byte-level BPE vocab-size must be at least 300")
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]", byte_fallback=True))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.Sequence([decoders.ByteFallback(), decoders.ByteLevel()])
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["[UNK]", PAD_TOKEN, BOS_TOKEN, EOS_TOKEN],
    )
    tokenizer.train_from_iterator(iter_documents(data_path), trainer=trainer)
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tokenizer_path))
    return tokenizer


def prepare_token_files(data_path: Path, tokenizer: Tokenizer, output_dir: Path):
    train_path, val_path = output_dir / "train.bin", output_dir / "validation.bin"
    if train_path.exists() and val_path.exists():
        return train_path, val_path
    bos_id, eos_id = tokenizer.token_to_id(BOS_TOKEN), tokenizer.token_to_id(EOS_TOKEN)
    if bos_id is None or eos_id is None:
        raise ValueError("Tokenizer is missing BOS/EOS tokens")
    train_count = val_count = doc_count = 0
    with train_path.open("wb") as train_file, val_path.open("wb") as val_file:
        for doc_count, doc in enumerate(iter_documents(data_path), start=1):
            ids = np.asarray([bos_id, *tokenizer.encode(doc, add_special_tokens=False).ids, eos_id], dtype=np.uint32)
            target = val_file if doc_count % 20 == 0 else train_file
            ids.tofile(target)
            if doc_count % 20 == 0:
                val_count += ids.size
            else:
                train_count += ids.size
            if doc_count % 10000 == 0:
                print(f"Tokenized {doc_count:,} documents")
    if train_count <= 1 or val_count <= 1:
        raise ValueError("Corpus needs enough documents for train and validation splits")
    print(f"Token cache: {train_count:,} train tokens, {val_count:,} validation tokens")
    return train_path, val_path


def get_batch(data, batch_size, block_size, device):
    starts = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
    offsets = np.arange(block_size + 1)
    values = np.asarray(data[starts[:, None] + offsets[None, :]], dtype=np.int64)
    batch = torch.from_numpy(values.copy()).to(device, non_blocking=True)
    return batch[:, :-1], batch[:, 1:]


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, eval_batches, device):
    model.eval()
    results = []
    for data in (train_data, val_data):
        losses = []
        for _ in range(eval_batches):
            x, y = get_batch(data, batch_size, block_size, device)
            logits = model(x)
            losses.append(F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1)).item())
        results.append(sum(losses) / len(losses))
    model.train()
    return results


def save_checkpoint(path, model, optimizer, step, best_loss, tokenizer_path, data_path, precision):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_name": "Bob",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": asdict(model.config),
        "step": step,
        "best_val_loss": best_loss,
        "tokenizer_path": str(tokenizer_path),
        "data_path": str(data_path),
        "precision": precision,
    }, path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/TorchLM-v2-Bob"))
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--prepare-tokenizer", action="store_true")
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--embed-dim", type=int, default=384)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--ff-dim", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--best-checkpoint", type=Path, default=None)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--prompt", default="Indonesia adalah")
    parser.add_argument("--generate", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(f"Corpus not found: {args.data}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.require_gpu and device.type != "cuda":
        raise RuntimeError("No CUDA GPU found; enable a GPU runtime or omit --require-gpu")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = args.tokenizer or args.output_dir / "tokenizer.json"
    if args.prepare_tokenizer or not tokenizer_path.exists():
        tokenizer = train_tokenizer(args.data, tokenizer_path, args.vocab_size)
        print(f"Tokenizer saved: {tokenizer_path} ({tokenizer.get_vocab_size():,} tokens)")
        if args.prepare_tokenizer:
            return
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        tokenizer.add_special_tokens([PAD_TOKEN, BOS_TOKEN, EOS_TOKEN])

    train_path, val_path = prepare_token_files(args.data, tokenizer, args.output_dir)
    train_data = np.memmap(train_path, dtype=np.uint32, mode="r")
    val_data = np.memmap(val_path, dtype=np.uint32, mode="r")

    config = Config(vocab_size=tokenizer.get_vocab_size(), embed_dim=args.embed_dim,
                    nhead=args.num_heads, num_layers=args.num_layers, ff_dim=args.ff_dim,
                    max_len=args.block_size, dropout=args.dropout)
    (args.output_dir / "config.json").write_text(
        json.dumps({**asdict(config), "model_name": "Bob"}, indent=2), encoding="utf-8"
    )
    model = Bob(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)

    print(f"PyTorch {torch.__version__} | Device: {device}")
    print(f"Model: Bob | Parameters: {count_parameters(model):,} ({count_parameters(model)/1e6:.2f}M)")
    print(f"Vocab: {config.vocab_size:,}")

    checkpoint_path = args.checkpoint or args.output_dir / "bob_latest.pt"
    best_path = args.best_checkpoint or args.output_dir / "bob_best.pt"
    log_path = args.output_dir / "training.csv"
    fresh_log = not log_path.exists() or log_path.stat().st_size == 0

    amp_dtype = None
    if device.type == "cuda":
        if args.precision == "bf16":
            amp_dtype = torch.bfloat16
        elif args.precision == "fp16":
            amp_dtype = torch.float16
        elif args.precision == "auto":
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    start_step, best_loss = 0, float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("model_config") != asdict(config):
            raise ValueError("Checkpoint architecture/tokenizer config differs from this run")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = int(checkpoint.get("step", 0))
        best_loss = float(checkpoint.get("best_val_loss", float("inf")))

    with log_path.open("a", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        if fresh_log:
            writer.writerow(["step", "train_loss", "validation_loss", "perplexity", "seconds"])
        for step in range(start_step + 1, args.steps + 1):
            model.train()
            x, y = get_batch(train_data, args.batch_size, args.block_size, device)
            optimizer.zero_grad(set_to_none=True)
            amp = torch.autocast("cuda", dtype=amp_dtype) if amp_dtype else contextlib.nullcontext()
            with amp:
                logits = model(x)
                loss = F.cross_entropy(logits.float().reshape(-1, config.vocab_size), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step == 1 or step % args.eval_interval == 0 or step == args.steps:
                started = time.time()
                train_loss, val_loss = estimate_loss(model, train_data, val_data,
                                                     args.batch_size, args.block_size,
                                                     args.eval_batches, device)
                perplexity = math.exp(min(val_loss, 80.0))
                elapsed = time.time() - started
                print(f"Step {step}/{args.steps} | train {train_loss:.4f} | val {val_loss:.4f} | ppl {perplexity:.2f}")
                writer.writerow([step, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{perplexity:.6g}", f"{elapsed:.2f}"])
                log_file.flush()
                save_checkpoint(checkpoint_path, model, optimizer, step, best_loss,
                                tokenizer_path, args.data, str(amp_dtype) if amp_dtype else "fp32")
                if val_loss < best_loss:
                    best_loss = val_loss
                    save_checkpoint(best_path, model, optimizer, step, best_loss,
                                    tokenizer_path, args.data, str(amp_dtype) if amp_dtype else "fp32")

    sample = model.generate(tokenizer, args.prompt, tokenizer.token_to_id(BOS_TOKEN),
                            tokenizer.token_to_id(EOS_TOKEN), args.generate,
                            args.temperature, args.top_k)
    (args.output_dir / "sample.txt").write_text(sample + "\n", encoding="utf-8")
    print(sample)


if __name__ == "__main__":
    main()
