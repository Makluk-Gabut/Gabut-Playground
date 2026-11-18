"""Generic small Transformer LM with byte-level BPE, tied weights, and CUDA AMP.

Tokenizer preparation only:
    python torchlm.py --data corpus.txt --prepare-tokenizer --tokenizer tokenizer.json --vocab-size 2000

Training:
    python torchlm.py --data corpus.txt --tokenizer tokenizer.json
"""
from __future__ import annotations

import argparse
import contextlib
import math
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
except ImportError as exc:
    raise SystemExit("Missing dependency: pip install tokenizers") from exc


class RotaryEmbedding(nn.Module):
    """Rotary positional embedding; registered buffer follows model device."""
    def __init__(self, dim: int) -> None:
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
    def __init__(self, embed_dim: int, nhead: int, dropout: float) -> None:
        super().__init__()
        if embed_dim % nhead:
            raise ValueError("embed_dim must be divisible by nhead")
        self.nhead, self.head_dim = nhead, embed_dim // nhead
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.rope = RotaryEmbedding(self.head_dim)
        self.dropout = dropout

    def forward(self, x, positions, past_kv=None, use_cache=False):
        b, t, c = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.nhead, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = self.rope(q, k, positions)
        cached = past_kv is not None
        if cached:
            k = torch.cat((past_kv[0], k), dim=-2)
            v = torch.cat((past_kv[1], v), dim=-2)
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=not cached,
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.out_proj(y), (k, v) if use_cache else None


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, nhead, ff_dim, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, nhead, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, ff_dim), nn.GELU(), nn.Linear(ff_dim, embed_dim), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, positions, past_kv=None, use_cache=False):
        out, kv = self.attn(self.ln1(x), positions, past_kv, use_cache)
        x = x + self.dropout(out)
        return x + self.mlp(self.ln2(x)), kv


class NanoTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, nhead=4, num_layers=10, ff_dim=512, max_len=128, dropout=0.1):
        super().__init__()
        self.max_len = max_len
        self.embed_dim, self.nhead = embed_dim, nhead
        self.num_layers, self.ff_dim = num_layers, ff_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, nhead, ff_dim, dropout) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        # Weight tying: input token embeddings and output projection share one Parameter.
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, token_ids):
        _, seq_len = token_ids.shape
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len={self.max_len}")
        positions = torch.arange(seq_len, device=token_ids.device)
        x = self.token_embedding(token_ids)
        for block in self.blocks:
            x, _ = block(x, positions)
        return self.lm_head(self.final_norm(x))

    def _prefill(self, ids):
        positions = torch.arange(ids.size(1), device=ids.device)
        x = self.token_embedding(ids)
        cache = []
        for block in self.blocks:
            x, kv = block(x, positions, use_cache=True)
            cache.append(kv)
        return self.lm_head(self.final_norm(x)), cache

    def _decode_one(self, token_id, absolute_position, cache):
        # Keep a sliding context of max_len tokens, while RoPE positions remain absolute.
        positions = torch.tensor([absolute_position], device=token_id.device)
        x = self.token_embedding(token_id)
        updated = []
        for block, past in zip(self.blocks, cache):
            past = (past[0][..., -(self.max_len - 1):, :], past[1][..., -(self.max_len - 1):, :])
            x, kv = block(x, positions, past_kv=past, use_cache=True)
            updated.append(kv)
        return self.lm_head(self.final_norm(x)), updated

    @torch.inference_mode()
    def generate(self, tokenizer, prompt, max_new_tokens=300, temperature=0.8, top_k=40):
        if temperature <= 0 or max_new_tokens < 0:
            raise ValueError("temperature must be positive and max_new_tokens non-negative")
        device = next(self.parameters()).device
        prompt_ids = tokenizer.encode(prompt).ids or [0]
        prompt_ids = prompt_ids[-self.max_len:]
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        was_training = self.training
        self.eval()
        try:
            logits, cache = self._prefill(ids)
            next_logits = logits[:, -1]
            generated = list(prompt_ids)
            for i in range(max_new_tokens):
                scores = next_logits / temperature
                if top_k > 0:
                    k = min(top_k, scores.size(-1))
                    threshold = torch.topk(scores, k).values[:, -1, None]
                    scores = scores.masked_fill(scores < threshold, float("-inf"))
                token = torch.multinomial(F.softmax(scores.float(), dim=-1), 1)
                generated.append(int(token.item()))
                if i + 1 < max_new_tokens:
                    next_logits, cache = self._decode_one(token, len(prompt_ids) + i, cache)
                    next_logits = next_logits[:, -1]
        finally:
            self.train(was_training)
        return tokenizer.decode(generated, skip_special_tokens=True)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_tokenizer(text, path: Path, vocab_size: int):
    if vocab_size < 300:
        raise ValueError("vocab-size must be at least 300 for byte-level BPE")
    tok = Tokenizer(models.BPE(unk_token="[UNK]", byte_fallback=True))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.Sequence([decoders.ByteFallback(), decoders.ByteLevel()])
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["[UNK]", "<|eos|>"],
    )
    tok.train_from_iterator([text], trainer=trainer)
    path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(path))
    return tok


def get_batch(data, batch_size, block_size):
    starts = torch.randint(0, data.numel() - block_size, (batch_size,), device=data.device)
    x = torch.stack([data[i:i + block_size] for i in starts])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, eval_batches):
    model.eval()
    result = []
    for data in (train_data, val_data):
        losses = []
        for _ in range(eval_batches):
            x, y = get_batch(data, batch_size, block_size)
            logits = model(x)
            losses.append(F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1)).item())
        result.append(sum(losses) / len(losses))
    model.train()
    return result


def choose_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def choose_precision(name, device):
    if device.type != "cuda" or name == "fp32":
        return None, "fp32"
    if name == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise ValueError("CUDA device does not support bf16")
        return torch.bfloat16, "bf16"
    if name == "fp16":
        return torch.float16, "fp16"
    return (torch.bfloat16, "bf16") if torch.cuda.is_bf16_supported() else (torch.float16, "fp16")


def save_checkpoint(path, model, optimizer, step, best_val_loss, tokenizer_path, data_path, precision):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": {
            "vocab_size": model.lm_head.out_features,
            "embed_dim": model.embed_dim,
            "nhead": model.nhead,
            "num_layers": model.num_layers,
            "ff_dim": model.ff_dim,
            "max_len": model.max_len,
        },
        "step": step,
        "best_val_loss": best_val_loss,
        "tokenizer_path": str(tokenizer_path),
        "data_path": str(data_path),
        "precision": precision,
    }, path)


def main():
    p = argparse.ArgumentParser(description="Train a compact Transformer language model on a UTF-8 corpus.")
    p.add_argument("--data", type=Path, required=True, help="UTF-8 corpus file")
    p.add_argument("--tokenizer", type=Path, default=Path("nanolm_tokenizer.json"))
    p.add_argument("--vocab-size", type=int, default=2000)
    p.add_argument("--prepare-tokenizer", action="store_true", help="Build tokenizer only; do not train model")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    p.add_argument("--device", default="auto")
    p.add_argument("--checkpoint", type=Path, default=Path("nanolm_checkpoint.pt"))
    p.add_argument("--best-checkpoint", type=Path, default=Path("nanolm_best.pt"))
    p.add_argument("--resume", type=Path)
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--generate", type=int, default=200)
    args = p.parse_args()

    text = args.data.read_text(encoding="utf-8")
    if args.prepare_tokenizer:
        tokenizer = train_tokenizer(text, args.tokenizer, args.vocab_size)
        sample = text[:2000]
        ids = tokenizer.encode(sample).ids
        print(f"Tokenizer saved: {args.tokenizer}")
        print(f"Vocabulary size: {tokenizer.get_vocab_size():,}")
        print(f"Sample characters/tokens: {len(sample):,}/{len(ids):,}")
        print(f"Round-trip matches: {tokenizer.decode(ids) == sample}")
        print("Model training was not started.")
        return

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    eos_id = tokenizer.token_to_id("<|eos|>")
    ids = tokenizer.encode(text).ids
    if eos_id is not None:
        ids.append(eos_id)
    if len(ids) <= args.block_size + 1:
        p.error("Tokenized corpus is too short for the selected block size")

    device = choose_device(args.device)
    amp_dtype, precision = choose_precision(args.precision, device)
    data = torch.tensor(ids, dtype=torch.long, device=device)
    split = int(0.9 * len(ids))
    train_data, val_data = data[:split], data[split:]
    if val_data.numel() <= args.block_size:
        p.error("Validation portion is too short for block size")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        cfg = checkpoint["model_config"]
        if cfg["vocab_size"] != tokenizer.get_vocab_size():
            p.error("Checkpoint vocabulary does not match tokenizer vocabulary")
        model = NanoTransformer(**cfg).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_step = checkpoint.get("step", 0)
        best = checkpoint.get("best_val_loss", float("inf"))
    else:
        model = NanoTransformer(tokenizer.get_vocab_size(), max_len=args.block_size).to(device)
        start_step, best = 0, float("inf")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
    if args.resume:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and amp_dtype == torch.float16))
    print(f"Device: {device} | Precision: {precision} | Tokens: {len(ids):,} | Vocab: {tokenizer.get_vocab_size():,} | Parameters: {count_parameters(model):,}")
    print(f"Embedding/output weights tied: {model.token_embedding.weight is model.lm_head.weight}")

    for step in range(start_step + 1, args.steps + 1):
        model.train()
        x, y = get_batch(train_data, args.batch_size, args.block_size)
        amp = torch.autocast("cuda", dtype=amp_dtype) if device.type == "cuda" and amp_dtype else contextlib.nullcontext()
        with amp:
            logits = model(x)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % args.eval_interval == 0 or step == args.steps:
            train_loss, val_loss = estimate_loss(model, train_data, val_data, args.batch_size, args.block_size, args.eval_batches)
            print(f"Step {step}/{args.steps} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | val perplexity {math.exp(min(val_loss, 20)):.2f}")
            if val_loss < best:
                best = val_loss
                save_checkpoint(args.best_checkpoint, model, optimizer, step, best, args.tokenizer, args.data, precision)
                print(f"Best checkpoint saved: {args.best_checkpoint}")
        if step % 500 == 0:
            save_checkpoint(args.checkpoint, model, optimizer, step, best, args.tokenizer, args.data, precision)
    save_checkpoint(args.checkpoint, model, optimizer, args.steps, best, args.tokenizer, args.data, precision)
    print(f"Final checkpoint saved: {args.checkpoint}")
    print(model.generate(tokenizer, args.prompt, args.generate))


if __name__ == "__main__":
    main()
