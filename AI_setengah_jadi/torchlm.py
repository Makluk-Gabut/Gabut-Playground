"""NanoLM: a small Shakespeare language model using BPE + byte fallback.

Install dependencies:
    pip install torch tokenizers

Train the tokenizer only (does NOT train the language model):
    python torchlm.py --data corpus.txt --prepare-tokenizer \
        --tokenizer nanolm_tokenizer.json --vocab-size 8000

Train the model afterward:
    python torchlm.py --data corpus.txt --tokenizer nanolm_tokenizer.json

The tokenizer is byte-level BPE with byte fallback. It uses a 256-byte
initial alphabet and merges frequent byte sequences into larger tokens.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import random
import urllib.request
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
except ImportError as exc:
    raise SystemExit("Missing dependency. Install it with: pip install tokenizers") from exc


SHAKESPEARE_URL = "https://www.gutenberg.org/cache/epub/100/pg100.txt"


def train_bpe_tokenizer(text: str, tokenizer_path: Path, vocab_size: int) -> Tokenizer:
    """Train byte-level BPE with byte fallback and save the tokenizer JSON."""
    if vocab_size < 300:
        raise ValueError("BPE vocab size must be at least 300 to fit byte alphabet and special tokens")
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]", byte_fallback=True))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.Sequence([decoders.ByteFallback(), decoders.ByteLevel()])
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["[UNK]"],
    )
    tokenizer.train_from_iterator([text], trainer=trainer)
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tokenizer_path))
    return tokenizer


def load_tokenizer(tokenizer_path: Path) -> Tokenizer:
    if not tokenizer_path.is_file():
        raise FileNotFoundError(
            f"Tokenizer not found: {tokenizer_path}. Run with --prepare-tokenizer first."
        )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size() < 257:
        raise ValueError("Tokenizer vocabulary is too small; expected byte-level BPE with byte fallback")
    return tokenizer


def apply_rope(q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dim = q.size(-1)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=q.device, dtype=torch.float32) / dim))
    angles = positions.to(torch.float32)[:, None] * inv_freq[None, :]
    angles = torch.cat((angles, angles), dim=-1)[None, None, :, :]
    cos, sin = angles.cos().to(q.dtype), angles.sin().to(q.dtype)

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        paired = x.reshape(*x.shape[:-1], dim // 2, 2)
        even, odd = paired.unbind(dim=-1)
        return torch.stack((-odd, even), dim=-1).flatten(-2)

    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, nhead: int, dropout: float) -> None:
        super().__init__()
        if embed_dim % nhead or (embed_dim // nhead) % 2:
            raise ValueError("embed_dim must divide evenly by nhead and head_dim must be even for RoPE")
        self.nhead = nhead
        self.head_dim = embed_dim // nhead
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout

    def forward(self, x, positions, past_kv=None, use_cache=False):
        batch, time, channels = x.shape
        qkv = self.qkv(x).view(batch, time, 3, self.nhead, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = apply_rope(q, k, positions)
        has_past = past_kv is not None
        if has_past:
            k = torch.cat((past_kv[0], k), dim=-2)
            v = torch.cat((past_kv[1], v), dim=-2)
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=not has_past,
        )
        y = y.transpose(1, 2).contiguous().view(batch, time, channels)
        new_kv = (k, v) if use_cache else None
        return self.out_proj(y), new_kv


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, nhead: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, nhead, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.GELU(), nn.Linear(ff_dim, embed_dim), nn.Dropout(dropout)
        )
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, positions, past_kv=None, use_cache=False):
        attn_out, kv = self.attn(self.ln1(x), positions, past_kv, use_cache)
        x = x + self.resid_dropout(attn_out)
        return x + self.mlp(self.ln2(x)), kv


class NanoTransformer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim=128, nhead=4, num_layers=10, ff_dim=512, max_len=128, dropout=0.1):
        super().__init__()
        self.max_len = max_len
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, nhead, ff_dim, dropout) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        _, seq_len = token_ids.shape
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len={self.max_len}")
        positions = torch.arange(seq_len, device=token_ids.device)
        x = self.token_embedding(token_ids)
        for block in self.blocks:
            x, _ = block(x, positions)
        return self.lm_head(self.final_norm(x))

    def _prefill(self, token_ids):
        positions = torch.arange(token_ids.size(1), device=token_ids.device)
        x = self.token_embedding(token_ids)
        cache = []
        for block in self.blocks:
            x, kv = block(x, positions, use_cache=True)
            cache.append(kv)
        return self.lm_head(self.final_norm(x)), cache

    def _decode_one(self, token_id, position, cache):
        positions = torch.tensor([position], device=token_id.device)
        x = self.token_embedding(token_id)
        next_cache = []
        for block, past_kv in zip(self.blocks, cache):
            x, kv = block(x, positions, past_kv, use_cache=True)
            # Keep the latest context window. RoPE positions already embedded in cached keys.
            next_cache.append((kv[0][..., -self.max_len :, :], kv[1][..., -self.max_len :, :]))
        return self.lm_head(self.final_norm(x)), next_cache

    @torch.inference_mode()
    def generate(self, tokenizer: Tokenizer, prompt: str, max_new_tokens=300, temperature=0.8, top_k=40) -> str:
        if temperature <= 0 or max_new_tokens < 0:
            raise ValueError("temperature must be positive and max_new_tokens non-negative")
        device = next(self.parameters()).device
        prompt_ids = tokenizer.encode(prompt).ids or [0]
        prompt_ids = prompt_ids[-self.max_len :]
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        was_training = self.training
        self.eval()
        try:
            logits, cache = self._prefill(ids)
            generated = list(prompt_ids)
            next_logits = logits[:, -1, :]
            for step in range(max_new_tokens):
                scores = next_logits / temperature
                if top_k > 0:
                    k = min(top_k, scores.size(-1))
                    threshold = torch.topk(scores, k).values[:, -1, None]
                    scores = scores.masked_fill(scores < threshold, float("-inf"))
                next_id = torch.multinomial(F.softmax(scores.float(), dim=-1), num_samples=1)
                token_id = int(next_id.item())
                generated.append(token_id)
                if step + 1 < max_new_tokens:
                    position = min(len(prompt_ids) + step, self.max_len - 1)
                    next_logits_all, cache = self._decode_one(next_id, position, cache)
                    next_logits = next_logits_all[:, -1, :]
        finally:
            self.train(was_training)
        return tokenizer.decode(generated, skip_special_tokens=True)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_batch(data: torch.Tensor, batch_size: int, block_size: int):
    starts = torch.randint(0, data.numel() - block_size, (batch_size,), device=data.device)
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size, eval_batches):
    model.eval()
    results = []
    for data in (train_data, val_data):
        losses = []
        for _ in range(eval_batches):
            x, y = get_batch(data, batch_size, block_size)
            logits = model(x)
            losses.append(F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1)).item())
        results.append(sum(losses) / len(losses))
    model.train()
    return results


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def precision_settings(requested: str, device: torch.device):
    if device.type != "cuda":
        return None, "fp32"
    if requested == "fp32":
        return None, "fp32"
    if requested == "fp16":
        return torch.float16, "fp16"
    if requested == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise ValueError("GPU does not support bf16; use --precision fp16 or auto")
        return torch.bfloat16, "bf16"
    if requested == "auto":
        return (torch.bfloat16, "bf16") if torch.cuda.is_bf16_supported() else (torch.float16, "fp16")
    raise ValueError(f"Unknown precision: {requested}")


def load_text(data_path: Path, download_shakespeare: bool) -> str:
    if download_shakespeare:
        request = urllib.request.Request(SHAKESPEARE_URL, headers={"User-Agent": "NanoLM educational experiment"})
        with urllib.request.urlopen(request, timeout=30) as response:
            text = response.read().decode("utf-8", errors="replace")
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_text(text, encoding="utf-8")
        return text
    if not data_path.is_file():
        raise FileNotFoundError(f"Training text not found: {data_path}")
    return data_path.read_text(encoding="utf-8", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a BPE tokenizer or train NanoLM.")
    parser.add_argument("--data", type=Path, default=Path("corpus.txt"), help="UTF-8 training text")
    parser.add_argument("--tokenizer", type=Path, default=Path("nanolm_tokenizer.json"))
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--prepare-tokenizer", action="store_true", help="Train/save tokenizer only; do not train LM")
    parser.add_argument("--download-shakespeare", action="store_true")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--checkpoint", type=Path, default=Path("nanolm_shakespeare.pt"))
    parser.add_argument("--prompt", default="To be, or not to be:")
    parser.add_argument("--generate", type=int, default=300)
    args = parser.parse_args()

    if args.vocab_size < 300:
        parser.error("--vocab-size must be at least 300 for byte-level BPE")
    if args.prepare_tokenizer:
        text = load_text(args.data, args.download_shakespeare)
        tokenizer = train_bpe_tokenizer(text, args.tokenizer, args.vocab_size)
        sample = text[:500]
        encoded_sample = tokenizer.encode(sample).ids
        print(f"Tokenizer saved: {args.tokenizer}")
        print(f"Vocabulary size: {tokenizer.get_vocab_size():,}")
        print(f"Sample chars: {len(sample):,} | sample tokens: {len(encoded_sample):,}")
        print(f"Round-trip sample matches: {tokenizer.decode(encoded_sample) == sample}")
        print("Language model training was not started.")
        return

    if args.steps < 1 or args.batch_size < 1 or args.block_size < 1 or args.eval_batches < 1 or args.eval_interval < 1:
        parser.error("steps, batch-size, block-size, eval-interval, and eval-batches must be positive")

    text = load_text(args.data, args.download_shakespeare)
    tokenizer = load_tokenizer(args.tokenizer)
    encoded = tokenizer.encode(text).ids
    if len(encoded) < args.block_size + 2:
        parser.error("Tokenized corpus is too short for the selected block size")
    data = torch.tensor(encoded, dtype=torch.long)
    split = int(0.9 * len(data))
    device = choose_device(args.device)
    train_data, val_data = data[:split].to(device), data[split:].to(device)
    if val_data.numel() <= args.block_size:
        parser.error("Validation split is too short for the selected block size")

    vocab_size = tokenizer.get_vocab_size()
    model = NanoTransformer(vocab_size=vocab_size, max_len=args.block_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
    amp_dtype, precision_name = precision_settings(args.precision, device)
    use_amp = device.type == "cuda" and amp_dtype is not None
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
    print(f"Device: {device} | Precision: {precision_name} | Tokens: {len(data):,} | Vocab: {vocab_size:,} | Params: {count_parameters(model):,}")

    for step in range(1, args.steps + 1):
        model.train()
        x, y = get_batch(train_data, args.batch_size, args.block_size)
        autocast = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else contextlib.nullcontext()
        with autocast:
            logits = model(x)
            loss = F.cross_entropy(logits.float().reshape(-1, vocab_size), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % args.eval_interval == 0 or step == args.steps:
            train_loss, val_loss = estimate_loss(model, train_data, val_data, args.batch_size, args.block_size, args.eval_batches)
            print(f"Step {step:>5}/{args.steps} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | val perplexity {math.exp(min(val_loss, 20)):.2f}")

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {"vocab_size": vocab_size, "embed_dim": 128, "nhead": 4, "num_layers": 10,
                         "ff_dim": 512, "max_len": args.block_size, "position_encoding": "RoPE"},
        "tokenizer_path": str(args.tokenizer),
        "training_data": str(args.data),
        "precision": precision_name,
    }, args.checkpoint)
    print(f"Checkpoint saved to {args.checkpoint}")
    print("\n--- Sample ---")
    print(model.generate(tokenizer, args.prompt, max_new_tokens=args.generate))


if __name__ == "__main__":
    main()
