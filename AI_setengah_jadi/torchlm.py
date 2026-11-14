"""NanoLM: a small byte-level/BPE language model trainer.

This file is intentionally generic: it can train on any UTF-8 text corpus,
not only Shakespeare. It supports byte-level BPE with byte fallback, mixed
precision on CUDA, a simple KV-cache generation path, and a small checkpoint
resume workflow.

Common usage:
    pip install tokenizers

    # Build tokenizer for your corpus only
    python torchlm.py --data /path/to/corpus.txt --prepare-tokenizer \
        --tokenizer /path/to/tokenizer.json --vocab-size 8000

    # Train the model
    python torchlm.py --data /path/to/corpus.txt --tokenizer /path/to/tokenizer.json \
        --steps 3000 --batch-size 32 --block-size 128

    # Resume a previous checkpoint
    python torchlm.py --data /path/to/corpus.txt --tokenizer /path/to/tokenizer.json \
        --resume /path/to/checkpoint.pt --steps 5000
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
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: pip install tokenizers") from exc


LEGACY_SHAKESPEARE_URL = "https://www.gutenberg.org/cache/epub/100/pg100.txt"


class ByteTokenizer:
    """Fallback tokenizer for raw UTF-8 bytes. This keeps compatibility with older scripts."""

    vocab_size = 256

    @staticmethod
    def encode(text: str) -> list[int]:
        return list(text.encode("utf-8"))

    @staticmethod
    def decode(tokens: list[int]) -> str:
        return bytes(tokens).decode("utf-8")


class RotaryEmbedding:
    def __init__(self, dim: int, device: torch.device) -> None:
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def apply(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dim = q.size(-1)
        angles = positions.to(torch.float32)[:, None] * self.inv_freq[None, :]
        angles = torch.cat((angles, angles), dim=-1)[None, None, :, :]
        cos = angles.cos().to(dtype=q.dtype)
        sin = angles.sin().to(dtype=q.dtype)

        def rotate_half(x: torch.Tensor) -> torch.Tensor:
            paired = x.reshape(*x.shape[:-1], dim // 2, 2)
            even, odd = paired.unbind(dim=-1)
            return torch.stack((-odd, even), dim=-1).flatten(-2)

        return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, nhead: int, dropout: float, device: Optional[torch.device] = None) -> None:
        super().__init__()
        if embed_dim % nhead != 0:
            raise ValueError("embed_dim must be divisible by nhead")
        head_dim = embed_dim // nhead
        if head_dim % 2:
            raise ValueError("head_dim must be even for rotary embeddings")
        self.nhead = nhead
        self.head_dim = head_dim
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout
        self.rotary = RotaryEmbedding(head_dim, device or torch.device("cpu"))

    def _reshape_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, time, channels = x.shape
        qkv = self.qkv(x).view(batch, time, 3, self.nhead, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        q, k, v = self._reshape_qkv(x)
        q, k = self.rotary.apply(q, k, positions)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=not (past_kv is not None),
        )
        y = y.transpose(1, 2).contiguous().view(x.size(0), x.size(1), x.size(2))
        y = self.out_proj(y)
        return y, (k, v) if use_cache else None


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, nhead: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, nhead, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, kv = self.attn(self.ln1(x), positions, past_kv, use_cache)
        x = x + self.resid_dropout(attn_out)
        x = x + self.mlp(self.ln2(x))
        return x, kv


class NanoTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        nhead: int = 4,
        num_layers: int = 10,
        ff_dim: int = 512,
        max_len: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        self.max_len = max_len
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, nhead, ff_dim, dropout) for _ in range(num_layers)]
        )
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

    def _prefill(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, list[Optional[tuple[torch.Tensor, torch.Tensor]]]]:
        seq_len = token_ids.size(1)
        positions = torch.arange(seq_len, device=token_ids.device)
        x = self.token_embedding(token_ids)
        cache: list[Optional[tuple[torch.Tensor, torch.Tensor]]] = []
        for block in self.blocks:
            x, kv = block(x, positions, use_cache=True)
            cache.append(kv)
        return self.lm_head(self.final_norm(x)), cache

    def _decode_one(
        self,
        token_id: torch.Tensor,
        position: int,
        cache: list[Optional[tuple[torch.Tensor, torch.Tensor]]],
    ) -> tuple[torch.Tensor, list[Optional[tuple[torch.Tensor, torch.Tensor]]]]:
        positions = torch.tensor([position], device=token_id.device)
        x = self.token_embedding(token_id)
        new_cache: list[Optional[tuple[torch.Tensor, torch.Tensor]]] = []
        for block, past_kv in zip(self.blocks, cache):
            assert past_kv is not None
            x, kv = block(x, positions, past_kv=past_kv, use_cache=True)
            new_cache.append(kv)
        return self.lm_head(self.final_norm(x)), new_cache

    @torch.inference_mode()
    def generate(
        self,
        tokenizer: Tokenizer,
        prompt: str,
        max_new_tokens: int = 300,
        temperature: float = 0.8,
        top_k: int = 40,
    ) -> str:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        device = next(self.parameters()).device
        prompt_ids = tokenizer.encode(prompt).ids or [0]
        prompt_ids = prompt_ids[-self.max_len :]
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        was_training = self.training
        self.eval()
        try:
            logits, cache = self._prefill(ids)
            next_logits = logits[:, -1, :]
            generated = list(prompt_ids)
            for _ in range(max_new_tokens):
                scaled = next_logits / temperature
                if top_k > 0:
                    k = min(top_k, scaled.size(-1))
                    threshold = torch.topk(scaled, k).values[:, -1, None]
                    scaled = scaled.masked_fill(scaled < threshold, float("-inf"))
                probs = F.softmax(scaled.float(), dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                token_id = int(next_id.item())
                generated.append(token_id)
                if len(generated) < self.max_len:
                    position = len(generated) - 1
                else:
                    position = self.max_len - 1
                next_logits, cache = self._decode_one(next_id, position, cache)
                next_logits = next_logits[:, -1, :]
        finally:
            self.train(was_training)

        return tokenizer.decode(generated, skip_special_tokens=False)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_text(data_path: Path, download_shakespeare: bool = False) -> str:
    if download_shakespeare:
        request = urllib.request.Request(
            LEGACY_SHAKESPEARE_URL,
            headers={"User-Agent": "NanoLM educational experiment"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            text = response.read().decode("utf-8", errors="replace")
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_text(text, encoding="utf-8")
        return text

    if not data_path.is_file():
        raise FileNotFoundError(
            f"Training text not found: {data_path}. Provide a corpus file or use --download-shakespeare for the legacy example."
        )
    return data_path.read_text(encoding="utf-8", errors="replace")


def train_bpe_tokenizer(text: str, tokenizer_path: Path, vocab_size: int) -> Tokenizer:
    if vocab_size < 300:
        raise ValueError("--vocab-size must be at least 300 to hold byte alphabet plus BPE merges")

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]", byte_fallback=True))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.Sequence([decoders.ByteFallback(), decoders.ByteLevel()])
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["[UNK]", "<|eos|>", "<|pad|>"],
    )
    tokenizer.train_from_iterator([text], trainer=trainer)
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tokenizer_path))
    return tokenizer


def load_tokenizer(tokenizer_path: Path) -> Tokenizer:
    if not tokenizer_path.is_file():
        raise FileNotFoundError(
            f"Tokenizer file not found: {tokenizer_path}. Run with --prepare-tokenizer first."
        )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size() < 300:
        raise ValueError("Tokenizer vocabulary is too small; it should include byte fallback and merges")
    return tokenizer


def add_eos_and_pad(tokens: list[int], eos_id: int, pad_id: int) -> list[int]:
    out = list(tokens)
    if out and out[-1] != eos_id:
        out.append(eos_id)
    while len(out) % 8:
        out.append(pad_id)
    return out


def encode_text(tokenizer: Tokenizer, text: str) -> list[int]:
    eos_id = tokenizer.token_to_id("<|eos|>")
    pad_id = tokenizer.token_to_id("<|pad|>")
    if eos_id is None or pad_id is None:
        return tokenizer.encode(text).ids
    input_ids = tokenizer.encode(text).ids
    return add_eos_and_pad(input_ids, eos_id, pad_id)


def get_batch(data: torch.Tensor, batch_size: int, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, data.numel() - block_size, (batch_size,), device=data.device)
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x, y


@torch.no_grad()
def estimate_loss(
    model: NanoTransformer,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    batch_size: int,
    block_size: int,
    eval_batches: int,
) -> tuple[float, float]:
    model.eval()
    losses = []
    for data in (train_data, val_data):
        batch_losses = []
        for _ in range(eval_batches):
            x, y = get_batch(data, batch_size, block_size)
            logits = model(x)
            batch_losses.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item())
        losses.append(sum(batch_losses) / len(batch_losses))
    model.train()
    return losses[0], losses[1]


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def choose_precision(requested: str, device: torch.device) -> tuple[Optional[torch.dtype], str]:
    if device.type != "cuda":
        if requested not in ("auto", "fp32"):
            print(f"Ignoring {requested} precision because device is not CUDA; using fp32.")
        return None, "fp32"
    if requested == "fp32":
        return None, "fp32"
    if requested == "fp16":
        return torch.float16, "fp16"
    if requested == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise ValueError("CUDA device does not support bf16. Use --precision fp16 or fp32.")
        return torch.bfloat16, "bf16"
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, "bf16"
    return torch.float16, "fp16"


def save_checkpoint(path: Path, model: nn.Module, tokenizer_path: Path, step: int, data_path: Path, precision_name: str, best_val_loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "vocab_size": model.lm_head.out_features,
                "embed_dim": 128,
                "nhead": 4,
                "num_layers": 10,
                "ff_dim": 512,
                "max_len": model.max_len,
                "position_encoding": "RoPE",
            },
            "step": step,
            "best_val_loss": best_val_loss,
            "tokenizer_path": str(tokenizer_path),
            "data_path": str(data_path),
            "precision": precision_name,
        },
        path,
    )


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[dict, NanoTransformer]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_cfg = checkpoint["model_config"]
    model = NanoTransformer(
        vocab_size=model_cfg["vocab_size"],
        embed_dim=model_cfg.get("embed_dim", 128),
        nhead=model_cfg.get("nhead", 4),
        num_layers=model_cfg.get("num_layers", 10),
        ff_dim=model_cfg.get("ff_dim", 512),
        max_len=model_cfg.get("max_len", 128),
        dropout=0.1,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint, model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small transformer language model on any UTF-8 corpus.")
    parser.add_argument("--data", type=Path, default=Path("corpus.txt"), help="UTF-8 corpus to train on")
    parser.add_argument("--tokenizer", type=Path, default=Path("nanolm_tokenizer.json"), help="Path for the trained tokenizer")
    parser.add_argument("--vocab-size", type=int, default=8000, help="BPE vocabulary size")
    parser.add_argument("--prepare-tokenizer", action="store_true", help="Train and save tokenizer only")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--checkpoint", type=Path, default=Path("nanolm_checkpoint.pt"))
    parser.add_argument("--best-checkpoint", type=Path, default=Path("nanolm_best.pt"), help="Best validation-loss checkpoint")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from a saved checkpoint")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--generate", type=int, default=300)
    parser.add_argument("--download-shakespeare", action="store_true", help="Legacy option: download the Project Gutenberg example")
    args = parser.parse_args()

    if args.vocab_size < 300:
        parser.error("--vocab-size must be at least 300")
    if args.steps < 1 or args.batch_size < 1 or args.block_size < 1 or args.eval_batches < 1:
        parser.error("steps, batch-size, block-size, and eval-batches must be positive")
    if args.eval_interval < 1:
        parser.error("eval-interval must be positive")

    random.seed(42)
    torch.manual_seed(42)
    device = choose_device(args.device)
    amp_dtype, precision_name = choose_precision(args.precision, device)

    if args.prepare_tokenizer:
        text = load_text(args.data, download_shakespeare=args.download_shakespeare)
        tokenizer = train_bpe_tokenizer(text, args.tokenizer, args.vocab_size)
        sample_ids = encode_text(tokenizer, text[:2000])
        print(f"Tokenizer saved to: {args.tokenizer}")
        print(f"Tokenizer vocab size: {tokenizer.get_vocab_size():,}")
        print(f"Sample length: {len(sample_ids):,} tokens")
        print(f"Roundtrip check: {tokenizer.decode(sample_ids, skip_special_tokens=False) == text[:2000]}")
        return

    if not args.tokenizer.is_file():
        parser.error(
            "Tokenizer file not found; train it first with --prepare-tokenizer --tokenizer <path> --vocab-size <N>."
        )

    tokenizer = load_tokenizer(args.tokenizer)
    text = load_text(args.data, download_shakespeare=args.download_shakespeare)
    tokenized = encode_text(tokenizer, text)
    if len(tokenized) < args.block_size + 2:
        parser.error(
            f"Corpus is too short after tokenization; need more than {args.block_size + 1} tokens. "
            f"Got {len(tokenized):,}."
        )

    data = torch.tensor(tokenized, dtype=torch.long, device=device)
    split = int(0.9 * data.numel())
    train_data, val_data = data[:split], data[split:]
    if val_data.numel() <= args.block_size:
        parser.error("Validation split is too short for the current block size; choose a larger corpus or smaller block size")

    if args.resume is not None:
        checkpoint, model = load_checkpoint(args.resume, device)
        start_step = checkpoint.get("step", 0)
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"Resuming from {args.resume} at step {start_step}")
    else:
        model = NanoTransformer(vocab_size=tokenizer.get_vocab_size(), max_len=args.block_size).to(device)
        start_step = 0
        best_val_loss = float("inf")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and amp_dtype == torch.float16))

    if args.resume is not None:
        optimizer.load_state_dict(torch.load(args.resume, map_location=device)["optimizer_state_dict"])

    print(
        f"Device: {device} | Precision: {precision_name} | Tokens: {data.numel():,} | Vocab: {tokenizer.get_vocab_size():,} "
        f"| Params: {count_parameters(model):,}"
    )

    for step in range(start_step + 1, args.steps + 1):
        model.train()
        x, y = get_batch(train_data, args.batch_size, args.block_size)
        autocast = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if device.type == "cuda" and amp_dtype is not None
            else contextlib.nullcontext()
        )
        with autocast:
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
            print(
                f"Step {step:>5}/{args.steps} | train loss {train_loss:.4f} "
                f"| val loss {val_loss:.4f} | val perplexity {math.exp(min(val_loss, 20)):.2f}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(args.best_checkpoint, model, args.tokenizer, step, args.data, precision_name, best_val_loss)
                print(f"Saved best checkpoint to {args.best_checkpoint}")

        if step % 500 == 0:
            save_checkpoint(args.checkpoint, model, args.tokenizer, step, args.data, precision_name, best_val_loss)
            print(f"Saved regular checkpoint to {args.checkpoint}")

    save_checkpoint(args.checkpoint, model, args.tokenizer, args.steps, args.data, precision_name, best_val_loss)
    print(f"Final checkpoint saved to {args.checkpoint}")
    print("\n--- Sample ---")
    print(model.generate(tokenizer, args.prompt, max_new_tokens=args.generate))


if __name__ == "__main__":
    main()
