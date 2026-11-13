"""NanoLM: a small byte-level GPT-style language model.

Train on a local UTF-8 text file (for example, Shakespeare):
    python torchlm.py --data corpus.txt

Optional Project Gutenberg download:
    python torchlm.py --download-shakespeare --data corpus.txt

The default model has about 2.05M parameters. This educational model uses
byte-level UTF-8 tokenization, RoPE, causal multi-head attention, optional
CUDA mixed precision, and KV-cached generation.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import random
import re
import urllib.request
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


SHAKESPEARE_URL = "https://www.gutenberg.org/cache/epub/100/pg100.txt"


class ByteTokenizer:
    """Fixed UTF-8 byte tokenizer. Every token ID is in [0, 255]."""

    vocab_size = 256

    @staticmethod
    def encode(text: str) -> list[int]:
        return list(text.encode("utf-8"))

    @staticmethod
    def decode(tokens: list[int]) -> str:
        # Generation constrains bytes to valid UTF-8, so strict decode is safe.
        return bytes(tokens).decode("utf-8")


def _rotate_half_pairs(x: torch.Tensor) -> torch.Tensor:
    paired = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    even, odd = paired.unbind(dim=-1)
    return torch.stack((-odd, even), dim=-1).flatten(-2)


def apply_rope(q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to (batch, heads, time, head_dim) tensors."""
    dim = q.size(-1)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=q.device, dtype=torch.float32) / dim))
    angles = positions.to(torch.float32)[:, None] * inv_freq[None, :]
    angles = torch.cat((angles, angles), dim=-1)[None, None, :, :]
    cos = angles.cos().to(dtype=q.dtype)
    sin = angles.sin().to(dtype=q.dtype)
    return q * cos + _rotate_half_pairs(q) * sin, k * cos + _rotate_half_pairs(k) * sin


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with an optional inference KV cache."""

    def __init__(self, embed_dim: int, nhead: int, dropout: float) -> None:
        super().__init__()
        if embed_dim % nhead != 0:
            raise ValueError("embed_dim must be divisible by nhead")
        self.nhead = nhead
        self.head_dim = embed_dim // nhead
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        batch, time, channels = x.shape
        qkv = self.qkv(x).view(batch, time, 3, self.nhead, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = apply_rope(q, k, positions)

        has_past = past_kv is not None
        if has_past:
            past_k, past_v = past_kv
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)

        # During incremental decoding the query is the newest token and every
        # supplied key is from its past or present, so no causal mask is needed.
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=not has_past,
        )
        y = y.transpose(1, 2).contiguous().view(batch, time, channels)
        y = self.out_proj(y)

        new_kv = None
        if use_cache:
            new_kv = (k, v)
        return y, new_kv


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
    """Decoder-only Transformer with RoPE and cached incremental generation."""

    def __init__(
        self,
        vocab_size: int = 256,
        embed_dim: int = 128,
        nhead: int = 4,
        num_layers: int = 10,
        ff_dim: int = 512,
        max_len: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if max_len < 1:
            raise ValueError("max_len must be positive")
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

    def _prefill(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        seq_len = token_ids.size(1)
        positions = torch.arange(seq_len, device=token_ids.device)
        x = self.token_embedding(token_ids)
        caches = []
        for block in self.blocks:
            x, kv = block(x, positions, use_cache=True)
            assert kv is not None
            caches.append(kv)
        return self.lm_head(self.final_norm(x)), caches

    def _decode_one(
        self,
        token_id: torch.Tensor,
        position: int,
        caches: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        positions = torch.tensor([position], device=token_id.device)
        x = self.token_embedding(token_id)
        updated = []
        for block, past_kv in zip(self.blocks, caches):
            x, kv = block(x, positions, past_kv=past_kv, use_cache=True)
            assert kv is not None
            # Sliding cache retains exactly the training context window. RoPE
            # keeps positions on retained keys, so they do not need re-encoding.
            updated.append((kv[0][..., -self.max_len :, :], kv[1][..., -self.max_len :, :]))
        return self.lm_head(self.final_norm(x)), updated

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 300,
        temperature: float = 0.8,
        top_k: int = 40,
    ) -> str:
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        device = next(self.parameters()).device
        prompt_bytes = ByteTokenizer.encode(prompt) or [ord(" ")]
        # A long prompt is truncated to the context the model was trained on.
        prompt_bytes = prompt_bytes[-self.max_len :]
        ids = torch.tensor([prompt_bytes], dtype=torch.long, device=device)
        was_training = self.training
        self.eval()
        try:
            logits, caches = self._prefill(ids)
            all_bytes = list(prompt_bytes)
            next_logits = logits[:, -1, :]

            for step in range(max_new_tokens):
                scaled = next_logits / temperature
                # Only sample bytes that keep the output a valid UTF-8 stream.
                allowed = _valid_utf8_next_byte_mask(all_bytes, device)
                scaled = scaled.masked_fill(~allowed[None, :], float("-inf"))
                if top_k > 0:
                    k = min(top_k, int(allowed.sum().item()))
                    threshold = torch.topk(scaled, k).values[:, -1, None]
                    scaled = scaled.masked_fill(scaled < threshold, float("-inf"))
                next_id = torch.multinomial(F.softmax(scaled.float(), dim=-1), num_samples=1)
                byte_value = int(next_id.item())
                all_bytes.append(byte_value)

                if step + 1 < max_new_tokens:
                    position = len(prompt_bytes) + step
                    next_logits_all, caches = self._decode_one(next_id, position, caches)
                    next_logits = next_logits_all[:, -1, :]
        finally:
            self.train(was_training)

        return ByteTokenizer.decode(all_bytes)


def _valid_utf8_next_byte_mask(prefix: list[int], device: torch.device) -> torch.Tensor:
    """Mask byte IDs to keep prefix + next byte valid UTF-8."""
    allowed = torch.zeros(256, dtype=torch.bool, device=device)
    i = len(prefix) - 1
    while i >= 0 and 0x80 <= prefix[i] <= 0xBF:
        i -= 1

    if i < 0:
        # No valid prompt/generated sequence ends this way; fail safely.
        allowed[0x80:0xC0] = True
        return allowed

    lead = prefix[i]
    continuations = len(prefix) - i - 1
    if lead < 0x80 or lead > 0xBF:
        if lead < 0x80 or continuations >= 3:
            allowed[:0x80] = True
            allowed[0xC2:0xF5] = True
            return allowed

    if 0xC2 <= lead <= 0xDF and continuations == 0:
        allowed[0x80:0xC0] = True
    elif 0xE0 <= lead <= 0xEF and continuations == 0:
        low, high = (0xA0, 0xC0) if lead == 0xE0 else ((0x80, 0xA0) if lead == 0xED else (0x80, 0xC0))
        allowed[low:high] = True
    elif 0xF0 <= lead <= 0xF4 and continuations == 0:
        low, high = (0x90, 0xC0) if lead == 0xF0 else ((0x80, 0x90) if lead == 0xF4 else (0x80, 0xC0))
        allowed[low:high] = True
    elif 0xC2 <= lead <= 0xF4:
        expected = 1 if lead <= 0xDF else (2 if lead <= 0xEF else 3)
        if 0 < continuations < expected:
            allowed[0x80:0xC0] = True
        else:
            allowed[:0x80] = True
            allowed[0xC2:0xF5] = True
    else:
        # Defensive fallback for malformed external prefix.
        allowed[:0x80] = True
        allowed[0xC2:0xF5] = True
    return allowed


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def load_text(data_path: Path, download_shakespeare: bool) -> str:
    if download_shakespeare:
        print(f"Downloading Shakespeare corpus from {SHAKESPEARE_URL}")
        request = urllib.request.Request(SHAKESPEARE_URL, headers={"User-Agent": "NanoLM educational experiment"})
        with urllib.request.urlopen(request, timeout=30) as response:
            text = response.read().decode("utf-8", errors="replace")
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_text(text, encoding="utf-8")
        print(f"Saved corpus to {data_path}")
        return text
    if not data_path.is_file():
        raise FileNotFoundError(f"Training text not found: {data_path}")
    return data_path.read_text(encoding="utf-8", errors="replace")


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
            batch_losses.append(F.cross_entropy(logits.float().reshape(-1, 256), y.reshape(-1)).item())
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
    """Return autocast dtype and display label; non-CUDA defaults to FP32."""
    if device.type != "cuda":
        if requested not in ("auto", "fp32"):
            print(f"{requested} mixed precision requested, but {device.type} is not CUDA; using FP32.")
        return None, "fp32"
    if requested == "fp32":
        return None, "fp32"
    if requested == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise ValueError("This CUDA GPU does not support bf16; choose --precision fp16 or auto")
        return torch.bfloat16, "bf16"
    if requested == "fp16":
        return torch.float16, "fp16"
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, "bf16"
    return torch.float16, "fp16"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a byte-level Shakespeare language model.")
    parser.add_argument("--data", type=Path, default=Path("corpus.txt"), help="UTF-8 training text")
    parser.add_argument("--download-shakespeare", action="store_true", help="Download Project Gutenberg complete works")
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
    parser.add_argument("--generate", type=int, default=300, help="Number of new bytes to generate")
    args = parser.parse_args()

    if args.steps < 1 or args.batch_size < 1 or args.block_size < 1 or args.eval_batches < 1:
        parser.error("steps, batch-size, block-size, and eval-batches must be positive")
    if args.eval_interval < 1:
        parser.error("eval-interval must be positive")

    random.seed(42)
    torch.manual_seed(42)
    device = choose_device(args.device)
    amp_dtype, precision_name = choose_precision(args.precision, device)
    text = load_text(args.data, args.download_shakespeare)
    encoded = torch.tensor(ByteTokenizer.encode(text), dtype=torch.long)
    if encoded.numel() < args.block_size + 2:
        parser.error(f"Corpus is too short for block-size={args.block_size}; provide a longer file or lower block size")

    split = int(0.9 * encoded.numel())
    train_data, val_data = encoded[:split].to(device), encoded[split:].to(device)
    if val_data.numel() <= args.block_size:
        parser.error("Validation split is too short for the selected block size")

    model = NanoTransformer(max_len=args.block_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
    use_amp = device.type == "cuda" and amp_dtype is not None
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
    print(
        f"Device: {device} | Precision: {precision_name} | Corpus: {encoded.numel():,} bytes "
        f"| Parameters: {count_parameters(model):,}"
    )

    for step in range(1, args.steps + 1):
        model.train()
        x, y = get_batch(train_data, args.batch_size, args.block_size)
        amp_context = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if use_amp
            else contextlib.nullcontext()
        )
        with amp_context:
            logits = model(x)
            loss = F.cross_entropy(logits.float().reshape(-1, 256), y.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % args.eval_interval == 0 or step == args.steps:
            train_loss, val_loss = estimate_loss(
                model, train_data, val_data, args.batch_size, args.block_size, args.eval_batches
            )
            print(
                f"Step {step:>5}/{args.steps} | train loss {train_loss:.4f} "
                f"| val loss {val_loss:.4f} | val perplexity {math.exp(min(val_loss, 20)):.2f}"
            )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "vocab_size": 256,
                "embed_dim": 128,
                "nhead": 4,
                "num_layers": 10,
                "ff_dim": 512,
                "max_len": args.block_size,
                "position_encoding": "RoPE",
            },
            "training_data": str(args.data),
            "precision": precision_name,
        },
        args.checkpoint,
    )
    print(f"Checkpoint saved to {args.checkpoint}")
    print("\n--- Sample ---")
    print(model.generate(args.prompt, max_new_tokens=args.generate))


if __name__ == "__main__":
    main()
