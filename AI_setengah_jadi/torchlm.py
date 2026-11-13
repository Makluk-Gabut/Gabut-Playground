"""NanoLM: a small byte-level GPT-style language model.

Train on a local text file (for example, Shakespeare):
    python torchlm.py --data corpus.txt

Optional Project Gutenberg download:
    python torchlm.py --download-shakespeare

The default model has about 1.9M parameters. This is an educational
character/byte-level model, not a production language model.
"""

from __future__ import annotations

import argparse
import math
import random
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


SHAKESPEARE_URL = "https://www.gutenberg.org/cache/epub/100/pg100.txt"


class ByteTokenizer:
    """UTF-8 byte tokenizer with a fixed, consistent 256-token vocabulary."""

    vocab_size = 256

    @staticmethod
    def encode(text: str) -> list[int]:
        return list(text.encode("utf-8"))

    @staticmethod
    def decode(tokens: list[int]) -> str:
        return bytes(tokens).decode("utf-8", errors="replace")


class NanoTransformer(nn.Module):
    """Decoder-only Transformer with learned token and position embeddings."""

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
        if embed_dim % nhead:
            raise ValueError("embed_dim must be divisible by nhead")

        self.max_len = max_len
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        _, seq_len = token_ids.shape
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len={self.max_len}")

        positions = torch.arange(seq_len, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.pos_embedding(positions)[None, :, :]
        # For PyTorch Transformer modules, True entries in a boolean mask are blocked.
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=token_ids.device),
            diagonal=1,
        )
        x = self.transformer(x, mask=causal_mask)
        return self.lm_head(self.final_norm(x))

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
        device = next(self.parameters()).device
        prompt_ids = ByteTokenizer.encode(prompt) or [ord(" ")]
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        was_training = self.training
        self.eval()

        for _ in range(max_new_tokens):
            context = ids[:, -self.max_len :]
            logits = self(context)[:, -1, :] / temperature
            if top_k > 0:
                k = min(top_k, logits.size(-1))
                threshold = torch.topk(logits, k).values[:, -1, None]
                logits = logits.masked_fill(logits < threshold, float("-inf"))
            next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            ids = torch.cat((ids, next_id), dim=1)

        self.train(was_training)
        return ByteTokenizer.decode(ids[0].tolist())


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def load_text(data_path: Path, download_shakespeare: bool) -> str:
    if download_shakespeare:
        print(f"Downloading Shakespeare corpus from {SHAKESPEARE_URL}")
        request = urllib.request.Request(
            SHAKESPEARE_URL,
            headers={"User-Agent": "NanoLM educational experiment"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            text = response.read().decode("utf-8", errors="replace")
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_text(text, encoding="utf-8")
        print(f"Saved corpus to {data_path}")
        return text

    if not data_path.is_file():
        raise FileNotFoundError(
            f"Training text not found: {data_path}. Provide --data PATH or use --download-shakespeare."
        )
    return data_path.read_text(encoding="utf-8", errors="replace")


def get_batch(data: torch.Tensor, batch_size: int, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(0, data.numel() - block_size - 1, (batch_size,), device=data.device)
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
            batch_losses.append(F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1)).item())
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small byte-level Shakespeare language model.")
    parser.add_argument("--data", type=Path, default=Path("corpus.txt"), help="UTF-8 training text")
    parser.add_argument("--download-shakespeare", action="store_true", help="Download Project Gutenberg's Complete Works")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--checkpoint", type=Path, default=Path("nanolm_shakespeare.pt"))
    parser.add_argument("--prompt", default="To be, or not to be:")
    parser.add_argument("--generate", type=int, default=300, help="Number of new bytes to generate")
    args = parser.parse_args()

    if args.steps < 1 or args.batch_size < 1 or args.block_size < 1:
        parser.error("steps, batch-size, and block-size must be positive")

    random.seed(42)
    torch.manual_seed(42)
    device = choose_device(args.device)
    text = load_text(args.data, args.download_shakespeare)
    encoded = torch.tensor(ByteTokenizer.encode(text), dtype=torch.long)
    if encoded.numel() < args.block_size + 2:
        parser.error(
            f"Corpus has only {encoded.numel()} bytes; need more than block-size + 1 "
            f"({args.block_size + 1}). Use a longer text or lower --block-size."
        )

    split = int(0.9 * encoded.numel())
    train_data = encoded[:split].to(device)
    val_data = encoded[split:].to(device)
    if val_data.numel() < args.block_size + 2:
        parser.error("Validation split is too short for this block size; use a larger corpus or smaller --block-size.")

    model = NanoTransformer(max_len=args.block_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
    print(f"Device: {device} | Corpus: {encoded.numel():,} bytes | Parameters: {count_parameters(model):,}")

    for step in range(1, args.steps + 1):
        model.train()
        x, y = get_batch(train_data, args.batch_size, args.block_size)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

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
            },
            "training_data": str(args.data),
        },
        args.checkpoint,
    )
    print(f"Checkpoint saved to {args.checkpoint}")
    print("\n--- Sample ---")
    print(model.generate(args.prompt, max_new_tokens=args.generate))


if __name__ == "__main__":
    main()
