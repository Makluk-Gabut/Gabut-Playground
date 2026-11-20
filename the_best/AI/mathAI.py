"""
Minimal Math AI - ultra-lightweight supervised arithmetic model

Features:
- Tiny LSTM + attention architecture
- Separate train/validation datasets
- Signed-log target normalization
- Confidence target based on prediction accuracy
- Best-checkpoint save/load
- Deterministic evaluation
- CSV and JSON training history
"""

import argparse
import csv
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class MathTokenizer:
    """Small character-style tokenizer for arithmetic expressions."""

    def __init__(self):
        self.tokens = [
            "<pad>", "<start>", "<end>", "<unk>",
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "+", "-", "*", "/", "%", "**", "(", ")",
            "floor", "ceil", "round", "abs", "sqrt", "."
        ]
        self.token2id = {token: index for index, token in enumerate(self.tokens)}
        self.id2token = {index: token for token, index in self.token2id.items()}
        self.vocab_size = len(self.tokens)

    def tokenize(self, expr):
        expr = str(expr).replace(" ", "")
        pattern = r"(\d+\.?\d*|floor|ceil|round|abs|sqrt|\*\*|[+\-*/%().])"
        matches = re.findall(pattern, expr)
        return ["<start>", *matches, "<end>"]

    def encode(self, expr, max_len=40):
        tokens = self.tokenize(expr)[:max_len]
        ids = [self.token2id.get(token, self.token2id["<unk>"]) for token in tokens]
        ids += [self.token2id["<pad>"]] * max(0, max_len - len(ids))
        return torch.tensor(ids[:max_len], dtype=torch.long)

    def decode(self, ids):
        tokens = []
        for value in ids:
            index = value.item() if isinstance(value, torch.Tensor) else value
            if index != self.token2id["<pad>"]:
                tokens.append(self.id2token.get(index, "<unk>"))
        return " ".join(tokens)


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        mask = torch.ne(x.abs().sum(dim=-1), 0)
        scores = self.attention(x).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return torch.einsum("bs,bsh->bh", weights, x)


def normalize_answer(answer):
    """Map signed values into a manageable signed-log scale."""
    answer = float(answer)
    sign = 1.0 if answer >= 0 else -1.0
    return sign * math.log1p(abs(answer))


def denormalize_answer(normalized):
    normalized = float(normalized)
    sign = 1.0 if normalized >= 0 else -1.0
    return sign * math.expm1(abs(normalized))


class MinimalMathAI(nn.Module):
    def __init__(self, vocab_size, embedding_dim=8, hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            num_layers=1,
        )
        self.attention = AttentionPooling(hidden_dim)
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
        )
        self.answer_head = nn.Linear(16, 1)
        self.confidence_head = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        for name, parameter in self.named_parameters():
            if "weight" in name:
                if parameter.dim() > 1:
                    nn.init.orthogonal_(parameter)
                else:
                    nn.init.normal_(parameter, std=0.02)
            elif "bias" in name:
                nn.init.zeros_(parameter)

    def forward(self, token_ids):
        embedded = self.embedding(token_ids)
        lstm_output, _ = self.lstm(embedded)
        context = self.attention(lstm_output)
        representation = self.shared(context)
        return {
            "answer": self.answer_head(representation),
            "confidence": self.confidence_head(representation),
        }

    def count_parameters(self):
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


class MathExpressionGenerator:
    """Generates expression/answer pairs without using hardcoded inference rules."""

    @staticmethod
    def generate(difficulty=1, seed=None):
        if seed is not None:
            random.seed(seed)

        limit = max(5, 20 * difficulty)
        operations = ["+", "-", "*", "/", "%"]

        if random.random() < 0.25:
            function = random.choice(["abs", "floor", "ceil", "round", "sqrt"])
            if function == "abs":
                a = random.randint(-limit, limit)
                b = random.randint(-limit, limit)
                expression = f"abs({a}-{b})"
                answer = abs(a - b)
            elif function == "sqrt":
                a = random.randint(1, limit * limit)
                expression = f"sqrt({a})"
                answer = math.sqrt(a)
            else:
                a = random.randint(1, limit)
                b = random.randint(1, limit)
                expression = f"{function}({a}/{b})"
                if function == "floor":
                    answer = math.floor(a / b)
                elif function == "ceil":
                    answer = math.ceil(a / b)
                else:
                    answer = round(a / b)
        else:
            operator = random.choice(operations)
            a = random.randint(-limit, limit)
            b = random.randint(-limit, limit)

            if operator == "/" and b == 0:
                b = 1

            if operator == "+":
                answer = a + b
            elif operator == "-":
                answer = a - b
            elif operator == "*":
                answer = a * b
            elif operator == "/":
                answer = a / b
            else:
                if b == 0:
                    b = 1
                answer = a % b

            expression = f"{a}{operator}{b}"

        return expression, float(answer)


class MathDataset(Dataset):
    def __init__(self, tokenizer, num_samples, difficulty, seed=42):
        self.tokenizer = tokenizer
        rng = random.Random(seed)
        self.data = []

        for _ in range(num_samples):
            old_state = random.getstate()
            random.setstate(rng.getstate())
            expression, answer = MathExpressionGenerator.generate(difficulty)
            rng.setstate(random.getstate())
            random.setstate(old_state)

            encoded = tokenizer.encode(expression)
            normalized_answer = normalize_answer(answer)
            self.data.append((encoded, torch.tensor([normalized_answer], dtype=torch.float32), answer))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        encoded, normalized_answer, raw_answer = self.data[index]
        return encoded, normalized_answer, torch.tensor([raw_answer], dtype=torch.float32)


class MathAITrainer:
    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.model.to(self.device)
        self.history = {
            "epoch": [],
            "train_loss": [], "train_mae": [],
            "val_loss": [], "val_mae": [],
            "confidence_mae": [],
        }

    def train_epoch(self, dataloader, optimizer):
        self.model.train()
        total_loss = total_mae = 0.0

        for token_ids, normalized_answers, raw_answers in dataloader:
            token_ids = token_ids.to(self.device)
            normalized_answers = normalized_answers.to(self.device)
            raw_answers = raw_answers.to(self.device)

            output = self.model(token_ids)
            normalized_predictions = output["answer"]

            answer_loss = nn.functional.smooth_l1_loss(normalized_predictions, normalized_answers)
            predicted_raw = torch.tensor(
                [denormalize_answer(value.item()) for value in normalized_predictions.detach().cpu()],
                dtype=torch.float32,
                device=self.device,
            ).view(-1, 1)

            confidence_target = torch.exp(-torch.abs(predicted_raw - raw_answers) / 100.0)
            confidence_loss = nn.functional.mse_loss(output["confidence"], confidence_target.detach())
            loss = answer_loss + 0.1 * confidence_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

            predicted_raw = predicted_raw.detach()
            total_loss += loss.item()
            total_mae += torch.abs(predicted_raw - raw_answers).mean().item()

        batches = max(1, len(dataloader))
        return total_loss / batches, total_mae / batches

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = total_mae = total_confidence_mae = 0.0

        with torch.no_grad():
            for token_ids, normalized_answers, raw_answers in dataloader:
                token_ids = token_ids.to(self.device)
                normalized_answers = normalized_answers.to(self.device)
                raw_answers = raw_answers.to(self.device)

                output = self.model(token_ids)
                answer_loss = nn.functional.smooth_l1_loss(output["answer"], normalized_answers)

                raw_predictions = torch.tensor(
                    [denormalize_answer(value.item()) for value in output["answer"].detach().cpu()],
                    dtype=torch.float32,
                    device=self.device,
                ).view(-1, 1)

                confidence_target = torch.exp(-torch.abs(raw_predictions - raw_answers) / 100.0)
                confidence_mae = torch.abs(output["confidence"] - confidence_target).mean().item()

                total_loss += answer_loss.item()
                total_mae += torch.abs(raw_predictions - raw_answers).mean().item()
                total_confidence_mae += confidence_mae

        batches = max(1, len(dataloader))
        return total_loss / batches, total_mae / batches, total_confidence_mae / batches

    def save_checkpoint(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": self.model.state_dict(),
            "history": self.history,
        }, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model"])
        self.history = checkpoint.get("history", self.history)

    def train(self, num_epochs=150, batch_size=64, lr=0.003, difficulty=1,
              train_samples=6000, val_samples=1000, save_path="mathai_checkpoint.pth"):
        train_dataset = MathDataset(self.tokenizer, train_samples, difficulty, seed=42)
        val_dataset = MathDataset(self.tokenizer, val_samples, difficulty, seed=84)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-6)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        best_val_mae = float("inf")
        patience = 30
        patience_counter = 0

        print(f"MathAI | parameters={self.model.count_parameters():,} | epochs={num_epochs}")

        for epoch in range(num_epochs):
            train_loss, train_mae = self.train_epoch(train_loader, optimizer)
            val_loss, val_mae, confidence_mae = self.evaluate(val_loader)
            scheduler.step()

            self.history["epoch"].append(epoch + 1)
            self.history["train_loss"].append(train_loss)
            self.history["train_mae"].append(train_mae)
            self.history["val_loss"].append(val_loss)
            self.history["val_mae"].append(val_mae)
            self.history["confidence_mae"].append(confidence_mae)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"Epoch {epoch + 1:4d} | train_loss={train_loss:.5f} "
                    f"train_mae={train_mae:.4f} | val_loss={val_loss:.5f} "
                    f"val_mae={val_mae:.4f}"
                )

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                patience_counter = 0
                self.save_checkpoint(save_path)
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        self.save_history(Path(save_path).with_name("mathai_history.json"))
        self.save_history_csv(Path(save_path).with_name("mathai_history.csv"))
        print(f"Training complete | best validation MAE={best_val_mae:.4f}")
        print(f"Best checkpoint: {save_path}")

    def save_history(self, path):
        with Path(path).open("w", encoding="utf-8") as file:
            json.dump(self.history, file, indent=2)

    def save_history_csv(self, path):
        with Path(path).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(self.history))
            writer.writeheader()
            for index in range(len(self.history["epoch"])):
                writer.writerow({key: values[index] for key, values in self.history.items()})

    def predict(self, expression):
        self.model.eval()
        with torch.no_grad():
            encoded = self.tokenizer.encode(expression).unsqueeze(0).to(self.device)
            output = self.model(encoded)
            answer = denormalize_answer(output["answer"].item())
            confidence = output["confidence"].item()
        return answer, confidence


class MathAIGame:
    def __init__(self, trainer):
        self.trainer = trainer

    def test(self, num_tests=100, difficulty=1, tolerance=0.05, seed=2026):
        rng = random.Random(seed)
        correct = 0
        total_error = 0.0

        print(f"Testing {num_tests} expressions (difficulty={difficulty}, tolerance={tolerance})")
        for index in range(num_tests):
            old_state = random.getstate()
            random.setstate(rng.getstate())
            expression, true_answer = MathExpressionGenerator.generate(difficulty)
            rng.setstate(random.getstate())
            random.setstate(old_state)

            predicted_answer, confidence = self.trainer.predict(expression)
            error = abs(predicted_answer - true_answer)
            if error <= tolerance:
                correct += 1
            total_error += error

            if index < 20:
                status = "OK" if error <= tolerance else "MISS"
                print(
                    f"{status:4s} {expression:18s} | true={true_answer:.4f} "
                    f"| pred={predicted_answer:.4f} | err={error:.4f} | conf={confidence:.2%}"
                )

        accuracy = correct / num_tests * 100
        average_error = total_error / num_tests
        print(f"Results: {correct}/{num_tests} ({accuracy:.1f}%) | MAE={average_error:.4f}")
        return accuracy


def main():
    parser = argparse.ArgumentParser(description="Train or test Minimal Math AI")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--difficulty", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--checkpoint", default="mathai_checkpoint.pth")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = MathTokenizer()
    model = MinimalMathAI(tokenizer.vocab_size)
    trainer = MathAITrainer(model, tokenizer, device=device)

    if args.train:
        trainer.train(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            difficulty=args.difficulty,
            save_path=args.checkpoint,
        )

    checkpoint = Path(args.checkpoint)
    if checkpoint.exists() and not args.train:
        trainer.load_checkpoint(checkpoint)

    if args.test:
        if not checkpoint.exists():
            raise FileNotFoundError("Checkpoint not found. Run --train first.")
        MathAIGame(trainer).test(num_tests=100, difficulty=args.difficulty)

    if args.interactive:
        if not checkpoint.exists():
            raise FileNotFoundError("Checkpoint not found. Run --train first.")
        while True:
            expression = input("Expression (or 'quit'): ").strip()
            if expression.lower() == "quit":
                break
            try:
                answer, confidence = trainer.predict(expression)
                print(f"Answer: {answer:.6f} | Confidence: {confidence:.2%}")
            except Exception as error:
                print(f"Error: {error}")

    if not (args.train or args.test or args.interactive):
        print("Use --train, --test, or --interactive")


if __name__ == "__main__":
    main()
