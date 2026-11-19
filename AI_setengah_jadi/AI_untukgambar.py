"""Train or run a transfer-learning image classifier for organic vs. inorganic waste.

Expected dataset layout (class folder names become the model labels):

    data/sampah_dataset/
        train/
            anorganik/...
            organik/...
        val/
            anorganik/...
            organik/...

Example training command:
    python AI_untukgambar.py --data-dir data/sampah_dataset --epochs 5

Example prediction command:
    python AI_untukgambar.py --predict path/to/image.jpg \
        --checkpoint checkpoints/best_model_sortir_sampah.pth
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from PIL import Image

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a ResNet50 image classifier or predict one image."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/sampah_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Use 0 for the most portable setting, especially on Windows/Colab.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Do not download/use ImageNet pretrained weights.")
    parser.add_argument("--predict", type=Path,
                        help="Image path for prediction; skips training when supplied.")
    parser.add_argument("--checkpoint", type=Path,
                        help="Checkpoint path for prediction (defaults to output-dir/best_model_sortir_sampah.pth).")
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available in this PyTorch installation")
    return args


def choose_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms():
    return {
        "train": transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
        "val": transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
    }


def load_datasets(data_dir):
    transforms_by_phase = build_transforms()
    datasets_by_phase = {}
    for phase in ("train", "val"):
        phase_dir = data_dir / phase
        if not phase_dir.is_dir():
            raise FileNotFoundError(
                f"Dataset folder not found: {phase_dir}. Expected train/ and val/ folders."
            )
        datasets_by_phase[phase] = datasets.ImageFolder(
            root=str(phase_dir), transform=transforms_by_phase[phase]
        )
        if len(datasets_by_phase[phase]) == 0:
            raise ValueError(f"No images found in {phase_dir}.")

    train_classes = datasets_by_phase["train"].classes
    val_classes = datasets_by_phase["val"].classes
    if train_classes != val_classes:
        raise ValueError(
            "Train and validation class folders must match. "
            f"Train classes: {train_classes}; validation classes: {val_classes}"
        )
    if len(train_classes) < 2:
        raise ValueError("At least two class folders are required.")
    return datasets_by_phase, train_classes


def make_model(num_classes, pretrained=True):
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    # Keep the frozen feature extractor (including BatchNorm statistics) fixed.
    model.eval()
    model.fc.train()
    loss_sum = 0.0
    correct = 0
    count = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        loss_sum += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        count += batch_size

    return loss_sum / count, correct / count


@torch.inference_mode()
def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)

        batch_size = labels.size(0)
        loss_sum += loss.item() * batch_size
        correct += (predictions == labels).sum().item()
        count += batch_size
        for actual, predicted in zip(labels.cpu().tolist(), predictions.cpu().tolist()):
            matrix[actual, predicted] += 1

    return loss_sum / count, correct / count, matrix


def save_confusion_matrix(path, matrix, class_names):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["actual\\predicted", *class_names])
        for class_name, row in zip(class_names, matrix.tolist()):
            writer.writerow([class_name, *row])


def train(args, device):
    datasets_by_phase, class_names = load_datasets(args.data_dir)
    pin_memory = device.type == "cuda"
    loaders = {
        phase: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(phase == "train"),
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            persistent_workers=(args.num_workers > 0),
        )
        for phase, dataset in datasets_by_phase.items()
    }

    model = make_model(len(class_names), pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.learning_rate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "best_model_sortir_sampah.pth"
    last_path = args.output_dir / "last_model_sortir_sampah.pth"
    history_path = args.output_dir / "training_history.csv"
    best_val_loss = float("inf")
    history = []

    print(f"Device: {device}")
    print(f"Classes: {class_names}")
    print("Image counts: " + ", ".join(
        f"{phase}={len(dataset)}" for phase, dataset in datasets_by_phase.items()
    ))

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device
        )
        val_loss, val_accuracy, _ = evaluate(
            model, loaders["val"], criterion, device, len(class_names)
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        })
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train loss={train_loss:.4f}, acc={train_accuracy:.4f} | "
            f"val loss={val_loss:.4f}, acc={val_accuracy:.4f}"
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "image_size": IMAGE_SIZE,
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
            "epoch": epoch,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }
        torch.save(checkpoint, last_path)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, best_path)

    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    _, best_accuracy, matrix = evaluate(
        model, loaders["val"], criterion, device, len(class_names)
    )
    matrix_path = args.output_dir / "validation_confusion_matrix.csv"
    save_confusion_matrix(matrix_path, matrix, class_names)

    print(f"Best checkpoint: {best_path}")
    print(f"Final checkpoint: {last_path}")
    print(f"Training history: {history_path}")
    print(f"Validation confusion matrix: {matrix_path}")
    print(f"Best validation loss: {best_val_loss:.4f} | accuracy: {best_accuracy:.4f}")


def predict(image_path, checkpoint_path, device):
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_names = checkpoint["class_names"]
    model = make_model(len(class_names), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image_transform = transforms.Compose([
        transforms.Resize((checkpoint.get("image_size", IMAGE_SIZE),) * 2),
        transforms.ToTensor(),
        transforms.Normalize(
            checkpoint.get("mean", IMAGENET_MEAN),
            checkpoint.get("std", IMAGENET_STD),
        ),
    ])
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        tensor = image_transform(image).unsqueeze(0).to(device)

    probabilities = torch.softmax(model(tensor), dim=1)[0]
    score, index = probabilities.max(dim=0)
    print(f"Prediction: {class_names[index.item()]}")
    print(f"Confidence: {score.item():.2%}")
    print("Class probabilities:")
    for class_index in probabilities.argsort(descending=True).tolist():
        print(f"  {class_names[class_index]}: {probabilities[class_index].item():.2%}")


def main():
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)

    if args.predict:
        checkpoint_path = args.checkpoint or (
            args.output_dir / "best_model_sortir_sampah.pth"
        )
        predict(args.predict, checkpoint_path, device)
    else:
        train(args, device)


if __name__ == "__main__":
    main()
