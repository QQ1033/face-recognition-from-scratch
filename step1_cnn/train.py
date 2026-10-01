"""Train the MNIST CNN.

Usage:
    python -m step1_cnn.train --epochs 5
    tensorboard --logdir runs        # then open http://localhost:6006
"""

import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from step1_cnn.data import get_loaders
from step1_cnn.model import MnistCNN

ROOT = Path(__file__).resolve().parent.parent


@torch.no_grad()
def accuracy(model: nn.Module, loader, device) -> float:
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        correct += (model(images).argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="'cuda' or 'cpu' -- try both to compare training speed")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "mnist_cnn.pt")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Training on {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    train_loader, val_loader, _ = get_loaders(args.batch_size)
    model = MnistCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    writer = SummaryWriter(ROOT / "runs" / time.strftime("%Y%m%d-%H%M%S"))

    step = 0
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            loss = loss_fn(model(images), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                writer.add_scalar("loss/train", loss.item(), step)
            step += 1

        val_acc = accuracy(model, val_loader, device)
        writer.add_scalar("accuracy/val", val_acc, epoch)
        print(f"epoch {epoch}/{args.epochs}  loss {loss.item():.4f}  val acc {val_acc:.4f}  "
              f"({time.perf_counter() - start:.1f}s elapsed)")

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.checkpoint)
    writer.close()
    print(f"Saved checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()
