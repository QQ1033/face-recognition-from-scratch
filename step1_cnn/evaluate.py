"""Evaluate a trained checkpoint on the held-out MNIST test set.

Usage:
    python -m step1_cnn.evaluate
"""

import argparse
from pathlib import Path

import torch

from step1_cnn.data import get_loaders
from step1_cnn.model import MnistCNN
from step1_cnn.train import ROOT, accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "mnist_cnn.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MnistCNN().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    _, _, test_loader = get_loaders()
    print(f"Test accuracy: {accuracy(model, test_loader, device):.4f}")


if __name__ == "__main__":
    main()
