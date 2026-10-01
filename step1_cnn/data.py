"""MNIST loading: train / validation / test splits."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Mean / std of the MNIST training set, the standard normalization values.
_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


def get_loaders(batch_size: int = 128, val_size: int = 5000, num_workers: int = 2, seed: int = 0):
    """Return (train_loader, val_loader, test_loader).

    The validation set is carved out of the 60k training images and is used to
    monitor training; the 10k test set is only touched by evaluate.py.
    """
    full_train = datasets.MNIST(DATA_DIR, train=True, download=True, transform=_TRANSFORM)
    test = datasets.MNIST(DATA_DIR, train=False, download=True, transform=_TRANSFORM)

    generator = torch.Generator().manual_seed(seed)
    train, val = random_split(full_train, [len(full_train) - val_size, val_size], generator=generator)

    pin = torch.cuda.is_available()
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin),
        DataLoader(val, batch_size=batch_size, num_workers=num_workers, pin_memory=pin),
        DataLoader(test, batch_size=batch_size, num_workers=num_workers, pin_memory=pin),
    )
