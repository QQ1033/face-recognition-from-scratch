"""CNN for MNIST digit classification.

Same architecture as the original TensorFlow tutorial:
    conv(5x5, 32) -> ReLU -> maxpool(2x2)
    conv(5x5, 64) -> ReLU -> maxpool(2x2)
    fully connected (1024) -> ReLU -> dropout
    fully connected (10)   -> logits
"""

import torch
from torch import nn


class MnistCNN(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()
        # padding=2 with a 5x5 kernel is "same" padding: 28x28 in, 28x28 out.
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 -> 14x14
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14 -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns raw logits; nn.CrossEntropyLoss applies softmax internally.
        return self.classifier(self.features(x))
