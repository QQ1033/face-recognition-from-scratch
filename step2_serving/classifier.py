"""Pre-trained ImageNet classifier (ResNet-50).

The modern stand-in for the Inception V3 model in the original Keras tutorial:
same idea (a CNN pre-trained on the 1000-class ImageNet dataset), but loaded
from torchvision with its matching preprocessing.
"""

import io

import torch
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50


class ImageClassifier:
    def __init__(self, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        weights = ResNet50_Weights.IMAGENET1K_V2
        # Downloads ~100 MB of weights on first use and caches them in ~/.cache/torch.
        self.model = resnet50(weights=weights).to(self.device).eval()
        # Resize to 232, center-crop to 224, convert to tensor, normalize with ImageNet mean/std.
        self.preprocess = weights.transforms()
        self.labels = weights.meta["categories"]

    @torch.inference_mode()
    def predict(self, image_bytes: bytes, top_k: int = 3) -> list[dict]:
        """Return the top_k predictions as [{"label", "probability"}], most likely first.

        Raises PIL.UnidentifiedImageError if the bytes are not a readable image.
        """
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        batch = self.preprocess(image).unsqueeze(0).to(self.device)  # model expects a batch: 1x3x224x224

        probabilities = self.model(batch).softmax(dim=1)[0]
        top = probabilities.topk(top_k)
        return [
            {"label": self.labels[i], "probability": round(p, 4)}
            for p, i in zip(top.values.tolist(), top.indices.tolist())
        ]
