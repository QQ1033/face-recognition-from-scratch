"""Stage 4: recognize every face in a new photo.

Draws a labeled box around each face and saves the annotated image. Faces whose
best match is below --threshold confidence are labeled "Unknown" -- without this,
the classifier would confidently give a stranger the name of whoever they most
resemble among the people it was trained on.

Usage:
    python -m step3_face_recognition.predict path/to/photo.jpg
"""

import argparse
from pathlib import Path

import joblib
from PIL import Image, ImageDraw

from step3_face_recognition.faces import FaceAligner, FaceEmbedder
from step3_face_recognition.train_classifier import DEFAULT_CLASSIFIER


class FaceRecognizer:
    def __init__(self, classifier_path: Path = DEFAULT_CLASSIFIER, threshold: float = 0.5):
        self.aligner = FaceAligner()
        self.embedder = FaceEmbedder()
        self.classifier = joblib.load(classifier_path)
        self.threshold = threshold

    def recognize(self, image: Image.Image) -> list[dict]:
        """Return [{"box", "name", "confidence"}] for each face, largest first."""
        faces = self.aligner.aligned_faces(image)
        if not faces:
            return []
        embeddings = self.embedder.embed([face for _, face in faces])
        probabilities = self.classifier.predict_proba(embeddings)

        results = []
        for (box, _), probs in zip(faces, probabilities):
            best = probs.argmax()
            confidence = float(probs[best])
            name = self.classifier.classes_[best] if confidence >= self.threshold else "Unknown"
            results.append({"box": [round(float(v)) for v in box], "name": name, "confidence": round(confidence, 4)})
        return results


def draw_results(image: Image.Image, results: list[dict]) -> Image.Image:
    annotated = image.convert("RGB")
    draw = ImageDraw.Draw(annotated)
    for r in results:
        color = "lime" if r["name"] != "Unknown" else "red"
        draw.rectangle(r["box"], outline=color, width=3)
        draw.text((r["box"][0], r["box"][3] + 4), f'{r["name"]} ({r["confidence"]:.0%})', fill=color)
    return annotated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, help="annotated image path (default: <image>_recognized.jpg)")
    args = parser.parse_args()

    image = Image.open(args.image)
    results = FaceRecognizer(threshold=args.threshold).recognize(image)
    if not results:
        print("No faces found.")
        return
    for r in results:
        print(f'{r["name"]:<30} confidence {r["confidence"]:.2%}  box {r["box"]}')

    output = args.output or args.image.with_name(f"{args.image.stem}_recognized.jpg")
    draw_results(image, results).save(output)
    print(f"Annotated image saved to {output}")


if __name__ == "__main__":
    main()
