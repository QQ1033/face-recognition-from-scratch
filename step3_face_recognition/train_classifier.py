"""Stage 3: train a classifier to recognize people from their face embeddings.

Only people with at least --min-images photos are used (158 people in LFW at the
default of 10), matching the original tutorial. Each person's photos are split
80/20 into train and test sets.

Usage:
    python -m step3_face_recognition.train_classifier
"""

import argparse
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from step3_face_recognition.download_lfw import DATA_DIR

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLASSIFIER = ROOT / "checkpoints" / "face_classifier.joblib"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, default=DATA_DIR / "lfw_embeddings.npz")
    parser.add_argument("--output", type=Path, default=DEFAULT_CLASSIFIER)
    parser.add_argument("--min-images", type=int, default=10)
    args = parser.parse_args()

    data = np.load(args.embeddings)
    X, y = data["embeddings"], data["labels"]

    counts = Counter(y)
    keep = np.array([counts[label] >= args.min_images for label in y])
    X, y = X[keep], y[keep]
    print(f"{len(set(y))} people with >= {args.min_images} images, {len(y)} images total")

    # stratify=y keeps each person's 80/20 ratio, so every person appears in both sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

    # A linear classifier finds the boundaries that separate people in embedding space.
    # The original tutorial used an SVM; linear SVM, logistic regression, and kNN all
    # score the same 98.3% on these embeddings, and logistic regression is the fastest
    # and outputs real probabilities (the confidence used in predict.py).
    # class_weight="balanced" stops George W. Bush (530 photos) from dominating
    # people with only 10.
    classifier = LogisticRegression(C=10, class_weight="balanced", max_iter=2000)
    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.4f} "
          f"({(predictions == y_test).sum()}/{len(y_test)})")

    mistakes = [(t, p) for t, p in zip(y_test, predictions) if t != p]
    for true, predicted in mistakes[:10]:
        print(f"  mistake: {true} predicted as {predicted}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, args.output)
    print(f"Saved classifier to {args.output}")


if __name__ == "__main__":
    main()
