"""Stage 2: turn every aligned face into a 512-d FaceNet embedding.

Saves one .npz file holding the embeddings, each image's person label, and its path,
so the classifier can be retrained in seconds without re-running the network.

Usage:
    python -m step3_face_recognition.embed
"""

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

from step3_face_recognition.download_lfw import DATA_DIR
from step3_face_recognition.faces import FaceEmbedder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DATA_DIR / "lfw_aligned")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "lfw_embeddings.npz")
    args = parser.parse_args()

    paths = sorted(args.input_dir.glob("*/*.jpg"))
    embedder = FaceEmbedder()
    print(f"Embedding {len(paths)} faces on {embedder.device}...")

    start = time.perf_counter()
    embeddings = embedder.embed([Image.open(p).convert("RGB") for p in paths])
    print(f"Done in {time.perf_counter() - start:.1f}s -> {embeddings.shape}")

    np.savez(
        args.output,
        embeddings=embeddings,
        labels=np.array([p.parent.name for p in paths]),
        paths=np.array([str(p.relative_to(args.input_dir)) for p in paths]),
    )
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
