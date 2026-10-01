"""Stage 1: detect, align, and crop the labeled person's face in every image of a dataset.

Input and output both use one folder per person:
    data/lfw_home/lfw_funneled/<Person>/<img>.jpg  ->  data/lfw_aligned/<Person>/<img>.jpg

LFW photos often contain several people, but each one is centered on the person in
its label, so the face closest to the center is kept (not the largest: a bystander
near the camera can have a bigger face than the subject).

Already-processed images are skipped, so the script can be stopped and resumed.

Usage:
    python -m step3_face_recognition.preprocess
"""

import argparse
import time
from itertools import groupby
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from step3_face_recognition.download_lfw import DATA_DIR, LFW_DIR
from step3_face_recognition.faces import FaceAligner, distance_from_center


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=LFW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR / "lfw_aligned")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    images = sorted(args.input_dir.glob("*/*.jpg"))
    todo = [p for p in images if not (args.output_dir / p.parent.name / p.name).exists()]
    aligner = FaceAligner()
    no_face = []
    start = time.perf_counter()

    with tqdm(total=len(todo), desc="Aligning faces") as progress:
        for i in range(0, len(todo), args.batch_size):
            loaded = [(p, Image.open(p).convert("RGB")) for p in todo[i:i + args.batch_size]]
            # MTCNN can only batch images of the same size (all of LFW is 250x250).
            for _, group in groupby(sorted(loaded, key=lambda x: x[1].size), key=lambda x: x[1].size):
                group = list(group)
                for (path, image), faces in zip(group, aligner.detect_batch([img for _, img in group])):
                    if not faces:
                        no_face.append(path)
                        continue
                    box, landmarks = min(faces, key=lambda f: distance_from_center(f[0], image.size))
                    out_path = args.output_dir / path.parent.name / path.name
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    aligner.align(image, box, landmarks).save(out_path, quality=95)
            progress.update(len(loaded))

    print(f"Aligned {len(todo) - len(no_face)} new images in {time.perf_counter() - start:.0f}s "
          f"({len(images) - len(todo)} already done, {len(no_face)} with no face found)")
    for path in no_face[:10]:
        print(f"  no face: {path.relative_to(args.input_dir)}")


if __name__ == "__main__":
    main()
