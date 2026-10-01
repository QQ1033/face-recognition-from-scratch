"""Download the LFW (Labeled Faces in the Wild) dataset.

13,233 photos of 5,749 people, one folder per person:
    lfw_funneled/George_W_Bush/George_W_Bush_0001.jpg

scikit-learn's fetcher is used only because it downloads from a reliable mirror;
we use the raw image folders it leaves behind, not the arrays it returns.

Usage:
    python -m step3_face_recognition.download_lfw
"""

from pathlib import Path

from sklearn.datasets import fetch_lfw_people

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LFW_DIR = DATA_DIR / "lfw_home" / "lfw_funneled"


def main():
    # Downloads ~230 MB on first run; resize=0.1 keeps the arrays sklearn loads tiny.
    fetch_lfw_people(data_home=DATA_DIR, min_faces_per_person=10, resize=0.1)
    people = [p for p in LFW_DIR.iterdir() if p.is_dir()]
    images = sum(len(list(p.glob("*.jpg"))) for p in people)
    print(f"LFW ready at {LFW_DIR}: {images} images of {len(people)} people")


if __name__ == "__main__":
    main()
