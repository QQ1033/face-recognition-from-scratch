"""Face detection, alignment, and embedding -- the building blocks shared by every stage.

Pipeline for one photo:
    1. Detect   - MTCNN finds each face's bounding box + 5 landmarks (eyes, nose, mouth corners).
    2. Align    - rotate the photo so the eyes are level, then crop a square around the face.
    3. Embed    - FaceNet (Inception-ResNet V1) maps the 160x160 crop to a 512-d vector.
                  Photos of the same person land close together in that vector space.
"""

import math

import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

FACE_SIZE = 160  # FaceNet's expected input size


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FaceAligner:
    def __init__(self, device: torch.device | None = None, min_confidence: float = 0.95):
        self.detector = MTCNN(keep_all=True, device=device or default_device())
        self.min_confidence = min_confidence

    def detect(self, image: Image.Image) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return [(box, landmarks)] for each confident face, largest face first.

        box is [x1, y1, x2, y2]; landmarks is 5x2: left eye, right eye, nose,
        left mouth corner, right mouth corner (from the viewer's perspective).
        """
        return self.detect_batch([image])[0]

    def detect_batch(self, images: list[Image.Image]) -> list[list[tuple[np.ndarray, np.ndarray]]]:
        """detect() for several same-sized images in one pass through the network."""
        all_boxes, all_probs, all_landmarks = self.detector.detect(images, landmarks=True)
        results = []
        for image, boxes, probs, landmarks in zip(images, all_boxes, all_probs, all_landmarks):
            if boxes is None:
                results.append([])
                continue
            # facenet-pytorch can return object arrays under NumPy 2; make them plain floats.
            faces = [(np.asarray(b, dtype=np.float32), np.asarray(l, dtype=np.float32))
                     for b, p, l in zip(boxes, probs, landmarks) if p >= self.min_confidence]
            results.append(sorted(faces, key=lambda f: visible_area(f[0], image.size), reverse=True))
        return results

    @staticmethod
    def align(image: Image.Image, box: np.ndarray, landmarks: np.ndarray, margin: float = 0.2) -> Image.Image:
        """Rotate so the eyes are horizontal, then crop a square around the face and resize."""
        left_eye, right_eye = landmarks[0], landmarks[1]
        dx, dy = right_eye - left_eye
        angle = math.degrees(math.atan2(dy, dx))  # >0 means the head is tilted clockwise
        eye_center = tuple(((left_eye + right_eye) / 2).tolist())
        # PIL rotates counter-clockwise, undoing the tilt. Rotating around the eye
        # center keeps the face where the box says it is.
        rotated = image.rotate(angle, center=eye_center, resample=Image.BILINEAR)

        x1, y1, x2, y2 = box
        half = max(x2 - x1, y2 - y1) * (1 + margin) / 2
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        crop = rotated.crop((cx - half, cy - half, cx + half, cy + half))
        return crop.resize((FACE_SIZE, FACE_SIZE), Image.BILINEAR)

    def aligned_faces(self, image: Image.Image) -> list[tuple[np.ndarray, Image.Image]]:
        """Return [(box, aligned 160x160 face)] for every face, largest first."""
        image = image.convert("RGB")
        return [(box, self.align(image, box, lms)) for box, lms in self.detect(image)]


def visible_area(box: np.ndarray, image_size: tuple[int, int]) -> float:
    """Area of the box after clipping it to the image.

    MTCNN extends boxes past the edge for faces that are cut off by the frame, which
    would otherwise make a half-visible face look like the largest one.
    """
    width, height = image_size
    x1, y1, x2, y2 = max(box[0], 0), max(box[1], 0), min(box[2], width), min(box[3], height)
    return max(x2 - x1, 0) * max(y2 - y1, 0)


def distance_from_center(box: np.ndarray, image_size: tuple[int, int]) -> float:
    width, height = image_size
    return math.hypot((box[0] + box[2]) / 2 - width / 2, (box[1] + box[3]) / 2 - height / 2)


class FaceEmbedder:
    def __init__(self, device: torch.device | None = None):
        self.device = device or default_device()
        # Weights trained on VGGFace2 (3.3M photos, 9k identities); downloaded once (~107 MB).
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    @staticmethod
    def to_tensor(face: Image.Image) -> torch.Tensor:
        # Scale pixels from [0, 255] to roughly [-1, 1], the range the weights were trained on.
        array = np.asarray(face, dtype=np.float32)
        return (torch.from_numpy(array).permute(2, 0, 1) - 127.5) / 128.0

    @torch.inference_mode()
    def embed(self, faces: list[Image.Image], batch_size: int = 128) -> np.ndarray:
        """Return an (N, 512) array of L2-normalized embeddings."""
        chunks = []
        for start in range(0, len(faces), batch_size):
            batch = torch.stack([self.to_tensor(f) for f in faces[start:start + batch_size]]).to(self.device)
            chunks.append(self.model(batch).cpu().numpy())
        return np.concatenate(chunks) if chunks else np.empty((0, 512), dtype=np.float32)
