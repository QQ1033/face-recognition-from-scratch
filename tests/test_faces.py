"""Tests for the face-selection geometry in step 3 (no models needed)."""

import numpy as np
from PIL import Image

from step3_face_recognition.faces import FACE_SIZE, FaceAligner, distance_from_center, visible_area


def test_visible_area_clips_boxes_that_leave_the_frame():
    # A face cut off by the right edge of a 250x250 photo: MTCNN reports the full
    # 122-px-wide box, but only 250 - 185 = 65 px of it is in the image.
    box = np.array([185, 118, 307, 252])
    assert visible_area(box, (250, 250)) == 65 * (250 - 118)


def test_center_face_beats_larger_edge_face():
    # The Harrison_Ford_0004 case: the subject is centered, a bystander's partly
    # visible face at the edge has the bigger box.
    subject = np.array([69, 59, 167, 193])
    bystander = np.array([185, 118, 307, 252])
    size = (250, 250)
    assert distance_from_center(subject, size) < distance_from_center(bystander, size)


def test_align_returns_square_face_crop_even_near_edges():
    image = Image.new("RGB", (250, 250), "gray")
    box = np.array([200, 200, 260, 260])
    landmarks = np.array([[215, 220], [245, 225], [230, 235], [218, 245], [242, 247]], dtype=np.float32)
    face = FaceAligner.align(image, box, landmarks)
    assert face.size == (FACE_SIZE, FACE_SIZE)
