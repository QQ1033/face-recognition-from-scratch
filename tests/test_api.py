"""API tests for the image classification and face recognition service.

Run with:  python -m pytest
"""

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from step2_serving.main import app


@pytest.fixture(scope="module")
def client():
    # Using TestClient as a context manager runs the startup code (model loading).
    with TestClient(app) as c:
        yield c


def make_image_bytes(color=(255, 0, 0)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (300, 300), color).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_top_k_sorted(client):
    response = client.post("/predict?top_k=5", files={"file": ("red.jpg", make_image_bytes(), "image/jpeg")})
    assert response.status_code == 200

    predictions = response.json()["predictions"]
    assert len(predictions) == 5
    probabilities = [p["probability"] for p in predictions]
    assert probabilities == sorted(probabilities, reverse=True)
    assert all(0 <= p <= 1 for p in probabilities)


def test_predict_rejects_non_image(client):
    response = client.post("/predict", files={"file": ("notes.txt", b"not an image", "text/plain")})
    assert response.status_code == 400


def test_people_lists_trained_identities(client):
    names = client.get("/people").json()
    assert len(names) == 158
    assert "George W Bush" in names


def test_recognize_blank_image_finds_no_faces(client):
    response = client.post("/recognize", files={"file": ("red.jpg", make_image_bytes(), "image/jpeg")})
    assert response.status_code == 200
    assert response.json()["faces"] == []


def test_recognize_rejects_non_image(client):
    response = client.post("/recognize", files={"file": ("notes.txt", b"not an image", "text/plain")})
    assert response.status_code == 400


FORD_PHOTO = Path(__file__).resolve().parent.parent / "data/lfw_home/lfw_funneled/Harrison_Ford/Harrison_Ford_0004.jpg"


@pytest.mark.skipif(not FORD_PHOTO.exists(), reason="LFW not downloaded")
def test_recognize_large_photo_returns_boxes_in_original_pixels(client):
    # Upscale 250 -> 2500 px so the API has to shrink it for detection and scale boxes back.
    big = Image.open(FORD_PHOTO).resize((2500, 2500))
    buffer = io.BytesIO()
    big.save(buffer, format="JPEG")
    body = client.post("/recognize", files={"file": ("ford.jpg", buffer.getvalue(), "image/jpeg")}).json()

    assert (body["width"], body["height"]) == (2500, 2500)
    ford = next(f for f in body["faces"] if f["name"] == "Harrison Ford")
    x1, y1, x2, y2 = ford["box"]
    # The face is at roughly (69, 59)-(167, 193) in the 250 px original, i.e. 10x that here.
    assert abs(x1 - 690) < 60 and abs(x2 - 1670) < 60
    # The bystander's face is cut off by the right edge; her box must be clipped to the image.
    assert all(f["box"][2] <= 2500 and f["box"][3] <= 2500 for f in body["faces"])
