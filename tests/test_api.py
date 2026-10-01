"""API tests for the step 2 image classification service.

Run with:  python -m pytest
"""

import io

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
