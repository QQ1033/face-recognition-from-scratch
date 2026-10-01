"""FastAPI service: ImageNet object classification (step 2) and face recognition (step 3).

Run locally:
    uvicorn step2_serving.main:app --reload
Then open http://localhost:8000 (upload page) or http://localhost:8000/docs (API docs).
"""

import io
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageOps, UnidentifiedImageError

from step2_serving.classifier import ImageClassifier
from step3_face_recognition.predict import FaceRecognizer

MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # same 8 MB limit as the original tutorial
# Face detection time grows with image size; phone photos (4000 px+) are shrunk to
# this before detection. Faces in a 1280 px photo are still plenty large for MTCNN.
MAX_FACE_IMAGE_SIDE = 1280
STATIC_DIR = Path(__file__).resolve().parent / "static"

classifier: ImageClassifier | None = None
recognizer: FaceRecognizer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the models once at startup, not on every request -- loading takes seconds,
    # a prediction takes milliseconds.
    global classifier, recognizer
    classifier = ImageClassifier()
    recognizer = FaceRecognizer()
    yield


app = FastAPI(title="Image Classifier & Face Recognizer", lifespan=lifespan)


async def read_upload(file: UploadFile) -> bytes:
    image_bytes = await file.read()
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image larger than 8 MB")
    return image_bytes


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "device": str(classifier.device)}


@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = Query(3, ge=1, le=10)):
    image_bytes = await read_upload(file)
    try:
        predictions = classifier.predict(image_bytes, top_k)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="File is not a valid image")
    return {"filename": file.filename, "predictions": predictions}


@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    """Find every face and name the ones the model knows; returns boxes in original-image pixels."""
    image_bytes = await read_upload(file)
    try:
        # Phones store photos sideways plus an EXIF "rotate me" tag; apply it so faces are upright.
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="File is not a valid image")

    width, height = image.size
    scale = min(1.0, MAX_FACE_IMAGE_SIDE / max(width, height))
    if scale < 1:
        image = image.resize((round(width * scale), round(height * scale)), Image.BILINEAR)

    faces = recognizer.recognize(image)
    for face in faces:
        # Scale back to original pixels, and clip: MTCNN's boxes extend past the
        # edge for faces cut off by the frame.
        x1, y1, x2, y2 = (v / scale for v in face["box"])
        face["box"] = [round(max(x1, 0)), round(max(y1, 0)), round(min(x2, width)), round(min(y2, height))]
        face["name"] = face["name"].replace("_", " ")
    return {"filename": file.filename, "width": width, "height": height, "faces": faces}


@app.get("/people")
def people():
    """The people the face recognizer was trained on."""
    return sorted(name.replace("_", " ") for name in recognizer.classifier.classes_)
