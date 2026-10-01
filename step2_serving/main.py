"""FastAPI service that classifies uploaded images.

Run locally:
    uvicorn step2_serving.main:app --reload
Then open http://localhost:8000 (upload page) or http://localhost:8000/docs (API docs).
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from PIL import UnidentifiedImageError

from step2_serving.classifier import ImageClassifier

MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # same 8 MB limit as the original tutorial
STATIC_DIR = Path(__file__).resolve().parent / "static"

classifier: ImageClassifier | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once at startup, not on every request -- loading takes seconds,
    # a prediction takes milliseconds.
    global classifier
    classifier = ImageClassifier()
    yield


app = FastAPI(title="Image Classifier", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "device": str(classifier.device)}


@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = Query(3, ge=1, le=10)):
    image_bytes = await file.read()
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image larger than 8 MB")
    try:
        predictions = classifier.predict(image_bytes, top_k)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="File is not a valid image")
    return {"filename": file.filename, "predictions": predictions}
