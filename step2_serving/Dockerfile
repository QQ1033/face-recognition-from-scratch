# Image classification API, CPU-only.
# Build from the project root:
#   docker build -t image-classifier -f step2_serving/Dockerfile .
#   docker run --rm -p 8000:8000 image-classifier
FROM python:3.12-slim

WORKDIR /app

# Install dependencies first so Docker caches this layer: editing app code later
# won't re-download PyTorch. The CPU wheels are ~5x smaller than the CUDA ones.
COPY step2_serving/requirements.txt .
RUN pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 \
        --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# Download the ResNet-50 weights at build time so containers start without network access.
RUN python -c "from torchvision.models import resnet50, ResNet50_Weights; resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)"

COPY step2_serving/ step2_serving/

# Run as a non-root user; the weights cache must be readable by it.
RUN useradd --create-home appuser && cp -r /root/.cache /home/appuser/.cache && chown -R appuser /home/appuser/.cache
USER appuser

EXPOSE 8000
CMD ["uvicorn", "step2_serving.main:app", "--host", "0.0.0.0", "--port", "8000"]
