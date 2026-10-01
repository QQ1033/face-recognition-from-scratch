# Facial Recognition from Scratch

A three-step deep learning project that builds up to a working facial recognition
system. It modernizes Cole Murray's 2017 tutorial series (TensorFlow 1 / Keras)
on a current PyTorch stack.

| Step | Folder | What it covers | Status |
|------|--------|----------------|--------|
| 1 | `step1_cnn/` | CNN fundamentals: build and train a CNN on MNIST, CPU vs GPU | Done |
| 2 | `step2_serving/` | Serve a pre-trained image classifier behind a REST API in Docker | Planned |
| 3 | `step3_face_recognition/` | Face detection + alignment, FaceNet embeddings, SVM classifier | Planned |

## Setup

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Step 1: CNN on MNIST

Architecture (`step1_cnn/model.py`): two 5x5 conv layers (32, 64 filters), each
followed by ReLU and 2x2 max pooling, then a 1024-unit fully connected layer with
dropout and a 10-way output.

```powershell
python -m step1_cnn.train --epochs 5      # add --device cpu to compare speed
python -m step1_cnn.evaluate              # accuracy on the 10k held-out test set
tensorboard --logdir runs                 # loss / accuracy curves at localhost:6006
```

Results (RTX 5070, 5 epochs, batch size 128, Adam lr=1e-3):

| | |
|---|---|
| Test accuracy | **99.19%** |
| Time per epoch, GPU | ~14 s |
| Time per epoch, CPU | ~32 s |

The original tutorial reported ~98.0% after 10k iterations of SGD.
