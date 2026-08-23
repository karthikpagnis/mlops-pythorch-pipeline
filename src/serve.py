"""
Model serving API for CIFAR-10 predictions.

Flask app exposing:
- POST /predict: Accepts an image file, returns class probabilities.
- GET /health: Returns 200 if the model is loaded and ready.
"""

import io
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from flask import Flask, jsonify, request
from PIL import Image
from torchvision import transforms

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import get_model

app = Flask(__name__)

# CIFAR-10 class names
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# CIFAR-10 normalization constants
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# Global model reference
model = None
device = None


def get_inference_transform() -> transforms.Compose:
    """Return the transform pipeline for inference images."""
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])


def load_model():
    """Load the trained model from the checkpoint directory."""
    global model, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Look for checkpoint
    checkpoint_dir = os.environ.get("CHECKPOINT_DIR", "/app/checkpoints")
    model_name = os.environ.get("MODEL_NAME", "classifier_v1.pt")
    checkpoint_path = Path(checkpoint_dir) / model_name

    if not checkpoint_path.exists():
        # Try local path
        checkpoint_path = Path("checkpoints") / model_name

    if not checkpoint_path.exists():
        print(f"WARNING: No checkpoint found at {checkpoint_path}", flush=True)
        return False

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    architecture = checkpoint.get("architecture", "resnet18")
    num_classes = checkpoint.get("num_classes", 10)

    model = get_model(architecture=architecture, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(json.dumps({
        "event": "model_loaded",
        "checkpoint": str(checkpoint_path),
        "architecture": architecture,
        "val_accuracy": checkpoint.get("val_accuracy"),
    }), flush=True)

    return True


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint. Returns 200 if model is loaded."""
    if model is not None:
        return jsonify({"status": "healthy", "model_loaded": True}), 200
    return jsonify({"status": "unhealthy", "model_loaded": False}), 503


@app.route("/predict", methods=["POST"])
def predict():
    """
    Prediction endpoint.

    Accepts a multipart form with an 'image' file field.
    Returns predicted class and probabilities for all 10 CIFAR-10 classes.
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Use 'image' field."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        # Read and preprocess the image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        transform = get_inference_transform()
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)

        probs = probabilities[0].cpu().tolist()
        predicted_idx = int(torch.argmax(probabilities, dim=1).item())

        return jsonify({
            "predicted_class": CIFAR10_CLASSES[predicted_idx],
            "predicted_index": predicted_idx,
            "confidence": round(probs[predicted_idx], 4),
            "probabilities": {
                CIFAR10_CLASSES[i]: round(p, 4) for i, p in enumerate(probs)
            },
        })

    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 400


# Load model on startup
load_model()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
