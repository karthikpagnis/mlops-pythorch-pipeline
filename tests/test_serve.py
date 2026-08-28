import io
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import serve


def test_health_reports_loaded_model(monkeypatch):
    monkeypatch.setattr(serve, "model", object())
    client = serve.app.test_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "healthy", "model_loaded": True}


def test_predict_requires_image(monkeypatch):
    monkeypatch.setattr(serve, "model", object())
    client = serve.app.test_client()

    response = client.post("/predict")

    assert response.status_code == 400
    assert "No image file provided" in response.get_json()["error"]


def test_predict_returns_class_probabilities(monkeypatch):
    class DummyModel:
        def __call__(self, inputs):
            return torch.zeros((inputs.shape[0], 10))

    monkeypatch.setattr(serve, "model", DummyModel())
    monkeypatch.setattr(serve, "device", torch.device("cpu"))
    image = Image.new("RGB", (32, 32), color="red")
    image_file = io.BytesIO()
    image.save(image_file, format="PNG")
    image_file.seek(0)
    client = serve.app.test_client()

    response = client.post(
        "/predict",
        data={"image": (image_file, "test.png")},
        content_type="multipart/form-data",
    )

    payload = response.get_json()
    assert response.status_code == 200
    assert payload["predicted_class"] == "airplane"
    assert len(payload["probabilities"]) == 10
