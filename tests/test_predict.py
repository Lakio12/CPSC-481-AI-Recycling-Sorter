
import pytest
import torch
from PIL import Image
from torch import nn

import predict


CLASSES = [
    "cardboard", "glass", "metal",
    "paper", "plastic", "trash"
]


class FakeModel(nn.Module):
    """Small model that returns predictable class scores."""

    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(6))

    def forward(self, x):
        assert x.shape[1:] == (3, 224, 224)
        scores = torch.zeros(x.shape[0], 6)
        scores[:, 4] = 5.0
        return scores.to(x.device) + self.bias


@pytest.fixture
def model():
    return FakeModel().eval()


@pytest.fixture
def sample_image(tmp_path):
    path = tmp_path / "plastic.jpg"
    Image.new("RGB", (300, 300), "blue").save(path)
    return path


def test_predict_returns_category(model, sample_image):
    category, confidence = predict.predict_image(
        sample_image, model, CLASSES, "cpu"
    )

    assert category == "plastic"
    assert isinstance(confidence, float)


def test_confidence_between_zero_and_one(model, sample_image):
    category, confidence = predict.predict_image(
        sample_image, model, CLASSES, "cpu"
    )

    assert 0.0 <= confidence <= 1.0


def test_predict_accepts_pil_image(model):
    image = Image.new("RGB", (300, 300), "red")

    category, confidence = predict.predict_image(
        image, model, CLASSES, "cpu"
    )

    assert category == "plastic"
    assert confidence > 0.0


def test_predict_converts_grayscale(model):
    image = Image.new("L", (300, 300), 128)

    category, confidence = predict.predict_image(
        image, model, CLASSES, "cpu"
    )

    assert category == "plastic"


def test_predict_rejects_invalid_input(model):
    with pytest.raises(TypeError):
        predict.predict_image(
            123, model, CLASSES, "cpu"
        )


def test_load_model_missing_file(tmp_path):
    missing = tmp_path / "missing.pth"

    with pytest.raises(FileNotFoundError):
        predict.load_model(missing)
