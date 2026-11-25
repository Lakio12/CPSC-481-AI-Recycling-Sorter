"""
Predict waste categories using the trained EfficientNet-B0 model.

Run: python src/predict.py path/to/image.jpg
"""

# Imports
# =================================================
import argparse
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import efficientnet_b0

# Configuration
# =================================================
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "Models" / "efficientnet_b0_trashnet.pth"

# Functions
# =================================================
def load_model(model_path=MODEL_PATH):
    """Load the trained EfficientNet-B0 model."""

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if not Path(model_path).is_file():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}"
        )

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )

    classes = checkpoint["classes"]

    model = efficientnet_b0(weights=None)

    # Match the number of output classes
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(
        in_features, len(classes)
    )

    # Restore trained weights
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()

    return model, classes, device


def predict_image(image, model, classes=None, device=None):
    """Preprocess an image and predict its waste category."""

    if classes is None:
        raise ValueError("Class labels are required.")

    if device is None:
        device = next(model.parameters()).device

    # Preprocessing must match the training pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Accept either a file path or a PIL image
    if isinstance(image, (str, Path)):
        with Image.open(image) as img:
            image = img.convert("RGB")
    elif isinstance(image, Image.Image):
        image = image.convert("RGB")
    else:
        raise TypeError("Image must be a file path or PIL image.")

    # Add batch dimension and move to device
    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)

    # Run inference without calculating gradients
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(
            probabilities, dim=1
        )

    category = classes[predicted.item()]
    confidence = confidence.item()

    return category, confidence


# Main
# =================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict a waste category from an image"
    )
    parser.add_argument("image", help="Path to the image")
    args = parser.parse_args()

    model, classes, device = load_model()

    category, confidence = predict_image(
        args.image, model, classes, device
    )

    print(f"Predicted category: {category}")
    print(f"Confidence: {confidence:.2%}")
