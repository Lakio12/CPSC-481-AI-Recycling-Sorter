
"""
Evaluate the EfficientNet-B0 Model

Run:
python evaluate.py

Output: 
Test Accuracy: 0.9474

Classification Report:
              precision    recall  f1-score   support

   cardboard       0.98      0.93      0.96        60
       glass       0.96      0.96      0.96        76
       metal       0.90      1.00      0.95        62
       paper       0.93      0.97      0.95        89
     plastic       0.98      0.88      0.93        73
       trash       0.90      0.95      0.93        20

    accuracy                           0.95       380
   macro avg       0.94      0.95      0.94       380
weighted avg       0.95      0.95      0.95       380
"""

# Imports
# =================================================

import torch
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from torchvision import models
from dataset import get_dataloaders

# Configurations
# =================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = Path(__file__).resolve().parent.parent / "path" / "efficientnet_b0_trashnet.pth"
REPORT_DIR = Path(__file__).resolve().parent.parent / "Reports"

# Functions
# =================================================

def load_model(num_classes):
    """Load the trained EfficientNet-B0 model."""

    # Create EfficientNet-B0 architecture
    model = models.efficientnet_b0(weights=None)

    # Modify classifier for TrashNet classes
    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features,
        num_classes
    )

    # Load checkpoint
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    # Extract trained model weights
    model.load_state_dict(checkpoint["model_state"])

    model = model.to(DEVICE)
    model.eval()

    return model

def evaluate_model():
    """Run predictions on the test dataset and calculate metrics."""

    # Load testing data
    _, _, test_loader, classes = get_dataloaders()

    # Load trained model
    model = load_model(len(classes))

    predictions = []
    actual = []

    # Disable gradient calculations
    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)

            # Make predictions
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)

            # Store predictions and actual labels
            predictions.extend(predicted.cpu().tolist())
            actual.extend(labels.tolist())

    # Calculate accuracy
    accuracy = accuracy_score(actual, predictions)

    print(f"Test Accuracy: {accuracy:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(
        classification_report(
            actual,
            predictions,
            labels=list(range(len(classes))),
            target_names=classes,
            zero_division=0
        )
    )

    # Generate confusion matrix
    plot_confusion_matrix(actual, predictions, classes)


def plot_confusion_matrix(actual, predictions, classes):
    """Visualize classification performance across six classes."""

    cm = confusion_matrix(
        actual,
        predictions,
        labels=list(range(len(classes)))
    )

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title("EfficientNet-B0 Confusion Matrix")

    plt.tight_layout()

    # Save confusion matrix
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(REPORT_DIR / "confusion_matrix.png")

    plt.show()

# Main
# =================================================
if __name__ == "__main__":
    evaluate_model()
