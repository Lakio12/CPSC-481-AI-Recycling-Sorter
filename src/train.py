"""
Training the EfficientNetB0 Model



Run:
python train.py
"""

# Imports
# =================================================
from dataset import get_dataloaders

train_loader, val_loader, test_loader, classes = get_dataloaders()
# Configuration
# =================================================

# Funcitons
# =================================================
def build_model(num_classes = 6):
    # Load in the pretrained EfficeintNet-B0 
    # and modify the classifier 
    pass

def train_model():
    # Train the model and validate
    # each of the epoch
    pass

def save_model(model, path):
    # Save trained model weights to the model folder
    pass

# Main 
# =================================================
if __name__ == "__main__":
    train_model
