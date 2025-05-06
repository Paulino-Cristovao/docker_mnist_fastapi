# inference.py
import os
import glob
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io

# Search for checkpoint files matching "model_epoch_*.pth"
checkpoint_files = glob.glob('./model_epoch_*.pth')

if checkpoint_files:
    # Extract epoch number from filename and choose the checkpoint with the highest epoch
    def get_epoch(checkpoint_path: str) -> int:
        # Expected filename format: model_epoch_{epoch}.pth
        filename = os.path.basename(checkpoint_path)
        epoch_str = filename.split('_')[-1].split('.')[0]
        return int(epoch_str)

    last_checkpoint = max(checkpoint_files, key=get_epoch)
    checkpoint_path = last_checkpoint
    print(f"Loading last checkpoint: {checkpoint_path}")
else:
    checkpoint_path = './model_final.pth'
    print(f"No checkpoint found; loading final model from: {checkpoint_path}")

# Recreate the model architecture (must match the training architecture)
model = nn.Sequential(
    nn.Conv2d(1, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),  # 28x28 -> 14x14
    nn.Conv2d(64, 128, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),  # 14x14 -> 7x7
    nn.Conv2d(128, 256, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(256 * 7 * 7, 10)
)

# Load the state dictionary into the model and set model to evaluation mode
state_dict = torch.load(checkpoint_path)
model.load_state_dict(state_dict)
model.eval()

def predict_image(image_bytes: bytes) -> int:
    """
    Predict the class label for the given image.

    This function handles both grayscale and color images. Any uploaded image is
    resized to 28x28 before further processing.

    Parameters:
        image_bytes (bytes): The raw image data.

    Returns:
        int: The predicted class label.
    """
    image = Image.open(io.BytesIO(image_bytes))
    # Convert to grayscale if not already
    if image.mode != "L":
        image = image.convert("L")
    # Resize to 28x28 (the expected MNIST image size)
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
    prediction = output.argmax(dim=1).item()
    return prediction
