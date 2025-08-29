"""MNIST model inference module.

Provides functionality for loading trained MNIST models and performing
prediction on image data.
"""

import os
import glob
import io
from typing import List

import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


class MyCNN(nn.Module):
    """CNN model architecture matching the training model.
    
    Contains 4 convolutional layers with ReLU activation and max pooling,
    followed by a fully connected layer for MNIST classification.
    """

    def __init__(self) -> None:
        """Initialize the CNN model layers."""
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256 * 7 * 7, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            Output tensor of shape (batch_size, 10)
        """
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)  # 28x28 -> 14x14
        x = self.relu(self.conv3(x))
        x = self.pool(x)  # 14x14 -> 7x7
        x = self.relu(self.conv4(x))
        x = self.flatten(x)
        x = self.fc(x)
        return x


def _find_checkpoint() -> str:
    """Find the latest checkpoint file if available, otherwise fallback to model_final.pth.
    
    Returns:
        Path to the most recent checkpoint file.
    """
    checkpoint_files = glob.glob("./model_epoch_*.pth")
    if checkpoint_files:
        def get_epoch(checkpoint_file: str) -> int:
            """Extract epoch number from checkpoint filename."""
            filename = os.path.basename(checkpoint_file)
            try:
                epoch_str = filename.split("_")[-1].split('.')[0]
                return int(epoch_str)
            except (ValueError, IndexError):
                return 0

        last_checkpoint = max(checkpoint_files, key=get_epoch)
        return last_checkpoint

    return "./model_final.pth"


def _load_state_dict(model_path: str, target_device: torch.device) -> dict:
    """Load a state dict with map_location and attempt to fix common prefix issues.
    
    Args:
        model_path: Path to the model file.
        target_device: Device to load the model on.
        
    Returns:
        Cleaned state dictionary ready for loading.
    """
    state = torch.load(model_path, map_location=target_device, weights_only=True)
    if not isinstance(state, dict):
        # if someone saved the model directly
        return state

    # if saved as a checkpoint with 'state_dict' key, use it
    if 'state_dict' in state and isinstance(state['state_dict'], dict):
        state = state['state_dict']

    # remove common DataParallel prefix if present
    new_state = {}
    for k, v in state.items():
        new_key = k
        if k.startswith('module.'):
            new_key = k[len('module.'):]
        new_state[new_key] = v
    return new_state


# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Inference using device: {device}")

# Build the model and load weights
checkpoint_path = _find_checkpoint()
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")

print(f"Loading checkpoint: {checkpoint_path}")
model = MyCNN().to(device)
try:
    state_dict = _load_state_dict(checkpoint_path, device)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded and set to eval mode.")
except Exception as exc:  # pragma: no cover - fail loudly in inference
    raise RuntimeError(f"Failed to load model state from {checkpoint_path}: {exc}") from exc


# Preprocessing transform matching training
_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])


def _preprocess_image(image_bytes: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != "L":
        image = image.convert("L")
    tensor = _transform(image)
    return tensor.unsqueeze(0)  # add batch dim


def predict_image(image_bytes: bytes) -> int:
    """Predict a single image and return the predicted class index."""
    tensor = _preprocess_image(image_bytes).to(device)
    with torch.no_grad():
        output = model(tensor)
        pred = int(output.argmax(dim=1).item())
    return pred


def predict_images_batch(image_bytes_list: List[bytes], show_progress: bool = True) -> List[int]:
    """Predict a list of images. Uses a tqdm progress bar when show_progress is True.

    This helper is useful for CLI/batch inference scenarios and will be used by
    higher-level code if multiple images need to be processed.
    """
    preds: List[int] = []
    iterator = image_bytes_list
    if show_progress:
        iterator = tqdm(image_bytes_list, desc="Running inference", unit="img")

    with torch.no_grad():
        for img_bytes in iterator:
            tensor = _preprocess_image(img_bytes).to(device)
            out = model(tensor)
            preds.append(int(out.argmax(dim=1).item()))

    return preds


# Convenience: when this module is run directly, do a quick smoke-test on all PNG/JPG in data/
if __name__ == "__main__":
    sample_paths = glob.glob("data/*.[pjP][pnPNm]*")  # quick grab of images
    if not sample_paths:
        print("No sample images found in data/ to run inference on.")
    else:
        print(f"Running inference on {len(sample_paths)} sample images...")

        # Get individual predictions for each image
        for i, image_path in enumerate(sample_paths, 1):
            try:
                with open(image_path, 'rb') as f:
                    prediction = predict_image(f.read())
                    print(f"Image {i} ({os.path.basename(image_path)}): {prediction}")
            except (OSError, IOError) as e:
                print(f"Error processing image {i} ({os.path.basename(image_path)}): {e}")

        # batch run with progress bar
        imgs = []
        for p in sample_paths:
            try:
                with open(p, 'rb') as f:
                    imgs.append(f.read())
            except (OSError, IOError):
                continue
        results = predict_images_batch(imgs, show_progress=True)
        print("Batch predictions:", results)
