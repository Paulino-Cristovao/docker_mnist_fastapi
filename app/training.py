# training.py - CNN with 4 Layers on MNIST
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

def train_mnist_model() -> None:
    """
    Train a CNN model with 4 convolutional layers on the MNIST dataset.

    This function downloads the MNIST dataset (if not already present), defines a
    CNN architecture with 4 convolutional layers using a sequential container,
    and trains the model for a specified number of epochs. After training, the model's
    state dictionary is saved to a file.

    Returns:
        None
    """
    # Set training parameters
    batch_size = 1024
    epochs = 11  # Increase epochs as needed
    checkpoint_interval = 5
    learning_rate = 0.01

    # Use GPU if available for faster computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare the MNIST dataset with additional workers for faster data loading
    transform = transforms.Compose([transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # CNN model with 4 convolutional layers
    model = nn.Sequential(
        # Layer 1: Conv layer
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        # Layer 2: Conv layer
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        # Pooling to reduce spatial dimensions
        nn.MaxPool2d(2),  # 28x28 -> 14x14
        
        # Layer 3: Conv layer
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        # Pooling to reduce spatial dimensions further
        nn.MaxPool2d(2),  # 14x14 -> 7x7
        
        # Layer 4: Conv layer
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(256 * 7 * 7, 10)
    ).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop with tqdm progress bars
    for epoch in tqdm(range(1, epochs + 1), desc="Epochs"):
        model.train()
        for images, labels in tqdm(train_loader, leave=False, desc="Batches"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Save checkpoint after every `checkpoint_interval` epochs
        if epoch % checkpoint_interval == 0:
            checkpoint_path = f'./model_epoch_{epoch}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    # Save the final trained model
    torch.save(model.state_dict(), './model_final.pth')
    print("Training complete. Final model saved.")

if __name__ == "__main__":
    train_mnist_model()
