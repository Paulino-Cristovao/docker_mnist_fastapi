"""MNIST CNN training module.

Trains a 4-layer CNN on the MNIST dataset with TensorBoard logging
and checkpoint saving capabilities.
"""

import os
from datetime import datetime

import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from tqdm import tqdm

def train_mnist_model() -> None:
    """
    Train a CNN model with 4 convolutional layers on the MNIST dataset.

    This function downloads the MNIST dataset (if not already present), defines a
    CNN architecture with 4 convolutional layers, and trains the model for a 
    specified number of epochs. The model's state dictionary is saved periodically
    and at the end of training.
    """
    # Set training parameters
    batch_size = 256
    epochs = 11  # Increase epochs as needed
    checkpoint_interval = 5
    learning_rate = 0.01

    # Use GPU if available for faster computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # TensorBoard setup
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("runs", f"mnist_{run_id}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be written to: {log_dir}")

    # Prepare the MNIST dataset with additional workers for faster data loading
    transform = transforms.Compose([transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    class MyCNN(nn.Module):
        """CNN model with 4 convolutional layers for MNIST classification."""
        def __init__(self) -> None:
            """Initialize the CNN model layers."""
            super().__init__()
            # Convolutional blocks
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

            # Non-linearities and pooling
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool2d(2)

            # Classifier
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
            x = self.pool(x)        # 28x28 -> 14x14
            x = self.relu(self.conv3(x))
            x = self.pool(x)        # 14x14 -> 7x7
            x = self.relu(self.conv4(x))
            x = self.flatten(x)
            x = self.fc(x)
            return x

    # instantiate the model
    model = MyCNN().to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop with informative tqdm progress bars
    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        processed = 0

        # outer epoch progress
        epoch_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{epochs}", unit="batch")

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # statistics for progress
            batch_loss = loss.item() * images.size(0)
            running_loss += batch_loss
            preds = outputs.argmax(dim=1)
            batch_correct = (preds == labels).sum().item()
            correct += batch_correct
            processed += labels.size(0)

            # log to TensorBoard per batch
            batch_acc = batch_correct / labels.size(0)
            writer.add_scalar('Loss/train_batch', loss.item(), global_step)
            writer.add_scalar('Accuracy/train_batch', batch_acc, global_step)
            # optionally log learning rate
            current_lr = optimizer.param_groups[0].get('lr', None)
            if current_lr is not None:
                writer.add_scalar('LR', current_lr, global_step)

            global_step += 1

            epoch_bar.update(1)
            epoch_bar.set_postfix({
                "avg_loss": f"{(running_loss/processed):.4f}",
                "acc": f"{(correct/processed):.4f}"
            })

        epoch_bar.close()

        # Save checkpoint after every `checkpoint_interval` epochs
        if epoch % checkpoint_interval == 0:
            checkpoint_path = f'./model_epoch_{epoch}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            tqdm.write(f"Checkpoint saved: {checkpoint_path}")

        # epoch summary
        epoch_loss = running_loss / processed if processed > 0 else 0.0
        epoch_acc = correct / processed if processed > 0 else 0.0
        tqdm.write(f"Epoch {epoch} completed — Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

        # log epoch metrics to TensorBoard
        writer.add_scalar('Loss/train_epoch', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train_epoch', epoch_acc, epoch)

    # Save the final trained model
    torch.save(model.state_dict(), './model_final.pth')
    print("Training complete. Final model saved.")

    # close TensorBoard writer
    writer.close()

if __name__ == "__main__":
    train_mnist_model()
