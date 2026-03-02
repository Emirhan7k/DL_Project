import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


def get_data_loaders(batch_size: int = 64) -> tuple:
   
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data1', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data1', train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def imshow(img):
    """Display a normalized image."""
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def get_samples(loader):
    """Get a batch of samples from the loader."""
    data_iter = iter(loader)
    images, labels = next(data_iter)
    return images, labels


def visualize_samples(n: int = 5):
    train_loader, _ = get_data_loaders()
    images, labels = get_samples(train_loader)

    plt.figure(figsize=(12, 3))

    for i in range(n):
        plt.subplot(1, n, i + 1)
        imshow(images[i])
        plt.title(f"Label: {labels[i].item()}", fontsize=11)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


class CNN(nn.Module):
    """Convolutional Neural Network for CIFAR-10 classification."""
    
    def __init__(self):
        """Initialize the CNN layers."""
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """Forward pass through the CNN."""
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def create_loss_and_optimizer(model, lr: float = 0.001) -> tuple:
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    return criterion, optimizer


def train_model(model, train_loader, criterion, optimizer, num_epochs: int = 5, device: str = 'cpu') -> list:
    
    model.train()
    train_losses = []
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    # Plot training losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, marker='o', linestyle='-', 
             color='blue', linewidth=2, markersize=6, label='Training Loss')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss over Epochs', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return train_losses


def evaluate_model(model, test_loader, device: str = 'cpu') -> float:
   
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'\nAccuracy: {accuracy:.2f}%')
    return accuracy


def main():
    """Main execution function."""
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}\n')
    
    # Load data
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_data_loaders(batch_size=64)
    
    # Visualize samples
    print("Visualizing sample images...")
    visualize_samples(n=5)
    
    # Initialize model
    model = CNN().to(device)
    
    # Loss and optimizer
    criterion, optimizer = create_loss_and_optimizer(model, lr=0.001)
    
    # Train model
    print("\nTraining model...")
    train_model(model, train_loader, criterion, optimizer, num_epochs=5, device=device)
    
    # Evaluate model
    print("Evaluating model on test dataset...")
    evaluate_model(model, test_loader, device=device)
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()