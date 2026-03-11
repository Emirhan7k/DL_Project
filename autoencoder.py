import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

transforms = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.FashionMNIST(root='./data', train = True, transform=transforms, download=True)
test_dataset = datasets.FashionMNIST(root='./data', train = False, transform=transforms, download=True)

batch_size = 128

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 28*28),
            nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28))
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-3):
        self.patience = patience 
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None

    def __call__(self, loss):
        if self.best_loss is None or loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            return True
        return False
    
epochs = 50
learning_rate = 1e-3

model = AutoEncoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
early_stopping = EarlyStopping(patience=5, min_delta=1e-3)


def training_model(model, train_loader, criterion, optimizer, early_stopping):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for input, _ in train_loader:
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, input)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.5f}')
        if early_stopping(avg_loss):
            print("Early stopping triggered")
            break

training_model(model, train_loader, criterion, optimizer, early_stopping)

def compute_ssim(img1, img2,sigma=1):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = gaussian_filter(img1, sigma=sigma)
    mu2 = gaussian_filter(img2, sigma=sigma)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = gaussian_filter(img1 ** 2, sigma=sigma) - mu1_sq
    sigma2_sq = gaussian_filter(img2 ** 2, sigma=sigma) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, sigma=sigma) - mu1_mu2

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map

def evaluate_model(model, test_loader, n_images=10):
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            input, _ = batch
            output = model(input)
            break
    
    input = input.numpy()
    output = output.numpy()
    fig, axes = plt.subplots(2, n_images, figsize=(n_images*2, 4))
    ssim_scores = []

    for i in range(n_images):
        img1 = np.squeeze(input[i])
        img2 = np.squeeze(output[i])

        ssim_score = compute_ssim(img1, img2)
        ssim_scores.append(ssim_score)

        axes[0, i].imshow(img1, cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(img2, cmap='gray')
        axes[1, i].axis('off')

    axes[0, 0].set_title('Original Images')
    axes[1, 0].set_title('Reconstructed Images')
    plt.suptitle(f'Average SSIM: {np.mean(ssim_scores):.4f}', fontsize=16)
    plt.show()

evaluate_model(model, test_loader, n_images=10)