import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

dataset = torchvision.datasets.MNIST(root='./data2', train=True, download=True, transform=transform)
 
data = torch.stack([dataset[i][0] for i in range(50000)]).to(device)


class SOM(nn.Module):
    def __init__(self, grid_size, input_dim, lr=0.5, sigma=None):
        super(SOM, self).__init__()
        
        self.grid_size = grid_size
        self.input_dim = input_dim
        self.lr_initial = lr
        self.sigma_initial = sigma if sigma else grid_size / 2
        
        # SOM ağırlıkları (grid_size x grid_size x input_dim)
        self.weights = nn.Parameter(
            torch.randn(grid_size, grid_size, input_dim),
            requires_grad=False
        )
        
        # Grid koordinatları
        self.locations = torch.tensor(
            [[i, j] for i in range(grid_size) for j in range(grid_size)]
        ).to(device)

    def forward(self, x):
        return self.find_bmu(x)

    def find_bmu(self, x):
        # x: (input_dim)
        diff = self.weights - x
        dist = torch.norm(diff, dim=2)
        bmu_idx = torch.argmin(dist)
        return np.unravel_index(bmu_idx.cpu(), (self.grid_size, self.grid_size))

    def update(self, x, bmu, epoch, total_epochs):
        lr = self.lr_initial * (1 - epoch / total_epochs)
        sigma = self.sigma_initial * (1 - epoch / total_epochs)
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                
                dist_sq = (i - bmu[0])**2 + (j - bmu[1])**2
                neighborhood = torch.exp(
                    torch.tensor(-dist_sq / (2 * sigma**2), device=device)
                )
                
                self.weights[i, j] += lr * neighborhood * (x - self.weights[i, j])

    def train_som(self, data, epochs):
        data = data.to(device)
        self.weights.data = self.weights.data.to(device)
        
        for epoch in range(epochs):
            for x in data:
                bmu = self.find_bmu(x)
                self.update(x, bmu, epoch, epochs)
            
            print(f"Epoch {epoch+1}/{epochs} tamamlandı")

som = SOM(grid_size=20, input_dim=784)
som.train_som(data, epochs=10)


fig, axes = plt.subplots(20, 20, figsize=(10,10))

for i in range(20):
    for j in range(20):
        axes[i,j].imshow(
            som.weights[i,j].cpu().view(28,28),
            cmap="gray"
        )
        axes[i,j].axis("off")

plt.show()