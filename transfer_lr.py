import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.models as models
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


train_dataset = datasets.Flowers102(root='data', split='train', transform=transform_train, download=True)
test_dataset = datasets.Flowers102(root='data', split='val', transform=transform_test, download=True)

indices = torch.randint(len(train_dataset),(5,))
samples = [train_dataset[i] for i in indices]

fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i, (image, label) in enumerate(samples):
    image = image.numpy().transpose((1, 2, 0))
    image = (image * 0.5) + 0.5
    axes[i].imshow(image)
    axes[i].set_title(f'Label: {label}')
    axes[i].axis('off')
plt.show()

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)