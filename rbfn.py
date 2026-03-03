import torch 
import torch.nn as nn
import torch.optim as optim 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 3 farklı sınıfa ait sınıflandırma problemi
df = pd.read_csv('iris.data')

X = df.iloc[:, :-1].values
y,_ = pd.factorize(df.iloc[:, -1])

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def to_tensor(data,target):
    data_tensor = torch.tensor(data, dtype=torch.float32)
    target_tensor = torch.tensor(target, dtype=torch.long)
    return data_tensor, target_tensor

X_train, y_train = to_tensor(X_train, y_train)
X_test, y_test = to_tensor(X_test, y_test)

rbf_kernel = lambda X,centers,beta: torch.exp(-beta * torch.cdist(X, centers)**2)

class RBFN(nn.Module):
    def __init__(self, input_dim, num_centers, output_dim):
        super(RBFN, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_centers, input_dim))
        self.beta = nn.Parameter(torch.ones(1)**2)
        self.linear = nn.Linear(num_centers, output_dim)

    def forward(self, x):
        phi = rbf_kernel(x, self.centers, self.beta)
        out = self.linear(phi)
        return out
    

model = RBFN(input_dim=4, num_centers=10, output_dim=3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

with torch.no_grad():
    model.eval()
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = (predicted == y_test).float().mean()
    print(f'Test Accuracy: {accuracy.item():.4f}')
    